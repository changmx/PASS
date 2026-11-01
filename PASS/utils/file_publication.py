"""Publish staged files without carrying private Windows staging permissions."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from uuid import uuid4


class FilePublicationRecoveryError(OSError):
    """Publication failed and the original remains in a named recovery file."""


def _copy_windows_dacl(source, target):
    """Copy access rules and inheritance state without requesting SACL privileges."""
    import ctypes
    from ctypes import wintypes

    security = ctypes.WinDLL("advapi32", use_last_error=True)
    get_security = security.GetFileSecurityW
    get_security.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    get_security.restype = wintypes.BOOL
    get_control = security.GetSecurityDescriptorControl
    get_control.argtypes = [wintypes.LPVOID, ctypes.POINTER(wintypes.WORD), ctypes.POINTER(wintypes.DWORD)]
    get_control.restype = wintypes.BOOL
    set_security = security.SetFileSecurityW
    set_security.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.LPVOID]
    set_security.restype = wintypes.BOOL
    needed = wintypes.DWORD()
    if not get_security(str(source.resolve()), 4, None, 0, ctypes.byref(needed)) and ctypes.get_last_error() != 122:
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor = ctypes.create_string_buffer(needed.value)
    if not get_security(str(source.resolve()), 4, descriptor, needed, ctypes.byref(needed)):
        raise ctypes.WinError(ctypes.get_last_error())
    control, revision = wintypes.WORD(), wintypes.DWORD()
    if not get_control(descriptor, ctypes.byref(control), ctypes.byref(revision)):
        raise ctypes.WinError(ctypes.get_last_error())
    information = 4 | (0x80000000 if control.value & 0x1000 else 0x20000000)
    if not set_security(str(target.resolve()), information, descriptor):
        raise ctypes.WinError(ctypes.get_last_error())


@contextmanager
def _publication_copy(source, target, permissions_from, check):
    # A sibling file inherits the destination directory, unlike mkdtemp(0700).
    fd, name = tempfile.mkstemp(prefix=".pass-publish-", suffix=".tmp", dir=target.parent)
    publication = Path(name)
    try:
        with os.fdopen(fd, "wb") as output:
            if permissions_from is not None:
                # Apply private destination rules before copying its replacement bytes.
                _copy_windows_dacl(permissions_from, publication)
            with source.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    if check is not None:
                        check()
                    output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if check is not None:
            check()
        yield publication
    finally:
        if publication.exists():
            publication.unlink()


def _replace_windows_file(source, target, check):
    import ctypes
    from ctypes import wintypes

    replace = ctypes.WinDLL("kernel32", use_last_error=True).ReplaceFileW
    replace.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD, wintypes.LPVOID, wintypes.LPVOID]
    replace.restype = wintypes.BOOL
    backup = target.parent / f".pass-replace-{uuid4().hex}.bak"
    with _publication_copy(source, target, target, check) as publication:
        try:
            # Do not ignore ACL merge failures. Keep recovery outside private staging.
            if not replace(str(target.resolve()), str(publication.resolve()), str(backup.resolve()), 0, None, None):
                raise ctypes.WinError(ctypes.get_last_error())
            backup.unlink()
        except OSError:
            # ReplaceFileW can fail after it has moved the original destination.
            if backup.exists():
                try:
                    os.replace(backup, target)
                except OSError as error:
                    raise FilePublicationRecoveryError(f"Could not restore {target}; original file preserved at {backup}") from error
            raise


def publish_file(source, target, *, overwrite=True, permissions_from=None, check=None):
    """Atomically publish bytes, preserving Windows destination access rules.

    New files inherit their destination directory; existing files retain their
    DACL. ``permissions_from`` supports callers that moved an original aside
    for rollback. Exclusive publication never replaces a concurrently created
    file. Windows uses one extra sibling copy; the staged source is retained.
    ``check`` may cancel before publication, including during the extra copy.
    """
    source, target = Path(source), Path(target)
    if check is not None:
        check()
    if os.name != "nt":
        if overwrite:
            os.replace(source, target)
        else:
            os.link(source, target)
        return
    if overwrite and target.is_symlink():
        # Preserve os.replace semantics: replace the link entry, not its referent.
        with _publication_copy(source, target, None, check) as publication:
            os.replace(publication, target)
        return
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {target}")
        _replace_windows_file(source, target, check)
        return
    permissions_from = Path(permissions_from) if permissions_from is not None else None
    with _publication_copy(source, target, permissions_from, check) as publication:
        try:
            # Windows rename is exclusive, including on filesystems without hard links.
            os.rename(publication, target)
        except FileExistsError:
            if not overwrite:
                raise
            # Preserve rules if another writer created the destination meanwhile.
            _replace_windows_file(publication, target, check)
