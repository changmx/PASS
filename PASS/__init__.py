from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("PASS")
except PackageNotFoundError:
    __version__ = "unknown"

# Enforce explicit imports to avoid 'from module import *'
__all__ = []
