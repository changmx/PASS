from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pass-sim")
except PackageNotFoundError:
    __version__ = "unknown"

# Enforce explicit imports to avoid 'from module import *'
__all__ = []
