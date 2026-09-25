from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import tomllib


def _source_project_metadata():
    """Read metadata only from the checkout containing this imported package."""
    path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        with path.open("rb") as stream:
            project = tomllib.load(stream).get("project", {})
        if isinstance(project, dict) and project.get("name") == "pass-sim":
            return project
    except (OSError, ValueError):
        pass
    return {}


def _package_version():
    source_version = _source_project_metadata().get("version")
    if isinstance(source_version, str) and source_version.strip():
        return source_version
    try:
        return version("pass-sim")
    except PackageNotFoundError:
        return "unknown"


__version__ = _package_version()

# Enforce explicit imports to avoid 'from module import *'
__all__ = []
