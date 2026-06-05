import os
import re
from setuptools import setup, find_packages
"""
`setup.py` is the build script for a Python package (cannot be renamed), defining the package's metadata 
and build information. By running `pip install -e .`, `setup.py` is executed, thereby installing your package.

Installing the package in editable mode with `pip install -e .` allows you to install your package into the 
current Python environment even if you do not intend to officially release it, saving you from manually managing paths.
    - `-e`: stands for `--editable`, indicating installation in editable mode.
    - `.`: refers to the current directory, which typically contains a Python package (i.e., the project root directory 
    containing `setup.py` or `pyproject.toml`).

What does editable mode do?
    - Normally, `pip install` copies your package into the `site-packages` directory of the Python environment. Subsequent 
    changes to the source code require a reinstallation to take effect. In contrast, `pip install -e .` creates a link 
    (a symbolic link or a `.pth` file) in `site-packages` pointing to your project source code directory.
    - Effect: Any changes you make to the source code (adding, modifying, or deleting files) take effect immediately without 
    needing to reinstall. This is ideal for development.

After installation, you can import your package from any path in a Python script:
    ```python
    import passkit                 # import the entire package
    from passkit import tool    # import a specific module
    from passkit.tool import moduleA  # import from a subpackage

Prerequisites
    - Same Python environment: If you install in a virtual environment, any script using the package must also be run in that same 
    virtual environment (i.e., activate the same virtual environment).
    - Valid package structure: Ensure that every directory that should be a package contains an __init__.py file (can be empty).

Uninstallation
    - Uninstallation is the same as for any regular Python library: run pip uninstall passkit.

Usage Instructions
    1. Execute the command in the root directory:
        pip install -e .
        When installing the dependent libraries, if the network is poor, you can manually specify a domestic mirror source:
        pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
"""


def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'config', 'config.h')
    if not os.path.exists(version_file):
        raise FileNotFoundError(f"Can't find the config.h file: {version_file}")
    with open(version_file, 'r') as f:
        content = f.read()

    def extract(macro):
        match = re.search(rf'#define\s+{macro}\s+(\d+)', content)
        if not match:
            raise ValueError(f"Can't find {macro}")
        return match.group(1)

    major = extract('PASS_VERSION_MAJOR')
    minor = extract('PASS_VERSION_MINOR')
    patch = extract('PASS_VERSION_PATCH')
    return f"{major}.{minor}.{patch}"


setup(
    name='passkit',  # package name
    version=get_version(),  # get version from config.h
    packages=find_packages(),
    description='Python analysis toolkit for PASS',
    install_requires=["numpy", "pandas", "scipy", "PySide6"],
)
