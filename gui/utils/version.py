import re
from pathlib import Path

CONFIG_PATH = (Path(__file__).resolve().parents[2] / "config" / "config.h")


def read_pass_version():

    text = CONFIG_PATH.read_text()

    major = re.search(r"#define PASS_VERSION_MAJOR (\d+)", text)

    minor = re.search(r"#define PASS_VERSION_MINOR (\d+)", text)

    patch = re.search(r"#define PASS_VERSION_PATCH (\d+)", text)

    major = major.group(1)

    minor = minor.group(1)

    patch = patch.group(1)

    return f"{major}.{minor}.{patch}"
