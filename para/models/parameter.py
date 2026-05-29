from dataclasses import dataclass


@dataclass
class Parameter:

    value: any

    unit: str = ""

    display_unit: str = ""

    minimum: float = -1e30

    maximum: float = 1e30

    scientific: bool = False

    scientific_scale: str = ""

    description: str = ""

    choices: list | None = None

    editable: bool = True
