from dataclasses import dataclass, field
from typing import Any


@dataclass
class DisableRule:
    # When the source parameter's value matches when_value, the target parameters will be disabled in GUI mode (editable=False).

    targets: list[str]  # Parameter names to disable when the rule is triggered

    when_value: Any = True  # Trigger value (default True, effective for boolean values)

    def should_enable(self, source_value: Any) -> bool:

        return source_value != self.when_value


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

    disables: list[DisableRule] = field(default_factory=list)
