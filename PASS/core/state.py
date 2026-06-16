from dataclasses import dataclass


@dataclass
class State:

    turn: int = 0

    time: float = 0.0

    revolution: int = 0
