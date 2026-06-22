from abc import ABC, abstractmethod
from PASS.core.simulation import Simulation


class Command(ABC):
    _registry = {}  # class dict, save "CommandType"->subclass

    @classmethod
    def register(cls, cmd_type: str):

        def decorator(subclass):
            cls._registry[cmd_type.lower()] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, beam_id: int, cmd_dict: dict, sim: Simulation):
        data = cmd_dict.copy()
        cmd_type = data.pop("command", None)
        if cmd_type is None:
            raise ValueError("Missing 'command' field in sequence data")
        cmd_type = cmd_type.lower()
        if cmd_type not in cls._registry:
            raise ValueError(f"Unknown command type: {cmd_type}")
        return cls._registry[cmd_type](beam_id, sim, **data)

    @abstractmethod
    def execute_cpu(self, sim):
        pass

    @abstractmethod
    def execute_gpu(self, sim):
        pass

    # future
    def execute_ad(self, sim):
        raise NotImplementedError

    @abstractmethod
    def print(self):
        pass
