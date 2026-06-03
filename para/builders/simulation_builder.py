import json
from typing import Any, Union


class SimulationBuilder:

    def __init__(self):

        self.main_para = {}

        self.modules = {}

        self.sequence = {"Sequence": {}}

    def set_main_para(self, data):

        self.main_para = data

    def add_module(self, name, data):

        self.modules[name] = data

    def add_sequence(self, data):

        self.sequence["Sequence"].update(data)

    def build(self):

        result = {}

        result.update(self.main_para)

        for module in self.modules.values():

            result.update(module)

        result.update(self.sequence)

        return result
