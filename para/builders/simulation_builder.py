import json
from typing import Any, Union


class SimulationBuilder:

    def __init__(self):

        self.main_para = {}

        self.modules = {}

    def set_main_para(self, data):

        self.main_para = data

    def add_module(self, name, data):

        self.modules[name] = data

    def build(self):

        result = {}

        result.update(self.main_para)

        for module in self.modules.values():

            result.update(module)

        return result

