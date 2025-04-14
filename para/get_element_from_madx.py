import re
from copy import deepcopy
import sys


class Element:
    def __init__(self, name):
        self.name = name
        self.S = 0.0  # S值只在sequence中设置


class SBend(Element):
    def __init__(self, name):
        super().__init__(name)
        self.l = 0.0
        self.angle = 0.0
        self.e1 = 0.0
        self.e2 = 0.0
        self.hgap = 0.0
        self.fint = 0.0
        self.fintx = 0.0


class RBend(Element):
    def __init__(self, name):
        super().__init__(name)
        self.l = 0.0
        self.angle = 0.0
        self.e1 = 0.0
        self.e2 = 0.0
        self.hgap = 0.0
        self.fint = 0.0
        self.fintx = 0.0


class Quadrupole(Element):
    def __init__(self, name):
        super().__init__(name)
        self.l = 0.0
        self.k1 = 0.0


class Sextupole(Element):
    def __init__(self, name):
        super().__init__(name)
        self.l = 0.0
        self.k2 = 0.0


class Marker(Element):
    pass


class_map = {
    "sbend": SBend,
    "rbend": RBend,
    "quadrupole": Quadrupole,
    "sextupole": Sextupole,
    "marker": Marker,
}


def parse_file(filepath):
    elements = {}
    sequence = []
    processing_sequence = False

    with open(filepath, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            if processing_sequence:
                if line.lower() == "endsequence;":
                    processing_sequence = False
                    continue

                line = re.sub(r"\s*=\s*", "=", line.rstrip(";"))  # 清理空格
                parts = line.split(",", 1)
                if len(parts) < 2:
                    print(f"Warning: Invalid sequence line '{line}'")
                    continue

                elem_name = parts[0].strip()
                at_part = parts[1].strip()
                if not at_part.startswith("at="):
                    print(f"Warning: Invalid at-part in '{line}'")
                    continue

                try:
                    s_value = float(at_part.split("=")[1])
                except:
                    print(f"Warning: Invalid S value in '{line}'")
                    continue

                if elem_name not in elements:
                    print(f"Warning: Undefined element '{elem_name}'")
                    continue

                elem_copy = deepcopy(elements[elem_name])
                elem_copy.S = s_value
                sequence.append(elem_copy)

            else:
                if line.lower().startswith("ring: sequence"):
                    processing_sequence = True
                    continue

                if not re.match(r"^\w+:", line):
                    continue  # 忽略非定义行

                name_part, rest = line.split(":", 1)
                name = name_part.strip()
                rest = rest.strip().rstrip(";")

                type_part, *params_part = rest.split(",", 1)
                elem_type = type_part.strip().lower()  # 类型名称大小写不敏感
                params_str = params_part[0].strip() if params_part else ""

                if elem_type not in class_map:
                    print(f"Warning: Unknown element type '{elem_type}'")
                    continue

                elem_class = class_map[elem_type]
                elem = elem_class(name)

                if params_str:
                    for param in params_str.split(","):
                        param = param.strip()
                        if not param:
                            continue

                        param = param.replace(":=", "=")  # 统一处理:=和=
                        if "=" not in param:
                            continue

                        key, value = param.split("=", 1)
                        key = key.strip()
                        value = value.strip()

                        try:
                            value = float(value)
                        except ValueError:
                            print(f"Error: Invalid value '{value}' for '{key}'")
                            sys.exit(1)

                        if not hasattr(elem, key):
                            print(f"Error: Attribute '{key}' not found in {elem_type}")
                            sys.exit(1)

                        setattr(elem, key, value)

                elements[name] = elem

    return sequence


def generate_element_json(filepath):

    sequence = parse_file(filepath)

    element_json = []

    for elem in sequence:
        element_dict = {}

        if isinstance(elem, Marker):
            continue
        elif isinstance(elem, SBend):
            element_dict = {
                str(elem.name)
                + "_"
                + str(elem.S): {
                    "S (m)": elem.S,
                    "Command": type(elem).__name__,
                    "l (m)": elem.l,
                    "angle (rad)": elem.angle,
                    "e1 (rad)": elem.e1,
                    "e2 (rad)": elem.e2,
                    "hgap (m)": elem.hgap,
                    "fint": elem.fint,
                    "fintx": elem.fintx,
                }
            }
        elif isinstance(elem, RBend):
            element_dict = {
                str(elem.name)
                + "_"
                + str(elem.S): {
                    "S (m)": elem.S,
                    "Command": type(elem).__name__,
                    "l (m)": elem.l,
                    "angle (rad)": elem.angle,
                    "e1 (rad)": elem.e1,
                    "e2 (rad)": elem.e2,
                    "hgap (m)": elem.hgap,
                    "fint": elem.fint,
                    "fintx": elem.fintx,
                }
            }
        elif isinstance(elem, Quadrupole):
            element_dict = {
                str(elem.name)
                + "_"
                + str(elem.S): {
                    "S (m)": elem.S,
                    "Command": type(elem).__name__,
                    "l (m)": elem.l,
                    "k1 (m^-2)": elem.k1,
                }
            }
        elif isinstance(elem, Sextupole):
            element_dict = {
                str(elem.name)
                + "_"
                + str(elem.S): {
                    "S (m)": elem.S,
                    "Command": type(elem).__name__,
                    "l (m)": elem.l,
                    "k2 (m^-3)": elem.k2,
                }
            }
        else:
            print(
                f"Warning: we don't support {elem.name} ({type(elem).__name__}) @ S={elem.S} now."
            )
            continue

        print(str(elem.name) + "_" + str(elem.S))
        element_json.append(element_dict)

    return element_json


def test_parse_file(filepath):
    sequence = parse_file(filepath)
    # for elem in sequence[:3]:
    for elem in sequence:
        print(f"{elem.name} ({type(elem).__name__}) @ S={elem.S}")
        if isinstance(elem, Quadrupole):
            print(f"  l={elem.l}, k1={elem.k1}")
        elif isinstance(elem, SBend):
            print(
                f"  l={elem.l}, angle={elem.angle}, e1={elem.e1}, e2={elem.e2}, fint={elem.fint}, fintx={elem.fintx}"
            )


if __name__ == "__main__":
    # test_parse_file(r"D:\AthenaLattice\SZA\v13\sza.seq")
    generate_element_json(r"D:\AthenaLattice\SZA\v13\sza.seq")
