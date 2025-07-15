import re
from copy import deepcopy
import sys
import numpy as np


class Element:
    def __init__(self, name):
        self.name = name
        self.S = 0.0  # S值只在sequence中设置


class SBendElement(Element):
    def __init__(self, name):
        super().__init__(name)
        self.l = 0.0
        self.angle = 0.0
        self.e1 = 0.0
        self.e2 = 0.0
        self.hgap = 0.0
        self.fint = 0.0
        self.fintx = 0.0
        self.isFieldError = False


class RBendElement(Element):
    def __init__(self, name):
        super().__init__(name)
        self.l = 0.0
        self.angle = 0.0
        self.e1 = 0.0
        self.e2 = 0.0
        self.hgap = 0.0
        self.fint = 0.0
        self.fintx = 0.0
        self.isFieldError = False


class QuadrupoleElement(Element):
    def __init__(self, name):
        super().__init__(name)
        self.l = 0.0
        self.k1 = 0.0
        self.k1s = 0.0
        self.isFieldError = False


class Sextupole(Element):
    def __init__(self, name):
        super().__init__(name)
        self.l = 0.0
        self.k2 = 0.0
        self.k2s = 0.0
        self.isFieldError = False


class OctupoleElement(Element):
    def __init__(self, name):
        super().__init__(name)
        self.l = 0.0
        self.k3 = 0.0
        self.k3s = 0.0
        self.isFieldError = False


class HKickerElement(Element):
    def __init__(self, name):
        super().__init__(name)
        self.l = 0.0
        self.kick = 0.0
        self.sinkick = 0
        self.sinpeak = 0
        self.sintune = 0
        self.sinphase = 0
        self.isFieldError = False


class VKickerElement(Element):
    def __init__(self, name):
        super().__init__(name)
        self.l = 0.0
        self.kick = 0.0
        self.sinkick = 0
        self.sinpeak = 0
        self.sintune = 0
        self.sinphase = 0
        self.isFieldError = False


class MarkerElement(Element):
    def __init__(self, name):
        super().__init__(name)
        self.l = 0.0


class_map = {
    "sbend": SBendElement,
    "rbend": RBendElement,
    "quadrupole": QuadrupoleElement,
    "sextupole": Sextupole,
    "octupole": OctupoleElement,
    "hkicker": HKickerElement,
    "vkicker": VKickerElement,
    "marker": MarkerElement,
}


def parse_file(filepath):
    elements = {}
    sequence = []
    processing_sequence = False

    circumference = 0

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
                    print(f"Warning: Undefined element '{elem_name}' at S = {s_value}")
                    continue

                elem_copy = deepcopy(elements[elem_name])
                elem_copy.S = s_value
                sequence.append(elem_copy)

            else:
                if ":sequence" in line.lower().replace(" ", ""):
                    processing_sequence = True

                    match_circum = re.search(r"=\s*([+-]?\d+\.?\d*)", line.lower())
                    if match_circum:
                        circumference = float(match_circum.group(1))
                        print(f"Circumference (m) = {circumference}")
                    else:
                        print(f"Error: can't match sequence, l = circumference")
                        sys.exit(1)
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
                    print(f"Warning: Unknown element type '{elem_type}' of '{name}'")
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

    # Check and delete duplicate element data with the same name and the same position
    # This process is mainly aimed at addressing the issue where multiple MARKERs with the same name repeatedly occur at the same position
    seen = set()
    data_remove_duplication = []
    for elem in sequence:
        identifier = (elem.S, elem.name, type(elem).__name__)
        if identifier not in seen:
            seen.add(identifier)
            data_remove_duplication.append(elem)
        else:
            print(
                f"Delete duplication data: S = {elem.S}, name = {elem.name}, type = {type(elem).__name__}"
            )

    data_array = np.array(data_remove_duplication)

    return data_array, circumference


def generate_element_json(filepath):

    sequence, circumference = parse_file(filepath)

    element_json = []

    for i in np.arange(len(sequence)):
        elem = sequence[i]
        element_dict = {}

        s = elem.S
        s_previous = 0 if i == 0 else sequence[i - 1].S
        l = elem.l
        l_previous = 0 if i == 0 else sequence[i - 1].l
        drift_length = s - l / 2 - (s_previous + l_previous / 2)

        if isinstance(elem, MarkerElement):
            element_dict = {
                str(elem.name)
                + "_"
                + str(elem.S): {
                    "S (m)": elem.S,
                    "Command": type(elem).__name__,
                    "L (m)": elem.l,
                    "Drift length (m)": drift_length,
                }
            }
        elif isinstance(elem, SBendElement):
            element_dict = {
                str(elem.name)
                + "_"
                + str(elem.S): {
                    "S (m)": elem.S,
                    "Command": type(elem).__name__,
                    "L (m)": elem.l,
                    "Drift length (m)": drift_length,
                    "angle (rad)": elem.angle,
                    "e1 (rad)": elem.e1,
                    "e2 (rad)": elem.e2,
                    "hgap (m)": elem.hgap,
                    "fint": elem.fint,
                    "fintx": elem.fintx,
                    "isFieldError": elem.isFieldError,
                }
            }
        elif isinstance(elem, RBendElement):
            element_dict = {
                str(elem.name)
                + "_"
                + str(elem.S): {
                    "S (m)": elem.S,
                    "Command": type(elem).__name__,
                    "L (m)": elem.l,
                    "Drift length (m)": drift_length,
                    "angle (rad)": elem.angle,
                    "e1 (rad)": elem.e1,
                    "e2 (rad)": elem.e2,
                    "hgap (m)": elem.hgap,
                    "fint": elem.fint,
                    "fintx": elem.fintx,
                    "isFieldError": elem.isFieldError,
                }
            }
        elif isinstance(elem, QuadrupoleElement):
            element_dict = {
                str(elem.name)
                + "_"
                + str(elem.S): {
                    "S (m)": elem.S,
                    "Command": type(elem).__name__,
                    "L (m)": elem.l,
                    "Drift length (m)": drift_length,
                    "k1 (m^-2)": elem.k1,
                    "k1s (m^-2)": elem.k1s,
                    "isFieldError": elem.isFieldError,
                }
            }
        elif isinstance(elem, Sextupole):
            if (np.abs(elem.k2 > 1e-9) and np.abs(elem.k2s) < 1e-9):
                element_dict = {
                    str(elem.name)
                    + "_"
                    + str(elem.S): {
                        "S (m)": elem.S,
                        "Command": type(elem).__name__ + "NormElement",
                        "L (m)": elem.l,
                        "Drift length (m)": drift_length,
                        "k2 (m^-3)": elem.k2,
                        "isFieldError": elem.isFieldError,
                    }
                }
            elif (np.abs(elem.k2 < 1e-9) and np.abs(elem.k2s) > 1e-9):
                element_dict = {
                    str(elem.name)
                    + "_"
                    + str(elem.S): {
                        "S (m)": elem.S,
                        "Command": type(elem).__name__ + "SkewElement",
                        "L (m)": elem.l,
                        "Drift length (m)": drift_length,
                        "k2s (m^-3)": elem.k2s,
                        "isFieldError": elem.isFieldError,
                    }
                }
            elif (np.abs(elem.k2 < 1e-9) and np.abs(elem.k2s) < 1e-9):
                element_dict = {
                    str(elem.name)
                    + "_"
                    + str(elem.S): {
                        "S (m)": elem.S,
                        "Command": type(elem).__name__ + "NorwElement",
                        "L (m)": elem.l,
                        "Drift length (m)": drift_length,
                        "k2 (m^-3)": elem.k2,
                        "isFieldError": elem.isFieldError,
                    }
                }
            else:
                print(
                    f"Error: Sextupole: k2 = {elem.k2}, k2s = {elem.k2s}, there should be and only 1 variable equal to 0"
                )
                sys.exit(1)
        elif isinstance(elem, OctupoleElement):
            element_dict = {
                str(elem.name)
                + "_"
                + str(elem.S): {
                    "S (m)": elem.S,
                    "Command": type(elem).__name__,
                    "L (m)": elem.l,
                    "Drift length (m)": drift_length,
                    "k3 (m^-4)": elem.k3,
                    "k3s (m^-4)": elem.k3s,
                    "isFieldError": elem.isFieldError,
                }
            }
        else:
            print(
                f"Warning: we don't support {elem.name} ({type(elem).__name__}) @ S={elem.S} now."
            )
            continue

        # print(str(elem.name) + "_" + str(elem.S))
        element_json.append(element_dict)

    if (circumference - sequence[-1].S - sequence[-1].l / 2) > 1e-9:
        # 如果sequence最后一个元素的S不等于环周长，则人为补一个maker，使其位于环尾端
        elem = class_map["marker"]("ring_end")
        element_dict = {
            str(elem.name)
            + "_"
            + str(circumference): {
                "S (m)": circumference,
                "Command": type(elem).__name__,
                "L (m)": elem.l,
                "Drift length (m)": circumference - sequence[-1].S - sequence[-1].l / 2,
            }
        }
        element_json.append(element_dict)

    return element_json, circumference


def test_parse_file(filepath):
    sequence = parse_file(filepath)
    # for elem in sequence[:3]:
    for elem in sequence:
        print(f"{elem.name} ({type(elem).__name__}) @ S={elem.S}")
        if isinstance(elem, QuadrupoleElement):
            print(f"  l={elem.l}, k1={elem.k1}")
        elif isinstance(elem, SBendElement):
            print(
                f"  l={elem.l}, angle={elem.angle}, e1={elem.e1}, e2={elem.e2}, fint={elem.fint}, fintx={elem.fintx}"
            )


if __name__ == "__main__":
    # test_parse_file(r"D:\AthenaLattice\SZA\v13\sza.seq")
    generate_element_json(r"D:\AthenaLattice\SZA\v13\sza.seq")
