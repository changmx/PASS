from collections import OrderedDict

class_map = {
    "marker": "MarkerElement",
    "drift": "DriftElement",
    "sbend": "SBendElement",
    "rbend": "RBendElement",
    "quadrupole": "QuadrupoleElement",
    "sextupole": "SextupoleElement",
    "octupole": "OctupoleElement",
    "multipole": "MultipoleElement",
    "kicker": "KickerElement",
}


def convert_ordereddict(obj):
    if isinstance(obj, OrderedDict):
        return {k: convert_ordereddict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ordereddict(item) for item in obj]
    else:
        return obj


def sort_sequence(sequence):
    Command_order = {
        "Injection": 0,  # 最高优先级
        "SortBunch": 100,  # 次级优先级
        "Twiss": 200,
        "MarkerElement": 300,
        "DriftElement": 300,
        "SBendElement": 300,
        "RBendElement": 300,
        "QuadrupoleElement": 300,
        "SextupoleElement": 300,
        "OctupoleElement": 300,
        "MultipoleElement": 300,
        "KickerElement": 300,
        "RFElement": 300,
        "ElSeparatorElement": 300,
        "TuneExciterElement": 300,
        "SpaceCharge": 400,
        "WakeField": 500,
        "BeamBeam": 600,
        "ElectronCloud": 700,
        "LumiMonitor": 800,
        "PhaseMonitor": 800,
        "DistMonitor": 800,
        "StatMonitor": 800,
        "ParticleMonitor": 800,
        "Other": 999,  # 最低优先级
    }

    # 首先按照位置S进行从小到大的排序，如果两个字典的S相同，按照上述Commnad自定义顺序进行排序
    sorted_sequence = OrderedDict(
        sorted(
            sequence.items(),
            key=lambda item: (
                item[1]["S (m)"],  # 主排序键
                Command_order.get(item[1]["Command"], 999),  # 次排序键
            ),
        ))

    sorted_sequence_dictType = convert_ordereddict(sorted_sequence)  # 递归转换，把OrderDict转化为dict类型

    return sorted_sequence_dictType
