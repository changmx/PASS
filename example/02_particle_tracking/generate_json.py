import sys
from pathlib import Path
import os


def _find_root(start_dir, marker="para"):
    # Find the root directory containing the marker folder
    current = Path(start_dir).resolve()
    while current.parent != current:
        if (current / marker).is_dir():
            return current
        current = current.parent
    return start_dir  # Fallback to the original directory if not found


# Add the root directory to sys.path for module imports
_root = _find_root(__file__)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from para.models.main_para import MainPara
from para.models.injection_para import InjectionPara
from para.models.twiss_para import TwissPara
from para.models.sc_para import SpaceChargePara

from para.builders.main_builder import MainBuilder
from para.builders.injection_builder import InjectionBuilder
from para.builders.twiss_builder import TwissBuilder
from para.builders.sc_builder import SpaceChargeBuilder

from para.builders.simulation_builder import SimulationBuilder

from para.exporters.json_exporter import JsonExporter

from gui.core.logger import logger


def main(save_path=None, save_name="beam0.json"):

    logger.info("Starting simulation data generation...")

    # 1. Change the parameters in the models as needed

    main_model = MainPara()
    main_model.particle_name.value = "proton"

    inj_model = InjectionPara()

    # twiss_model = TwissPara(twiss_file="bring.tfs", )

    # sc_model = SpaceChargePara()

    # 2. Build the simulation data using the builders

    builder = SimulationBuilder()

    builder.set_main_para(MainBuilder.build(main_model))

    builder.add_module("injection", InjectionBuilder.build(inj_model))

    # builder.add_module("spacecharge", SpaceChargeBuilder.build(sc_model))

    # seq, circum = TwissBuilder.build(twiss_model)

    # builder.add_sequence(seq)

    data = builder.build()

    # 3. Export the data to a JSON file

    if save_path is None:
        save_path = os.getcwd()
        save_path = os.path.join(save_path, "input")

    JsonExporter.export(data, os.path.join(save_path, save_name))

    logger.info(f"Simulation data exported to {os.path.join(save_path, save_name)}")


if __name__ == "__main__":
    script_dir = str(Path(__file__).resolve().parent)
    main(save_path=script_dir, save_name="beam0.json")
