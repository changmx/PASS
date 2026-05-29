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

import os


def main(save_path=None, save_name="beam0.json"):
    main_model = MainPara()

    inj_model = InjectionPara()

    # twiss_model = TwissPara(twiss_file="bring.tfs", )

    # sc_model = SpaceChargePara()

    builder = SimulationBuilder()

    builder.set_main_para(MainBuilder.build(main_model))

    builder.add_module("injection", InjectionBuilder.build(inj_model))

    # builder.add_module("spacecharge", SpaceChargeBuilder.build(sc_model))

    # seq, circum = TwissBuilder.build(twiss_model)

    # builder.add_sequence(seq)

    data = builder.build()

    if save_path is None:
        save_path = os.getcwd()
        save_path = os.path.join(save_path, "input")

    JsonExporter.export(data, os.path.join(save_path, save_name))
    
    logger.info(f"Simulation data exported to {os.path.join(save_path, save_name)}")


if __name__ == "__main__":
    main()
