import sys
from pathlib import Path
import os

from passkit.para.models.main_para import MainPara
from passkit.para.models.injection_para import InjectionPara, BunchPara
# from passkit.para.models.twiss_para import TwissPara
from passkit.para.models.sc_para import SpaceChargePara

from passkit.para.builders.main_builder import MainBuilder
from passkit.para.builders.injection_builder import InjectionBuilder
# from passkit.para.builders.twiss_builder import TwissBuilder
from passkit.para.builders.sc_builder import SpaceChargeBuilder

from passkit.para.builders.simulation_builder import SimulationBuilder

from passkit.para.exporters.json_exporter import JsonExporter

from passkit.gui.core.logger import logger


def main(save_dir=None, file_name="beam0.json"):

    try:

        logger.info("Starting simulation data generation...")

        # 1. Change the parameters in the models as needed

        main_model = MainPara()

        main_model.beam_name.value = "12C6+"
        main_model.num_proton.value = 6
        main_model.num_neutron.value = 6
        main_model.num_electron.value = 6
        main_model.num_turns.value = 1
        main_model.num_gpus.value = 1
        main_model.device_id.value = [0]
        main_model.output_dir.value = os.path.join(str(Path(__file__).resolve().parent), "output")
        main_model.is_plot.value = False

        inj_bunch0 = BunchPara()
        inj_bunch0.alpha_x.value = -2.614303952  # HIAF-BRing Twiss Parateters at the injection point
        inj_bunch0.beta_x.value = 17.56341783
        inj_bunch0.alpha_y.value = 1.57442348
        inj_bunch0.beta_y.value = 8.624482365
        inj_bunch0.emit_x.value = 200e-6
        inj_bunch0.emit_y.value = 100e-6
        inj_bunch0.injection_turn.value = 1
        inj_bunch0.injection_interval.value = 1
        inj_bunch0.kinetic_energy.value = 10e6
        inj_bunch0.num_real_particles.value = 1e11
        inj_bunch0.num_macro_articles.value = 1e5

        inj_model = InjectionPara(num_bunches=0)
        inj_model.bunches.append(inj_bunch0)

        # twiss_model = TwissPara(twiss_file="bring.tfs", )

        # sc_model = SpaceChargePara()

        # 2. Build the simulation data using the builders

        builder = SimulationBuilder()

        builder.set_main_para(MainBuilder.build(main_model))
        builder.add_sequence(InjectionBuilder.build(inj_model))

        # builder.add_module("spacecharge", SpaceChargeBuilder.build(sc_model))

        # seq, circum = TwissBuilder.build(twiss_model)

        # builder.add_sequence(seq)

        data = builder.build()

        # 3. Export the data to a JSON file

        if save_dir is None:
            save_dir = str(Path(__file__).resolve().parent)
            save_dir = os.path.join(save_dir, "input")

        JsonExporter.export(data, os.path.join(save_dir, file_name))

        logger.info(f"Simulation input file has been saved to {os.path.join(save_dir, file_name)}")

    except Exception as e:

        logger.error(f"Error during simulation data generation: {str(e)}")


if __name__ == "__main__":

    script_dir = str(Path(__file__).resolve().parent)

    main(save_dir=script_dir, file_name="beam0.json")
