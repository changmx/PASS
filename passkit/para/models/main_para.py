from passkit.para.models.parameter import Parameter
from passkit.gui.core.logger import logger
import os
from pathlib import Path


class MainPara:

    def __init__(self):
        logger.info("Initializing main parameters...")
        self.beam_name = Parameter(
            value="proton",
            description="Beam Name",
        )

        self.num_proton = Parameter(
            value=1,
            description="Number of Protons per particle",
        )

        self.num_neutron = Parameter(
            value=0,
            description="Number of Neutrons per particle",
        )

        self.num_electron = Parameter(
            value=1,
            description="Number of Electrons per particle",
        )

        self.gamma_t = Parameter(
            value=7.635074035,
            description="Transition Gamma",
        )

        self.circumference = Parameter(
            value=569.1,
            unit="m",
            display_unit="m",
            description="Circumference",
        )

        self.num_turns = Parameter(
            value=100,
            description="Number of Turns",
        )

        self.num_gpus = Parameter(
            value=1,
            description="Number of GPUs",
        )

        self.device_id = Parameter(
            value=[0],
            description="GPU Device ID",
        )

        current_dir = str(Path(__file__).resolve().parent.parent.parent.parent)

        self.output_dir = Parameter(
            value=os.path.join(current_dir, "output"),
            description="Output Directory",
        )

        self.is_plot = Parameter(
            value=False,
            description="Whether to generate plots",
        )
