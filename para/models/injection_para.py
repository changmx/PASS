from para.models.parameter import Parameter, DisableRule


class BunchPara:
    """Holds all injection parameters for a single bunch."""

    def __init__(self):

        self.kinetic_energy = Parameter(
            value=33.2e6,
            unit="eV",
            display_unit="MeV",
            description="Kinetic Energy per Nucleon",
        )

        self.num_real_particles = Parameter(
            value=1e11,
            scientific=True,
            scientific_scale="1e9",
            description="Number of Real Particles per Bunch",
        )

        self.num_macro_articles = Parameter(
            value=1e5,
            scientific=True,
            scientific_scale="1e3",
            description="Number of Macro Particles per Bunch",
        )

        self.is_load_from_file = Parameter(
            value=False,
            description="Whether Load distribution from file",
            disables=[
                DisableRule(
                    targets=[
                        "alpha_x", "alpha_y", "beta_x", "beta_y", "emit_x", "emit_y", "Dx", "Dpx", "sigma_z", "dp", "dist_trans", "dist_longi",
                        "rf_voltage", "rf_phase", "num_harmonics", "harmonic_id", "rf_delta_dist"
                    ],
                    when_value=True,
                ),
                DisableRule(targets=["file_path"], when_value=False),
            ],
        )

        self.file_path = Parameter(
            value="",
            description="File path for loading distribution",
        )

        self.injection_turn = Parameter(
            value=1,
            description="Total Injection Turns",
        )

        self.injection_interval = Parameter(
            value=1,
            description="Injection Interval",
        )

        self.alpha_x = Parameter(
            value=-2.614303952,
            description="Alpha X",
        )

        self.alpha_y = Parameter(
            value=1.57442348,
            description="Alpha Y",
        )

        self.beta_x = Parameter(
            value=0.5,
            unit="m",
            display_unit="m",
            description="Beta X",
        )

        self.beta_y = Parameter(
            value=0.5,
            unit="m",
            display_unit="m",
            description="Beta Y",
        )

        self.emit_x = Parameter(
            value=200e-6,
            unit="pi*m*rad",
            display_unit="pi*mm*mrad",
            description="Emit X",
        )

        self.emit_y = Parameter(
            value=100e-6,
            unit="pi*m*rad",
            display_unit="pi*mm*mrad",
            description="Emit Y",
        )

        self.Dx = Parameter(
            value=0.0,
            unit="m",
            display_unit="m",
            description="Dispersion X",
        )

        self.Dpx = Parameter(
            value=0.0,
            description="Dispersion Slope X",
        )

        self.sigma_z = Parameter(
            value=0.1,
            unit="m",
            display_unit="m",
            description="Bunch Length",
        )

        self.dp = Parameter(
            value=0.001,
            description="Momentum Spread",
            scientific=True,
            scientific_scale="1e-3",
        )

        self.dist_trans = Parameter(
            value="gaussian",
            description="Distribution in Transverse Phase Space",
            choices=["gaussian", "kv", "uniform"],
        )

        self.dist_longi = Parameter(
            value="gaussian",
            description="Distribution in Longitudinal Phase Space",
            choices=["gaussian", "coasting", "matchZ", "matchDp"],
            disables=[
                DisableRule(
                    targets=["rf_voltage", "rf_phase", "num_harmonics", "harmonic_id", "rf_delta_dist"],
                    when_value="gaussian",
                ),
                DisableRule(
                    targets=["rf_voltage", "rf_phase", "num_harmonics", "harmonic_id", "rf_delta_dist"],
                    when_value="coasting",
                ),
            ],
        )

        self.rf_voltage = Parameter(
            value=0.0,
            unit="V",
            display_unit="kV",
            description="RF Voltage",
        )
        self.rf_phase = Parameter(
            value=0.0,
            unit="rad",
            display_unit="rad",
            description="RF Phase",
        )

        self.num_harmonics = Parameter(
            value=1,
            description="Harmonic Number",
        )

        self.harmonic_id = Parameter(
            value=0,
            description="Harmonic ID of this bunch",
        )

        self.rf_delta_dist = Parameter(
            value=0.0,
            description="RF s position refer to injection point",
        )

        self.is_offset_x = Parameter(
            value=False,
            description="Enable X offset",
            disables=[
                DisableRule(
                    targets=["is_load_offset_x_from_file", "offset_x_filepath", "offset_x_file_timekind", "offset_x_position", "offset_x_momentum"],
                    when_value=False,
                ),
            ],
        )

        self.is_load_offset_x_from_file = Parameter(
            value=False,
            description="Whether load X offset time-dependent data from file",
            disables=[
                DisableRule(
                    targets=["offset_x_filepath", "offset_x_file_timekind"],
                    when_value=False,
                ),
                DisableRule(
                    targets=["offset_x_position", "offset_x_momentum"],
                    when_value=True,
                ),
            ],
        )

        self.offset_x_filepath = Parameter(
            value="",
            description="File path for X offset distribution",
        )

        self.offset_x_file_timekind = Parameter(
            value="turn",
            description="Time kind for X offset file",
            choices=["turn", "time"],
        )

        self.offset_x_position = Parameter(
            value=0.0,
            unit="m",
            display_unit="m",
            description="X offset position",
        )

        self.offset_x_momentum = Parameter(
            value=0.0,
            description="X offset momentum",
        )

        self.is_offset_y = Parameter(
            value=False,
            description="Enable Y offset",
            disables=[
                DisableRule(
                    targets=["is_load_offset_y_from_file", "offset_y_file_timekind", "offset_y_filepath", "offset_y_position", "offset_y_momentum"],
                    when_value=False,
                ),
            ],
        )

        self.is_load_offset_y_from_file = Parameter(
            value=False,
            description="Whether load Y offset time-dependent data from file",
            disables=[
                DisableRule(
                    targets=["offset_y_filepath", "offset_y_file_timekind"],
                    when_value=False,
                ),
                DisableRule(
                    targets=["offset_y_position", "offset_y_momentum"],
                    when_value=True,
                )
            ],
        )

        self.offset_y_filepath = Parameter(
            value="",
            description="File path for Y offset distribution",
        )

        self.offset_y_file_timekind = Parameter(
            value="turn",
            description="Time kind for Y offset file",
            choices=["turn", "time"],
        )

        self.offset_y_position = Parameter(
            value=0.0,
            unit="m",
            display_unit="m",
            description="Y offset position",
        )

        self.offset_y_momentum = Parameter(
            value=0.0,
            description="Y offset momentum",
        )

        self.save_init_dist = Parameter(
            value=False,
            description="Whether save initial distribution to file",
        )

        self.insert_particle = Parameter(
            value=[],  # List of [x, px, y, py, z, dp] for each particle
            description="Manually insert particles (x, px, y, py, z, dp)",
        )


class InjectionPara:
    """Model for the Injection section. Holds common settings + a list of bunches."""

    def __init__(self, num_bunches=1):
        self.s = Parameter(
            value=0.0,
            unit="m",
            display_unit="m",
            description="Injection Position (S)",
        )

        self.command = Parameter(
            value="Injection",
            description="Command",
        )

        self.bunches = []
        for i in range(num_bunches):
            self.bunches.append(BunchPara())
