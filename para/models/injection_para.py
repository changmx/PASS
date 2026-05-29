from para.models.parameter import Parameter


class InjectionPara:

    def __init__(self):

        self.s = Parameter(
            value=0.0,
            unit="m",
            display_unit="m",
            description="Injection Position",
        )

        self.kinetic_energy = Parameter(
            value=33.2e6,
            unit="eV",
            display_unit="MeV",
            description="Kinetic Energy",
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
            description="Number of Macroparticles per Bunch",
        )

        self.injection_turn = Parameter(
            value=0,
            description="Injection Turns",
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
