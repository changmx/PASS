from para.models.parameter import Parameter


class SpaceChargePara:

    def __init__(self):

        self.enable = Parameter(
            value=True,
            description="Enable Space Charge",
        )

        self.num_slices = Parameter(
            value=100,
            minimum=1,
            maximum=100000,
            description="Number of Slices",
        )

        self.field_solver = Parameter(
            value="PIC_FD_CUDSS",
            choices=[
                "PIC_FD_CUDSS",
                "PIC_FFT",
                "PIC_OPENMP",
            ],
            description="Field Solver",
        )
