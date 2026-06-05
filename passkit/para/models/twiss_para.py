from dataclasses import dataclass
from passkit.para.models.base_model import BaseModel


@dataclass
class TwissPara(BaseModel):

    twiss_file: str = ""

    error_file: str = ""

    interpolate: bool = False

    num_interp_slice: int = 100

    muz: float = 0.001

    dqx: float = 0.0

    dqy: float = 0.0