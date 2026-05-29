from dataclasses import dataclass
from para.models.base_model import BaseModel


@dataclass
class MonitorPara(BaseModel):

    enable_stat_monitor: bool = True

    enable_dist_monitor: bool = False

    enable_phase_monitor: bool = False