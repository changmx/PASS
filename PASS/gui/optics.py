"""Validated inputs for the GUI's Twiss sequence generators."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from PASS.para.schema.twiss import TwissItem
from PASS.para.smooth import generate_smooth_twiss


class OpticsParameters(BaseModel):
    """One ring, expressed either as one periodic map or N smooth segments."""

    model_config = ConfigDict(allow_inf_nan=False, str_strip_whitespace=True)

    mode: Literal["one_turn", "smooth"]
    name: str = Field(min_length=1)
    circumference: float = Field(gt=0)
    qx: float
    qy: float
    num_segments: int = Field(default=100, strict=True, ge=1)
    alpha_x: float = 0.0
    alpha_y: float = 0.0
    beta_x: float = Field(default=1.0, gt=0)
    beta_y: float = Field(default=1.0, gt=0)
    dx: float = 0.0
    dpx: float = 0.0
    dqx: float = 0.0
    dqy: float = 0.0
    muz: float = 0.0
    longitudinal_transfer: Literal["off", "drift", "matrix"] = "off"

    @model_validator(mode="after")
    def validate_smooth_optics(self):
        if self.mode == "smooth":
            if self.qx <= 0 or self.qy <= 0:
                raise ValueError("平滑近似要求 Qx、Qy 大于 0（使用完整 tune）。")
            for beta in self.betas:
                if not math.isfinite(beta) or beta <= 0:
                    raise ValueError("平滑近似计算出的 beta 必须是有限正数。")
            if self.circumference / self.num_segments == 0:
                raise ValueError("分段步长过小。")
        return self

    @property
    def betas(self) -> tuple[float, float]:
        if self.mode == "smooth":
            return (self.circumference / (2 * math.pi * self.qx), self.circumference / (2 * math.pi * self.qy))
        return self.beta_x, self.beta_y

    def generate(self) -> tuple[list[TwissItem], list[str]]:
        """Return ordinary Twiss commands; no generator metadata enters JSON."""
        muz = self.muz if self.longitudinal_transfer == "matrix" else 0.0
        if self.mode == "smooth":
            items, _, _ = generate_smooth_twiss(
                circumference=self.circumference,
                qx=self.qx,
                qy=self.qy,
                num_points=self.num_segments + 1,
                alpha_x=self.alpha_x,
                alpha_y=self.alpha_y,
                dx=self.dx,
                dpx=self.dpx,
                muz=muz,
                dqx=self.dqx,
                dqy=self.dqy,
                longitudinal_transfer=self.longitudinal_transfer,
            )
            # Index names remain distinct even for steps smaller than 0.001 m.
            width = len(str(self.num_segments))
            return items, [f"{self.name}_{i:0{width}d}" for i in range(len(items))]
        item = TwissItem(
            s=self.circumference,
            s_previous=0.0,
            alpha_x=self.alpha_x,
            alpha_y=self.alpha_y,
            beta_x=self.beta_x,
            beta_y=self.beta_y,
            alpha_x_previous=self.alpha_x,
            alpha_y_previous=self.alpha_y,
            beta_x_previous=self.beta_x,
            beta_y_previous=self.beta_y,
            dx=self.dx,
            dpx=self.dpx,
            dx_previous=self.dx,
            dpx_previous=self.dpx,
            mu_x=self.qx,
            mu_y=self.qy,
            mu_z=muz,
            mu_x_previous=0.0,
            mu_y_previous=0.0,
            mu_z_previous=0.0,
            dqx=self.dqx,
            dqy=self.dqy,
            longitudinal_transfer=self.longitudinal_transfer,
        )
        return [item], [self.name]
