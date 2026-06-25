from PASS.plot.plot_distribution import plot_distribution
from PASS.core.simulation import Simulation
from PASS.core.bunch import BunchInfo
from PASS.core.beam import Beam
from PASS.core.config import Config

import numpy as np
import os
from pathlib import Path
import logging
import tfs

logger = logging.getLogger(__name__)


def plot_main(sim: Simulation):
    cfg: Config = sim.cfg

    output_dir_plot = cfg.output_dir_plot
    Path(output_dir_plot).mkdir(parents=True, exist_ok=True)

    # 1. Plot distribution
    dist_all_path = get_file_in_dir(cfg.output_dir_dist)

    for dist_path in dist_all_path:
        logger.info(f"Plotting distribution: {dist_path}")

        if "injection" not in dist_path:
            plot_distribution(dist_path,
                              is_plot_hist=True,
                              bins=100,
                              cmap="magma",
                              is_plot_twiss_ellipse=True,
                              plot_twiss_rms_levels=[4, 9, 16],
                              is_plot_bucket=False,
                              save_dir=output_dir_plot)
        else:
            df = tfs.read(dist_path)
            headers = df.headers

            if headers["Longi type"] in ("gaussian", "coasting"):
                plot_distribution(dist_path,
                                  is_plot_hist=True,
                                  bins=100,
                                  cmap="magma",
                                  is_plot_twiss_ellipse=True,
                                  plot_twiss_rms_levels=[4, 9, 16],
                                  is_plot_bucket=False,
                                  save_dir=output_dir_plot)
            else:
                V0 = headers["RF voltage"]
                phi_s_deg = headers["RF phase"] / np.pi * 180
                h = headers["Harmonic num"]
                R0 = headers["Rho"]
                Ek = headers["Ek"]
                m0 = headers["m0"]
                A = headers["Proton num"] + headers["Neutron num"]
                q = headers["Charge num"]
                gamma_t = headers["Gamma T"]

                plot_distribution(dist_path,
                                  is_plot_hist=True,
                                  bins=100,
                                  cmap="magma",
                                  is_plot_twiss_ellipse=True,
                                  plot_twiss_rms_levels=[4, 9, 16],
                                  is_plot_bucket=True,
                                  plot_bucket_V0=V0,
                                  plot_bucket_phi_s_deg=phi_s_deg,
                                  plot_bucket_h=h,
                                  plot_bucket_R0=R0,
                                  plot_bucket_Ek_per_nucleon=Ek,
                                  plot_bucket_E0_per_nucleon=m0,
                                  plot_bucket_A=A,
                                  plot_bucket_q=q,
                                  plot_bucket_gamma_t=gamma_t,
                                  plot_bucket_num_orbits=4,
                                  save_dir=output_dir_plot)


def get_file_in_dir(target_dir: str, recursive: bool = False) -> list[str]:
    root = Path(target_dir).resolve()
    if not root.exists() or not root.is_dir():
        return []
    if recursive:
        # Recursively traverse all the files in all subdirectories
        return [str(p.resolve()) for p in root.rglob('*') if p.is_file()]
    else:
        # Only files in the current directory
        return [str(p.resolve()) for p in root.iterdir() if p.is_file()]
