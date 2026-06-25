import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import tfs
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch

from PASS.plot.plot_bucket import plot_bucket_on_ax
from PASS.tool.calc_bucket import resolve_kinematics, calc_bucket_width, calc_bucket_height, compute_bucket_data


def plot_distribution(tfs_file_path,
                      is_plot_hist: bool = True,
                      bins: int = 100,
                      cmap: str = "magma",
                      is_plot_twiss_ellipse: bool = True,
                      plot_twiss_rms_levels: list[int] = [4, 6, 9, 16],
                      is_plot_bucket: bool = False,
                      plot_bucket_V0: float = 0.0,
                      plot_bucket_phi_s_deg: float = 0.0,
                      plot_bucket_h: int = 0,
                      plot_bucket_R0: float = 0.0,
                      plot_bucket_Ek_per_nucleon: float = 0.0,
                      plot_bucket_E0_per_nucleon: float = 0.0,
                      plot_bucket_A: int = 0,
                      plot_bucket_q: int = 0,
                      plot_bucket_gamma_t: float = 0.0,
                      plot_bucket_num_orbits: int = 4,
                      save_dir=None):

    df = tfs.read(tfs_file_path)

    file_dir = Path(tfs_file_path).resolve().parent
    file_name = Path(tfs_file_path).resolve().stem
    if save_dir is None:
        output_dir = file_dir
    else:
        output_dir = Path(save_dir).resolve()
        os.makedirs(output_dir, exist_ok=True)

    x = df["x"].to_numpy()
    px = df["px"].to_numpy()
    y = df["y"].to_numpy()
    py = df["py"].to_numpy()
    z = df["z"].to_numpy()
    pz = df["pz"].to_numpy()

    plot_trans_distribution(x,
                            px,
                            is_plot_hist=is_plot_hist,
                            bins=bins,
                            cmap=cmap,
                            is_plot_twiss_ellipse=is_plot_twiss_ellipse,
                            rms_levels=plot_twiss_rms_levels,
                            direction="x-px",
                            save_path=output_dir / f"{file_name}_x-px.png")
    plot_trans_distribution(y,
                            py,
                            is_plot_hist=is_plot_hist,
                            bins=bins,
                            cmap=cmap,
                            is_plot_twiss_ellipse=is_plot_twiss_ellipse,
                            rms_levels=plot_twiss_rms_levels,
                            direction="y-py",
                            save_path=output_dir / f"{file_name}_y-py.png")
    plot_trans_distribution(x,
                            y,
                            is_plot_hist=is_plot_hist,
                            bins=bins,
                            cmap=cmap,
                            is_plot_twiss_ellipse=False,
                            rms_levels=plot_twiss_rms_levels,
                            direction="x-y",
                            save_path=output_dir / f"{file_name}_x-y.png")

    plot_longi_distribution(
        z,
        pz,
        is_plot_hist=is_plot_hist,
        bins=bins,
        cmap=cmap,
        is_plot_bucket=is_plot_bucket,
        plot_bucket_V0=plot_bucket_V0,
        plot_bucket_phi_s_deg=plot_bucket_phi_s_deg,
        plot_bucket_h=plot_bucket_h,
        plot_bucket_R0=plot_bucket_R0,
        plot_bucket_Ek_per_nucleon=plot_bucket_Ek_per_nucleon,
        plot_bucket_E0_per_nucleon=plot_bucket_E0_per_nucleon,
        plot_bucket_A=plot_bucket_A,
        plot_bucket_q=plot_bucket_q,
        plot_bucket_gamma_t=plot_bucket_gamma_t,
        plot_bucket_num_orbits=plot_bucket_num_orbits,
        save_path=output_dir / f"{file_name}_z-pz.png",
    )


def plot_trans_distribution(
    x,
    px,
    is_plot_hist: bool = False,
    bins: int = 100,
    cmap: str = "magma",
    is_plot_twiss_ellipse: bool = False,
    rms_levels: list[int] = [3, 6, 9, 12],
    direction=None,
    save_path=None,
):

    # =========================
    # 2D histogram
    # =========================
    H, xedges, yedges = np.histogram2d(x, px, bins=bins)

    H = H.T
    H = np.ma.masked_where(H == 0, H)

    fig = plt.figure(figsize=(12, 8))
    ax_heat = fig.add_subplot(111)

    # =========================
    # heatmap
    # =========================
    base_cmap = plt.colormaps[cmap].copy()
    base_cmap.set_bad(color='white', alpha=0.0)

    im = ax_heat.imshow(H, origin='lower', aspect='auto', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=base_cmap)

    if direction is None:
        raise ValueError(f"Direction must be 'x-px'/'y-py'/'x-y'")
    if direction.lower() == "x-px":
        ax_heat.set_xlabel("X (m)")
        ax_heat.set_ylabel("Px (rad)")
        legend_sigma1 = "x"
        legend_sigma2 = "px"
    elif direction.lower() == "y-py":
        ax_heat.set_xlabel("Y (m)")
        ax_heat.set_ylabel("Py (rad)")
        legend_sigma1 = "y"
        legend_sigma2 = "py"
    elif direction.lower() == "x-y":
        ax_heat.set_xlabel("X (m)")
        ax_heat.set_ylabel("Y (m)")
        legend_sigma1 = "x"
        legend_sigma2 = "y"
    else:
        raise ValueError(f"Direction must be 'x-px'/'y-py'/'x-y', but now is {direction}")

    # =========================
    # Twiss ellipse
    # =========================

    stat = {
        'x': x.mean(),
        'x2': (x**2).mean(),
        'xpx': (x * px).mean(),
        'px2': (px**2).mean(),
    }

    sigma_x = np.sqrt(stat['x2'] - stat['x']**2)
    sigma_px = np.sqrt(stat['px2'] - px.mean()**2)
    sig_xpx = stat['xpx'] - stat['x'] * px.mean()

    emit_x = np.sqrt(sigma_x**2 * sigma_px**2 - sig_xpx**2)

    beta = sigma_x**2 / emit_x
    alpha = -sig_xpx / emit_x
    gamma = (1 + alpha**2) / beta

    if direction.lower() in ("x-px", "y-py"):
        param_legend_text = (f"$\\alpha = {alpha:.3f}$\n"
                             f"$\\beta  = {beta:.3f}$\n"
                             f"$\\gamma = {gamma:.3f}$\n"
                             f"$\\sigma_{{\\mathrm{{{legend_sigma1}}}}} = {sigma_x:.3f}$\n"
                             f"$\\sigma_{{\\mathrm{{{legend_sigma2}}}}} = {sigma_px:.3f}$\n"
                             f"$\\varepsilon_{{\\mathrm{{rms}}}} = {emit_x:.3e}$")
    elif direction.lower() == "x-y":
        param_legend_text = (f"$\\sigma_{{\\mathrm{{{legend_sigma1}}}}} = {sigma_x:.3f}$\n"
                             f"$\\sigma_{{\\mathrm{{{legend_sigma2}}}}} = {sigma_px:.3f}$")
    else:
        raise ValueError(f"Direction must be 'x-px'/'y-py'/'x-y', but now is {direction}")
        
    # 创建一个不可见的矩形，仅用于承载图例文字
    dummy_patch = Patch(visible=False)
    ax_heat.legend(
        handles=[dummy_patch],
        labels=[param_legend_text],
        loc='best',  # 'best' 也可，但 upper left 通常较稳
        framealpha=0.8,
        fontsize=9,
        handlelength=0)  # 隐藏图例标志的线段

    # =========================
    # colorbar
    # =========================
    divider = make_axes_locatable(ax_heat)
    cax = divider.append_axes("right", size="3.5%", pad=0.12)

    cb = plt.colorbar(im, cax=cax)
    cb.set_label("Counts", rotation=270, labelpad=10)
    cb.ax.tick_params(direction='in')

    # =========================
    # style helper
    # =========================
    def style(ax):
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(1.0)
        ax.tick_params(direction='in', length=4, width=1)

    if is_plot_twiss_ellipse:

        theta = np.linspace(0, 2 * np.pi, 1000)
        gap_frac = 0.03
        gap = int(len(theta) * gap_frac)

        for n in rms_levels:
            eps = emit_x * n
            a = np.sqrt(eps * beta)
            b = np.sqrt(eps / beta)

            # 新版椭圆参数化
            x_ell = -a * np.cos(theta)
            xp_ell = b * (alpha * np.cos(theta) + np.sin(theta))

            # 缺口：切除 theta 首尾 gap 个点（θ=0 附近，即左侧）
            mask = np.ones_like(theta, dtype=bool)
            mask[:gap] = False
            mask[-gap:] = False

            x_ell_plot = np.where(mask, x_ell, np.nan)
            xp_ell_plot = np.where(mask, xp_ell, np.nan)

            ax_heat.plot(x_ell_plot, xp_ell_plot, color='red', linewidth=1.2)

            # 标注位置：缺口中心（θ=0 附近）
            theta_mid = theta[:gap].mean()
            x_lab = -a * np.cos(theta_mid)
            y_lab = b * (alpha * np.cos(theta_mid) + np.sin(theta_mid))

            # 可选：沿法线方向微调，防止文字压线（见下文）
            # 目前直接标注即可
            ax_heat.text(x_lab, y_lab, rf"${n}\epsilon_{{rms}}$", color='red', fontsize=9, horizontalalignment='center', verticalalignment='center')

    if is_plot_hist:

        # =========================
        # top histogram
        # =========================
        ax_top = divider.append_axes("top", size="25%", pad=0.12)

        ax_top.hist(x, bins=bins, color="#3B82F6", alpha=0.85, edgecolor="white", linewidth=0.3)

        ax_top.set_ylabel("Counts")
        ax_top.set_xlim(ax_heat.get_xlim())
        ax_top.xaxis.set_major_locator(ax_heat.xaxis.get_major_locator())
        ax_top.tick_params(labelbottom=False, direction='in')

        style(ax_top)

        # =========================
        # right histogram
        # =========================
        ax_right = divider.append_axes("right", size="28%", pad=0.85)

        ax_right.hist(px, bins=bins, orientation='horizontal', color="#10B981", alpha=0.85, edgecolor="white", linewidth=0.3)

        ax_right.set_xlabel("Counts")

        ax_right.set_ylim(ax_heat.get_ylim())
        ax_right.yaxis.set_major_locator(ax_heat.yaxis.get_major_locator())
        ax_right.tick_params(labelleft=False, direction='in')

        style(ax_right)

        # move to right
        pos = ax_right.get_position()
        ax_right.set_position([pos.x0 + 0.04, pos.y0, pos.width, pos.height])

    style(ax_heat)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    # plt.show()


def plot_longi_distribution(
    z,
    pz,
    is_plot_hist: bool = False,
    bins: int = 100,
    cmap: str = "magma",
    is_plot_bucket: bool = False,
    plot_bucket_V0: float = 0.0,
    plot_bucket_phi_s_deg: float = 0.0,
    plot_bucket_h: int = 0,
    plot_bucket_R0: float = 0.0,
    plot_bucket_Ek_per_nucleon: float = 0.0,
    plot_bucket_E0_per_nucleon: float = 0.0,
    plot_bucket_A: int = 0,
    plot_bucket_q: int = 0,
    plot_bucket_gamma_t: float = 0.0,
    plot_bucket_num_orbits: int = 4,
    save_path=None,
):

    # =========================
    # 2D histogram
    # =========================
    H, xedges, yedges = np.histogram2d(z, pz, bins=bins)

    H = H.T
    H = np.ma.masked_where(H == 0, H)

    fig = plt.figure(figsize=(12, 8))
    ax_heat = fig.add_subplot(111)

    # =========================
    # heatmap
    # =========================
    base_cmap = plt.colormaps[cmap].copy()
    base_cmap.set_bad(color='white', alpha=0.0)

    im = ax_heat.imshow(H, origin='lower', aspect='auto', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=base_cmap)

    ax_heat.set_xlabel("z (m)")
    ax_heat.set_ylabel(r"$\delta_p/p$")

    stat = {
        'z': z.mean(),
        'z2': (z**2).mean(),
        'zpz': (z * pz).mean(),
        'pz2': (pz**2).mean(),
    }

    sigma_z = np.sqrt(stat['z2'] - stat['z']**2)
    sigma_pz = np.sqrt(stat['pz2'] - pz.mean()**2)
    sig_zpz = stat['zpz'] - stat['z'] * pz.mean()

    emit_z = np.sqrt(sigma_z**2 * sigma_pz**2 - sig_zpz**2)

    beta = sigma_z**2 / emit_z
    alpha = -sig_zpz / emit_z
    gamma = (1 + alpha**2) / beta

    param_legend_text = (f"$\\alpha = {alpha:.3f}$\n"
                         f"$\\beta  = {beta:.3f}$\n"
                         f"$\\gamma = {gamma:.3f}$\n"
                         f"$\\sigma_z = {sigma_z:.3f}$\n"
                         f"$\\delta_p/p = {sigma_pz:.3f}$\n"
                         f"$\\varepsilon_{{\\mathrm{{rms}}}} = {emit_z:.3e}$")

    # 创建一个不可见的矩形，仅用于承载图例文字
    dummy_patch = Patch(visible=False)
    ax_heat.legend(
        handles=[dummy_patch],
        labels=[param_legend_text],
        loc='best',  # 'best' 也可，但 upper left 通常较稳
        framealpha=0.8,
        fontsize=9,
        handlelength=0)  # 隐藏图例标志的线段

    # =========================
    # colorbar
    # =========================
    divider = make_axes_locatable(ax_heat)
    cax = divider.append_axes("right", size="3.5%", pad=0.12)

    cb = plt.colorbar(im, cax=cax)
    cb.set_label("Counts", rotation=270, labelpad=10)
    cb.ax.tick_params(direction='in')

    # =========================
    # style helper
    # =========================
    def style(ax):
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(1.0)
        ax.tick_params(direction='in', length=4, width=1)

    # =========================
    # Bucket
    # =========================
    if is_plot_bucket:
        ion_params = {
            'V0': 100e3,
            'phi_s_deg': 30.0,
            'h': 1,
            'R0': 40.0,
            'Ek_per_nucleon': 45e6,
            'E0_per_nucleon': 938.272e6,
            'A': 1,
            'q': 1,
            'gamma_t': np.sqrt(1 / 0.0434),
            'num_orbits': 4
        }

        w_data = calc_bucket_width(phi_s_deg=ion_params['phi_s_deg'],
                                   Ek_per_nucleon=ion_params['Ek_per_nucleon'],
                                   E0_per_nucleon=ion_params['E0_per_nucleon'],
                                   A=ion_params['A'],
                                   gamma_t=ion_params['gamma_t'])

        h_data = calc_bucket_height(V0=ion_params['V0'],
                                    phi_s_deg=ion_params['phi_s_deg'],
                                    h=ion_params['h'],
                                    Ek_per_nucleon=ion_params['Ek_per_nucleon'],
                                    E0_per_nucleon=ion_params['E0_per_nucleon'],
                                    A=ion_params['A'],
                                    q=ion_params['q'],
                                    gamma_t=ion_params['gamma_t'])

        # 1. Compute physics data ONCE
        bucket_data = compute_bucket_data(**ion_params)

        # print(f"Bucket widt h: {w_data['width_deg']:.2f} deg ({w_data['width_rad']:.2f} rad)")
        # print(f"Bucket height: max dp/p = {h_data['dp_p_max']:.5e}, max dE = {h_data['dE_max_MeV']:.2f} MeV\n")
        # print(f"Calculated gamma: {bucket_data['gamma']:.4f}")
        # print(f"Calculated beta : {bucket_data['beta']:.4f}")
        # print(f"Calculated eta  : {bucket_data['eta']:.6f}")

        plot_bucket_on_ax(ax_heat, bucket_data, x_axis='z', y_axis='dp', phi_unit="rad", is_plot_label=False, is_plot_grid=False)

    if is_plot_hist:

        # =========================
        # top histogram
        # =========================
        ax_top = divider.append_axes("top", size="25%", pad=0.12)

        ax_top.hist(z, bins=bins, color="#3B82F6", alpha=0.85, edgecolor="white", linewidth=0.3)

        ax_top.set_ylabel("Counts")
        ax_top.set_xlim(ax_heat.get_xlim())
        ax_top.xaxis.set_major_locator(ax_heat.xaxis.get_major_locator())
        ax_top.tick_params(labelbottom=False, direction='in')

        style(ax_top)

        # =========================
        # right histogram
        # =========================
        ax_right = divider.append_axes("right", size="28%", pad=0.85)

        ax_right.hist(pz, bins=bins, orientation='horizontal', color="#10B981", alpha=0.85, edgecolor="white", linewidth=0.3)

        ax_right.set_xlabel("Counts")

        ax_right.set_ylim(ax_heat.get_ylim())
        ax_right.yaxis.set_major_locator(ax_heat.yaxis.get_major_locator())
        ax_right.tick_params(labelleft=False, direction='in')

        style(ax_right)

        # move to right
        pos = ax_right.get_position()
        ax_right.set_position([pos.x0 + 0.04, pos.y0, pos.width, pos.height])

    style(ax_heat)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    # plt.show()


if __name__ == "__main__":
    plot_distribution(
        r"C:\Users\changmx\Documents\PASS\output\2026_0624\1054_32\distribution\1054_32_beam0_bunch0_100000_hor_gaussian_longi_matchdp_Dx_0.0_injection.tfs",
        is_plot_hist=True,
        is_plot_twiss_ellipse=False,
    )
