"""Generated-input PIC/frozen/quasi-frozen comparisons with independent theory.

Run ``python -m tests.integration.space_charge analytic``. Every case writes
input JSON, HDF5 fields, sampled particle kicks, CSV/JSON and comparison plots.
``analyse(run_dir)`` can regenerate plots from these artifacts without tracking.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from PASS.main import main as pass_main
from PASS.commands.space_charge import SpaceCharge as TrackingSpaceCharge
from PASS.para.api import generate_input
from PASS.para.schema.bunch import BunchConfig, InjectionItem
from PASS.para.schema.main import MainConfig
from PASS.para.schema.sequence import Sequence
from PASS.para.schema.slicer import Slicer
from PASS.para.schema.space_charge import SpaceCharge, SpaceChargeConfig, SpaceChargeResourceConfig
from PASS.utils.constants import const


CASES = ("gaussian_round", "gaussian_ellipse", "uniform_round", "uniform_ellipse")
METHODS = ("pic", "frozen", "quasi-frozen")


def reference_field(x, y, charge, parameters, gaussian):
    """Independent improper-integral solution; no production formula imports."""
    cx, cy, a, b, angle = parameters
    c, s = np.cos(angle), np.sin(angle)
    u, v = c*(x-cx)+s*(y-cy), -s*(x-cx)+c*(y-cy)
    scale = min(a, b)
    aa, bb = (a/scale)**2, (b/scale)**2
    xn, yn = u/scale, v/scale
    lower = 0.0
    if not gaussian and xn*xn/aa + yn*yn/bb > 1:
        lower = brentq(lambda t: xn*xn/(aa+t)+yn*yn/(bb+t)-1,
                       0, xn*xn+yn*yn)

    def integrand(t, horizontal):
        da, db = (aa+lower)*(1-t)+t, (bb+lower)*(1-t)+t
        weight = np.exp(-.5*(1-t)*(xn*xn/da+yn*yn/db)) if gaussian else 1.0
        return weight / (np.sqrt(da*db)*(da if horizontal else db))

    factor = charge / ((4 if gaussian else 2)*np.pi*const.epsilon0*scale**2)
    eu = factor*u*quad(integrand, 0, 1, args=(True,), epsabs=1e-12, epsrel=1e-12)[0]
    ev = factor*v*quad(integrand, 0, 1, args=(False,), epsabs=1e-12, epsrel=1e-12)[0]
    return c*eu-s*ev, s*eu+c*ev


def simulate(case: str, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    gaussian, ellipse = case.startswith("gaussian"), case.endswith("ellipse")
    sx, sy = .006, (.0035 if ellipse else .006)
    sizes = ({"sigma_x":sx,"sigma_y":sy} if ellipse else {"sigma":sx}) if gaussian else (
        {"a":2*sx,"b":2*sy} if ellipse else {"radius":2*sx})
    configurations = {}
    sequence = Sequence()
    sequence.add("injection", InjectionItem(harmonic_number=1, random_seed=20260909,
        bunches=[BunchConfig(kinetic_energy=33.2e6, num_real_particles=1_000_000_000,
            num_macro_particles=180_000, is_load_from_file=False, beta_x=1, beta_y=1,
            emit_x=sx*sx, emit_y=sy*sy, sigma_z=.1, dp=1e-6,
            dist_trans="gaussian" if gaussian else "kv", dist_longi="coasting", save_init_dist=False)]))
    sequence.add("slicer", Slicer(s=0, slice_set="space_charge", num_slices=3,
        z_range_mode="explicit", explicit={"z min":-.1,"z max":.1}, save_turns=[]))
    for method in METHODS:
        configurations[method] = SpaceChargeResourceConfig(
            method=method, solver="fft_free_space" if method == "pic" else f"{case}_free_space",
            nx=193, ny=193, grid_width_x=.09, grid_width_y=.09,
            **(sizes if method == "frozen" else {}))
        sequence.add(method, SpaceCharge(s=0, configuration=method, sc_length=.1,
            save_field=True, save_density=True, save_turns=[0]))
    input_dir = run_dir / "input"
    input_dir.mkdir(exist_ok=True)
    generate_input(MainConfig(beam_name=case, num_turns=1, backend="cpu",
        particle_precision="float64", circumference=100, is_plot=False,
        output_dir=str((run_dir/"simulation").resolve())), sequence,
        str(input_dir/"beam0.json"), space_charge=SpaceChargeConfig(enabled=True, configurations=configurations))

    # Observe actual command kicks without changing production execution.
    original = TrackingSpaceCharge._apply_bunch_cpu
    captured = []
    def observe(command, beam, bunch, particles, sim, turn):
        sid = np.asarray(bunch.slice_sets[command.slice_set_name].slice_id)
        n = bunch.end_idx-bunch.start_idx
        sl = slice(bunch.start_idx,bunch.end_idx)
        x, y = np.asarray(particles.x[sl]), np.asarray(particles.y[sl])
        indices = np.linspace(0,n-1,192,dtype=int)
        px, py = particles.px[sl][indices].copy(), particles.py[sl][indices].copy()
        # Independent statistics from the whole population, recorded before kick.
        stats = []
        for k in range(3):
            xy = np.column_stack((x[sid==k],y[sid==k]))
            cov = np.cov(xy,rowvar=False,bias=True)
            stats.append([len(xy),*xy.mean(axis=0),cov[0,0],cov[1,1],cov[0,1]])
        result = original(command,beam,bunch,particles,sim,turn)
        np.savez_compressed(run_dir/f"{command.method}_particle_kicks.npz",
            x=x[indices],y=y[indices],slice_id=sid[indices],
            kick_x=particles.px[sl][indices]-px,kick_y=particles.py[sl][indices]-py,
            statistics=np.asarray(stats),
            kick_factor=.1/(bunch.beta*const.c*bunch.brho*bunch.gamma**2))
        captured.append(command.method)
        return result
    with patch.object(TrackingSpaceCharge,"_apply_bunch_cpu",observe):
        pass_main(str(input_dir/"beam0.json"))
    assert set(captured) == set(METHODS), f"tracking did not execute all methods: {captured}"
    fields = {}
    for path in (run_dir/"simulation").rglob("*.h5"):
        with h5py.File(path) as handle:
            fields[handle.attrs["method"]] = str(path.relative_to(run_dir))
    assert set(fields) == set(METHODS), f"missing field snapshots: {fields}"
    (run_dir/"simulation_manifest.json").write_text(json.dumps({"case":case,"fields":fields},indent=2),encoding="utf-8")


def analyse(run_dir: Path) -> dict:
    manifest = json.loads((run_dir/"simulation_manifest.json").read_text(encoding="utf-8"))
    case = manifest["case"]
    gaussian = case.startswith("gaussian")
    analysis = run_dir/"analysis"
    figures = analysis/"figures"
    figures.mkdir(parents=True,exist_ok=True)
    data = {}
    rows, metrics = [], {}
    for method, relative in manifest["fields"].items():
        with h5py.File(run_dir/relative) as handle:
            data[method] = {name:handle[name][:] for name in handle}
        field = data[method]
        kicks = np.load(run_dir/f"{method}_particle_kicks.npz")
        if method == "pic":
            continue
        parameters = np.column_stack([field[name] for name in ("center_x","center_y","size_x","size_y","angle")])
        if method == "quasi-frozen":
            for k, stat in enumerate(kicks["statistics"]):
                count,cx,cy,vx,vy,cxy = stat
                np.testing.assert_allclose(parameters[k,:2],[cx,cy],atol=1e-15)
                if "round" in case:
                    variance = np.array([(vx+vy)/2]*2)
                else:
                    variance = np.linalg.eigvalsh([[vx,cxy],[cxy,vy]])[::-1]
                expected = np.sqrt(variance)*(1 if gaussian else 2)
                np.testing.assert_allclose(parameters[k,2:4],expected,rtol=1e-12,
                    err_msg=f"measured sizes={parameters[k,2:4]}, theory={expected}")
                np.testing.assert_allclose(field["macro_count"][k],count)
        theory = np.array([reference_field(x,y,field["slice_charge"][sid],parameters[sid],gaussian)
                           for x,y,sid in zip(kicks["x"],kicks["y"],kicks["slice_id"])])
        theory *= float(kicks["kick_factor"])/field["delta_z"][kicks["slice_id"]][:,None]
        measured = np.column_stack([kicks["kick_x"],kicks["kick_y"]])
        error = np.linalg.norm(measured-theory)/np.linalg.norm(theory)
        metrics[f"{method}_kick_relative_l2"] = float(error)
        assert error < 2e-8, f"{case}/{method}: measured_error={error}, theory=0, tolerance=2e-8"
        for i in range(len(theory)):
            rows.append({"method":method,"slice_id":int(kicks['slice_id'][i]),
                "x_m":kicks['x'][i],"y_m":kicks['y'][i],
                "measured_kick_x":measured[i,0],"theory_kick_x":theory[i,0],
                "measured_kick_y":measured[i,1],"theory_kick_y":theory[i,1]})
    pd.DataFrame(rows).to_csv(analysis/"particle_kick_comparison.csv",index=False)

    pic = data["pic"]
    xx, yy = np.meshgrid(pic["x"],pic["y"])
    comparison_rows = []
    for method in ("frozen","quasi-frozen"):
        analytic = data[method]
        # Compare in the resolved central region, using a global norm so zero
        # field at the center does not create artificial relative-error spikes.
        mask = (xx**2+yy**2 < .025**2)
        pf = np.stack((pic["integrated_Ex"],pic["integrated_Ey"]))
        af = np.stack((analytic["integrated_Ex"],analytic["integrated_Ey"]))
        error = np.linalg.norm((pf-af)[:,:,mask])/np.linalg.norm(af[:,:,mask])
        metrics[f"pic_vs_{method}_field_relative_l2"] = float(error)
        assert error < .06, f"{case}: PIC vs {method} measured_error={error}, theory=0, tolerance=0.06"
        # PIC accumulates floating macro charges, whereas analytic charge is
        # count*q_macro. Bound ordinary sequential-summation roundoff by N*eps.
        charge_tolerance = 2 * float(np.max(analytic["macro_count"])) * np.finfo(float).eps
        np.testing.assert_allclose(pic["slice_charge"],analytic["slice_charge"],rtol=charge_tolerance,atol=0,
            err_msg=f"PIC charge={pic['slice_charge']}, count*q={analytic['slice_charge']}, rtol={charge_tolerance}")
    iy = len(pic['y'])//2
    fig, axes = plt.subplots(2,3,figsize=(13,7),layout="constrained")
    for k in range(3):
        for method in METHODS:
            f = data[method]
            axes[0,k].plot(f['x']*1000,f['integrated_Ex'][k,iy],label=method)
        for method in ("frozen","quasi-frozen"):
            residual = np.hypot(pic['integrated_Ex'][k]-data[method]['integrated_Ex'][k],
                                pic['integrated_Ey'][k]-data[method]['integrated_Ey'][k])
            peak = np.max(np.hypot(data[method]['integrated_Ex'][k],data[method]['integrated_Ey'][k]))
            axes[1,k].plot(pic['x']*1000,residual[iy]/peak,label=f"PIC vs {method}")
        axes[0,k].set(title=f"Slice {k}",xlabel="x (mm)",ylabel="Integrated Ex (V)")
        axes[1,k].set(xlabel="x (mm)",ylabel="Field error / analytic peak")
        for ax in axes[:,k]:
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        for j,x in enumerate(pic['x']):
            comparison_rows.append({"slice_id":k,"x_m":x,**{
                f"{m}_Ex_V":data[m]['integrated_Ex'][k,iy,j] for m in METHODS}})
    fig.suptitle(case.replace('_',' ') + ' — free space')
    fig.savefig(figures/"pic_frozen_quasi_frozen_field_comparison.png",dpi=150)
    plt.close(fig)
    pd.DataFrame(comparison_rows).to_csv(analysis/"field_scan.csv",index=False)
    fig, ax = plt.subplots(figsize=(6,5),layout="constrained")
    frame = pd.DataFrame(rows)
    for method, group in frame.groupby('method'):
        ax.scatter(group.theory_kick_x,group.measured_kick_x,s=9,label=method,alpha=.6)
    lim = max(abs(frame.theory_kick_x).max(),abs(frame.measured_kick_x).max())
    ax.plot([-lim,lim],[-lim,lim],'k--',lw=1,label='equality')
    ax.set(xlabel="Independent theoretical kick x",ylabel="Measured kick x",title=case)
    ax.legend()
    ax.grid(alpha=.25)
    fig.savefig(figures/"analytic_kick_vs_independent_integral.png",dpi=150)
    plt.close(fig)
    summary = {"case":case,"metrics":metrics,"kick_tolerance":2e-8,"pic_field_tolerance":.06}
    (analysis/"summary.json").write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


@pytest.mark.parametrize("case",CASES)
def test_generated_analytic_tracking_against_integrals_and_pic(case,sc_output_dir):
    run_dir = sc_output_dir/f"{case}_analytic_free_space"/"run_001"
    simulate(case,run_dir)
    analyse(run_dir)


@pytest.mark.parametrize("profile", ["gaussian_ellipse", "uniform_ellipse"])
def test_repeated_kicks_follow_slice_evolution_only_in_quasi_frozen(profile,sc_output_dir):
    """Prescribed affine transport isolates parameter refresh from lattice optics."""
    root = sc_output_dir/f"{profile}_parameter_evolution"
    root.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(90209)
    source = rng.normal(size=(512,2))
    source -= source.mean(axis=0)
    source = source @ np.linalg.inv(np.linalg.cholesky(np.cov(source,rowvar=False,bias=True))).T
    source *= [.004,.002]
    source = np.vstack((source,1.5*source))
    sid = np.repeat([0,1],512)
    gaussian = profile.startswith("gaussian")
    records = []
    for method in ("frozen","quasi-frozen"):
        sizes = {"sigma_x":.004,"sigma_y":.002} if gaussian else {"a":.008,"b":.004}
        config = SpaceChargeResourceConfig(method=method,solver=f"{profile}_free_space",
            nx=17,ny=17,grid_width_x=.1,grid_width_y=.1,**(sizes if method=="frozen" else {}))
        p = SimpleNamespace(x=np.zeros(1024),y=np.zeros(1024),px=np.zeros(1024),py=np.zeros(1024),
                            tag=np.ones(1024,dtype=np.int32))
        slices = SimpleNamespace(slice_id=sid,slice_table={"delta_z":np.array([.1,.2])},valid_turn=-1,valid_s=999)
        bunch = SimpleNamespace(start_idx=0,end_idx=1024,bunch_id=0,ratio=1e5,num_charge=1,
            beta=.5,gamma=1.1547005383792517,brho=3.,slice_sets={"space_charge":slices})
        beam = SimpleNamespace(particles=p,bunches=[bunch])
        cfg = SimpleNamespace(space_charge=[SpaceChargeConfig(enabled=True,configurations={"test":config})],
            input_data=[{"sequence":{"sc":{"command":"spacecharge","configuration":"test"}}}],
            output_dir_space_charge=str(root/method),output_hms="evolution",backend="cpu")
        sim = SimpleNamespace(cfg=cfg,beams=[beam],state=SimpleNamespace(turn=0))
        command = TrackingSpaceCharge(0,sim,**{"Configuration":"test","SC length (m)":.1,
            "Save field":True,"Save turns":[[0,8,1]]})
        for turn in range(9):
            angle = .05*turn
            c,s = np.cos(angle),np.sin(angle)
            xy = source @ np.diag([1+.1*turn,1+.04*turn]) @ np.array([[c,s],[-s,c]])
            xy += [turn*.0003,-turn*.0001]
            p.x[:],p.y[:] = xy[:,0],xy[:,1]
            px,py = p.px.copy(),p.py.copy()
            sim.state.turn = turn
            command.execute_cpu(sim)
            path = next((root/method).glob(f"*/turn_{turn:06d}/*.h5"))
            with h5py.File(path) as f:
                for k in range(2):
                    params = np.array([f[name][k] for name in ("center_x","center_y","size_x","size_y","angle")])
                    if method=="frozen":
                        expected = np.array([0,0,.004,.002,0.])
                    else:
                        expected = np.array([turn*.0003,-turn*.0001,
                            .004*(1+.1*turn)*(1+.5*k),.002*(1+.04*turn)*(1+.5*k),angle])
                    if not gaussian:
                        expected[2:4] *= 2
                    np.testing.assert_allclose(params,expected,rtol=1e-12,atol=1e-15,
                        err_msg=f"{method}, turn={turn}, slice={k}: measured={params}, theory={expected}")
                    indices = np.flatnonzero(sid==k)[::64]
                    theoretical_field = np.array([reference_field(p.x[i],p.y[i],f['slice_charge'][k],expected,gaussian) for i in indices])
                    theoretical_kick = theoretical_field*.1/(bunch.beta*const.c*bunch.brho*bunch.gamma**2*f['delta_z'][k])
                    measured = np.column_stack(((p.px-px)[indices],(p.py-py)[indices]))
                    error = np.linalg.norm(measured-theoretical_kick)/np.linalg.norm(theoretical_kick)
                    assert error < 2e-8, f"{method}, turn={turn}: measured error={error}, theory=0, tolerance=2e-8"
                    records.append({"method":method,"turn":turn,"slice":k,"size_x_m":params[2],
                        "center_x_m":params[0],"angle_rad":params[4],"kick_relative_l2":error})
    frame = pd.DataFrame(records)
    frame.to_csv(root/'parameter_evolution.csv',index=False)
    fig,axes = plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for (method,k),group in frame.groupby(['method','slice']):
        for ax,column,factor,label in zip(axes,['size_x_m','center_x_m','angle_rad'],[1e3,1e3,1],
                                         ['Principal size x (mm)','Center x (mm)','Angle (rad)']):
            ax.plot(group.turn,group[column]*factor,'o-',ms=3,label=f'{method}, slice {k}')
            ax.set(xlabel='Kick index',ylabel=label)
            ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.suptitle(profile.replace('_',' ')+' — prescribed affine transport')
    fig.savefig(root/'frozen_vs_quasi_frozen_parameter_evolution.png',dpi=150)
    plt.close(fig)
    (root/'summary.json').write_text(json.dumps({"max_kick_relative_l2":float(frame.kick_relative_l2.max()),
        "kick_tolerance":2e-8,"transport":"prescribed affine maps, not a ring stability test"},indent=2),encoding='utf-8')
