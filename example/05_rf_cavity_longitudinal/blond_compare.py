"""Compare the migrated Example 05 waveform with BLonD's native Python kernels.

PASS evaluates the physical program at every particle time. This diagnostic uses
BLonD's simple drift and a locally linear RF phase with frozen voltage per
passage, so finite-amplitude/model differences are expected. Both runs start
from the same PASS state immediately after kick 0; the first kick is excluded.
Results and embedded figures are written to one standalone HTML file.
"""
from __future__ import annotations

import argparse, base64, importlib.util, importlib.metadata, io, json
from pathlib import Path

import numpy as np
import tfs
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_results import find_latest_output, measure_tune
from generate_input import CASES, CIRCUM, GAMMA_T, NUM_CHARGE, NUM_PROTON, NUM_NEUTRON, build_case, SCRIPT_DIR
from PASS.utils.constants import const

CASES_TO_COMPARE = ('twiss_h1_fixed', 'twiss_h1_ramping', 'twiss_h1_waveform')
TAGS = (1, 2, 3, 4, 5)


def native_backend():
    path = Path(importlib.util.find_spec('blond').origin).parent / 'utils' / 'butils_wrap_python.py'
    spec = importlib.util.spec_from_file_location('blond_example_native_python', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def run_pass(case):
    from PASS.core.config import Config
    from PASS.core.beam import Beam
    from PASS.core.state import SimulationState
    from PASS.core.simulation import Simulation
    from PASS.core.sequence import CommandSequence
    from PASS.core.executor import Executor
    from PASS.utils.logger import setup_logging
    from PASS.validation import validate_files
    path = build_case(case, SCRIPT_DIR, include_reference=True)
    report = validate_files([path])
    if not report.ok:
        raise ValueError(report.text())
    cfg = Config()
    cfg.load_input(path)
    setup_logging(log_file=cfg.get_log_path())
    beam = Beam(path, cfg)
    sim = Simulation(cfg, [beam], SimulationState())
    seq = CommandSequence(cfg.input_data[0], 0, sim)
    seq.sort()
    Executor().run(sim, [seq])
    return Path(cfg.output_dir_stat)


def read_particles(output):
    result = {}
    files = list((output / 'particle').glob('*_tag*.tfs')) or list(output.glob('*_tag*.tfs'))
    for path in files:
        tag = int(path.stem.split('_tag')[-1].lstrip('_'))
        if tag not in TAGS:
            continue
        table = tfs.read(path)
        missing = {'referenceTime', 'referenceBeta', 'referenceMomentum'} - set(table.columns)
        if missing:
            raise ValueError(f'{path}: missing reference columns {sorted(missing)}. '
                             'Enable "Include reference": true in ParticleMonitor '
                             '(generate_input.py --include-reference) and rerun PASS before BLonD comparison.')
        if np.any(table.tag <= 0):
            raise ValueError(f'Comparison tag {tag} is lost or missing')
        result[tag] = table
    if set(result) != set(TAGS):
        raise ValueError('Missing tagged monitor output')
    return result


def program_functions(path, origin=0.):
    """Independent analytic integral of the physical table, outside PASS tracker."""
    table = tfs.read(path)
    t = table.TIME.to_numpy()
    f = table.FREQUENCY.to_numpy()
    phi = table.PHASE.to_numpy()
    voltage = table.VOLTAGE.to_numpy()
    slopes = np.r_[np.diff(f) / np.diff(t), 0.]
    phase_slopes = np.r_[np.diff(phi) / np.diff(t), 0.]
    prefix = np.r_[0., np.cumsum(np.diff(t) * (f[1:] + f[:-1]) / 2)]

    def primitive(time):
        j = int(np.clip(np.searchsorted(t, time, side='right') - 1, 0, len(t) - 1))
        dx = time - t[j]
        slope = 0. if time < t[0] else slopes[j]
        return prefix[j] + f[j] * dx + slope * dx * dx / 2

    epoch = primitive(origin)

    def sample(time):
        j = int(np.clip(np.searchsorted(t, time, side='right') - 1, 0, len(t) - 1))
        modulation_slope = 0. if time < t[0] else phase_slopes[j]
        return (float(np.interp(time, t, voltage)), 2 * np.pi * (primitive(time) - epoch) + float(np.interp(time, t, phi)),
                2 * np.pi * float(np.interp(time, t, f)) + modulation_slope)

    return sample


def compare(case, output, destination):
    native, backend_path = native_backend()
    data = read_particles(output)
    first = data[1]
    turns = len(first)
    A = NUM_PROTON + NUM_NEUTRON
    mass = A * const.m_u_eV
    # Resolve the waveform from the input snapshot belonging to this run.
    inputs = list(output.glob('*input*.json')) + list((output / 'para').glob('*.json'))
    if not inputs:
        sibling = output.parent / 'beam0.json'
        inputs = [sibling if sibling.is_file() else SCRIPT_DIR / f'beam0_{case}.json']
    raw = json.loads(inputs[0].read_text(encoding='utf-8-sig'))
    rf = next(v for v in raw['Sequence'].values() if v.get('Command') == 'RFCavity')
    path = Path(rf['Components'][0]['Program file'])
    if not path.is_absolute():
        path = inputs[0].parent / path
    clock = raw.get('Reference clock') or {}
    sample = program_functions(path, float(clock.get('Time origin (s)', 0.)))
    ref_p = A * float(first.referenceMomentum.iloc[0])
    energy = np.hypot(ref_p, mass)
    beta = float(first.referenceBeta.iloc[0])
    T = float(first.referenceTime.iloc[0])
    dt = np.array([-float(data[tag].z.iloc[0]) / (beta * const.c) for tag in TAGS])
    dE = np.array([np.hypot(ref_p * (1 + float(data[tag].dp.iloc[0])), mass) - energy for tag in TAGS])
    native_times = np.empty((turns, len(TAGS)))
    native_energy = np.empty_like(native_times)
    for n in range(turns):
        if n:
            V, phase, omega = sample(T)
            gain = NUM_CHARGE * V * np.sin(phase)
            native.kick(dt, dE, np.array([V]), np.array([omega]), np.array([phase]), NUM_CHARGE, 1, -gain)
            energy += gain
            beta = np.sqrt((energy - mass) * (energy + mass)) / energy
        native_times[n] = T + dt
        native_energy[n] = energy + dE
        period = CIRCUM / (beta * const.c)
        eta = 1 / GAMMA_T**2 - (mass / energy)**2
        native.drift(dt, dE, 'simple', period, 1., 0, eta, 0., 0., 1 / GAMMA_T**2, 0., 0., beta, energy)
        T += period
    measured_times = np.column_stack([data[tag].referenceTime - data[tag].z / (data[tag].referenceBeta * const.c) for tag in TAGS])
    measured_energy = np.column_stack([np.hypot(A * data[tag].referenceMomentum * (1 + data[tag].dp), mass) for tag in TAGS])
    time_error = native_times - measured_times
    energy_error = native_energy - measured_energy
    metrics = [
        dict(tag=tag,
             max_time_difference_s=float(np.max(abs(time_error[:, j]))),
             max_energy_difference_eV_per_ion=float(np.max(abs(energy_error[:, j])))) for j, tag in enumerate(TAGS)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    reference = first.referenceTime.to_numpy()
    for j, tag in enumerate(TAGS):
        if tag == 1:
            continue
        axes[0, 0].plot((measured_times[:, j] - reference) * 1e9, label=f'PASS tag {tag}')
        axes[0, 1].plot((native_times[:, j] - reference) * 1e9, label=f'BLonD tag {tag}')
        axes[1, 0].plot(time_error[:, j] * 1e12, label=f'tag {tag}')
        axes[1, 1].plot(energy_error[:, j], label=f'tag {tag}')
    for ax in axes.flat:
        ax.grid(alpha=.3)
        ax.set_xlabel('turn')
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel('PASS arrival relative to reference / ns')
    axes[0, 1].set_ylabel('BLonD arrival relative to PASS reference / ns')
    axes[1, 0].set_ylabel('BLonD - PASS time / ps')
    axes[1, 1].set_ylabel('BLonD - PASS energy / eV per ion')
    fig.suptitle(case + ': declared map approximations')
    fig.tight_layout()
    image = io.BytesIO()
    fig.savefig(image, format='png', dpi=150)
    plt.close(fig)
    info = dict(case=case,
                turns=turns,
                initial_state='shared post-kick-0 state',
                blond_version=importlib.metadata.version('blond'),
                backend_source=str(backend_path),
                pass_output=str(output),
                rf_program=str(path),
                metrics=metrics)
    rows = ''.join(f'<tr><td>{r["tag"]}</td><td>{r["max_time_difference_s"]:.6e}</td><td>{r["max_energy_difference_eV_per_ion"]:.6e}</td></tr>'
                   for r in metrics)
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>PASS and BLonD comparison</title>
<style>body{max-width:1100px;margin:32px auto;font:16px/1.6 system-ui;padding:20px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}pre{white-space:pre-wrap;background:#eee;padding:15px}</style>
<h1>PASS and BLonD: physical-time RF input</h1>
<p>This comparison starts from a common state after kick 0. It runs the unmodified official BLonD Python kick and simple-drift kernels. PASS evaluates its waveform at each particle's physical time; BLonD samples a local carrier plus phase-modulation slope and freezes voltage within each passage. BLonD's simple drift is linear in energy deviation; PASS Twiss drift is linear in momentum deviation. Finite-amplitude and time-program differences are expected. No automatic agreement verdict is inferred from these plots.</p>
<table><tr><th>tag</th><th>max time difference / s</th><th>max energy difference / eV per ion</th></tr>''' + rows + '</table><img src="data:image/png;base64,' + base64.b64encode(
        image.getvalue()).decode() + '"><pre>' + json.dumps(info, indent=2) + '</pre></html>'
    destination.mkdir(parents=True, exist_ok=True)
    report = destination / 'pass_blond_report.html'
    report.write_text(html, encoding='utf-8')
    return report, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES_TO_COMPARE, default='twiss_h1_waveform')
    parser.add_argument('--skip-pass', action='store_true')
    parser.add_argument('--output-dir', type=Path, help='Existing PASS run; implies --skip-pass')
    args = parser.parse_args()
    output = args.output_dir or (find_latest_output(args.case) if args.skip_pass else run_pass(args.case))
    if output is None:
        parser.error('No complete output; run the PASS case first')
    report, info = compare(args.case, output, SCRIPT_DIR / 'blond_comparison_output' / args.case)
    print(json.dumps(info, indent=2))
    print('Report:', report)


if __name__ == '__main__':
    main()
