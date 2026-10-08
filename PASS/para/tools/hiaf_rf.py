"""Normalize HIAF chart exports and apply explicitly declared RF phase rules.

The chart columns alone do not identify the hardware phase convention. Raw
conversion is therefore separate from constructing executable PASS components.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import tfs

from PASS.para.schema.rf import RFComponent
from PASS.utils.program import LinearProgram


@dataclass(frozen=True)
class HiafRFData:
    """Common physical-time samples, SI-valued channels and source provenance."""

    times: np.ndarray
    channels: dict[str, np.ndarray]
    provenance: dict


def load_hiaf_rf(source_directory):
    """Read the 14 two-column HIAF RF chart files without interpreting phases.

    Input columns are time in ms and voltage/frequency/phase/harmonic in
    kV/kHz/rad/dimensionless units. The unsuffixed channel is channel 0; the
    suffixes 1 and 2 denote channels 1 and 2. All files must share one finite,
    strictly increasing time grid. Unrelated files are ignored.
    """
    source = Path(source_directory).resolve()
    expected = {name + suffix for name in ('Voltage', 'Frequency', 'Phase', 'Harmonic') for suffix in ('', '1', '2')}
    expected.update(('DeltaPhi1', 'DeltaPhi2'))
    paths = {}
    for path in sorted(source.glob('*.txt')):
        match = re.search(r'#RF(Voltage|Frequency|Phase|Harmonic|DeltaPhi)([12]?)PlotData', path.name)
        if match is None:
            continue
        name = ''.join(match.groups())
        if name not in expected:
            continue
        if name in paths:
            raise ValueError(f'Duplicate HIAF channel {name}: {paths[name].name}, {path.name}')
        paths[name] = path
    missing = expected - paths.keys()
    if missing:
        raise ValueError(f'Missing HIAF RF channels: {sorted(missing)}')

    times = None
    channels = {}
    provenance = {'source_directory': str(source), 'files': {}}
    for name, path in sorted(paths.items()):
        raw = np.loadtxt(path, delimiter=',', ndmin=2)
        if raw.shape[1] != 2 or len(raw) < 2 or not np.isfinite(raw).all() or np.any(np.diff(raw[:, 0]) <= 0):
            raise ValueError(f'{path.name}: require two finite columns and at least two strictly increasing times')
        if times is None:
            times = raw[:, 0].copy()
        elif not np.array_equal(raw[:, 0], times):
            raise ValueError(f'{path.name}: time grid differs from the other RF channels')
        factor = 1000. if name.startswith(('Voltage', 'Frequency')) else 1.
        values = raw[:, 1] * factor
        if name.startswith('Harmonic') and (np.any(values < 0) or np.any(values != np.rint(values))):
            raise ValueError(f'{path.name}: harmonic values must be nonnegative integers; zero denotes disabled metadata')
        channels[name] = values
        provenance['files'][name] = {'name': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'samples': len(raw)}
    return HiafRFData(times * .001, channels, provenance)


def _select_time_range(data, start_time, end_time):
    start = data.times[0] if start_time is None else float(start_time)
    end = data.times[-1] if end_time is None else float(end_time)
    if not np.isfinite([start, end]).all():
        raise ValueError('The selected time range must be finite')
    # Preserve exact existing nodes, including endpoints differing only by a
    # decimal-to-binary round trip. New endpoints use declared linear samples.
    for name, value in (('start', start), ('end', end)):
        index = int(np.argmin(np.abs(data.times - value)))
        if abs(data.times[index] - value) <= 4 * np.spacing(max(abs(value), abs(data.times[index]))):
            if name == 'start':
                start = data.times[index]
            else:
                end = data.times[index]
    if start < data.times[0] or end > data.times[-1] or not start < end:
        raise ValueError('The selected time range must be increasing and contained in the export')
    times = np.r_[start, data.times[(data.times > start) & (data.times < end)], end]
    channels = {}
    for name, values in data.channels.items():
        if name.startswith('Harmonic'):
            # Harmonics are discrete labels, not interpolated physical fields.
            indices = np.clip(np.searchsorted(data.times, times, side='right') - 1, 0, len(data.times) - 1)
            channels[name] = values[indices]
        else:
            channels[name] = np.interp(times, data.times, values)
    return HiafRFData(times, channels, data.provenance)


def _phase_values(channels, rule):
    if not isinstance(rule, dict) or not rule:
        raise ValueError('Each phase rule must be a nonempty object of phase-channel coefficients and optional offset')
    phase = np.zeros_like(channels['Phase'])
    for name, coefficient in rule.items():
        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) or not np.isfinite(coefficient):
            raise ValueError(f'Phase coefficient {name} must be a finite number')
        if name == 'offset':
            phase += coefficient
        elif name in channels and name.startswith(('Phase', 'DeltaPhi')):
            phase += coefficient * channels[name]
        else:
            raise ValueError(f'Unknown phase channel in rule: {name}')
    return phase


def _build_components(data, clock_data, phase_rules, base_harmonic, phase_origin):
    if not isinstance(base_harmonic, int) or isinstance(base_harmonic, bool) or base_harmonic < 1:
        raise ValueError('base_harmonic must be a positive integer')
    if phase_origin is None or not np.isfinite(phase_origin):
        raise ValueError('Executable RF conversion requires an explicit finite phase_origin in seconds')
    if not isinstance(phase_rules, dict) or not phase_rules:
        raise ValueError('phase_rules must be a nonempty object keyed by channel:harmonic, e.g. 1:8')
    for key in phase_rules:
        if re.fullmatch(r'[012]:[1-9][0-9]*', key) is None:
            raise ValueError(f'Invalid phase rule key {key!r}; use channel:harmonic, e.g. 0:4')
        _phase_values(data.channels, phase_rules[key])
    base_metadata = clock_data.channels['Harmonic']
    if np.any((base_metadata != 0) & (base_metadata != base_harmonic)):
        raise ValueError('Base-channel harmonic changes; split the source into ranges with one base harmonic')
    frequency = data.channels['Frequency'] / base_harmonic
    # Keep the full prescribed clock when cropping component domains: changing
    # prehistory would change the integrated phase at every retained sample.
    clock_frequency = clock_data.channels['Frequency'] / base_harmonic
    if np.any(clock_frequency <= 0):
        raise ValueError('The source base revolution frequency must be positive')
    clock = LinearProgram(clock_frequency, clock_data.times, origin=float(phase_origin))
    tables = []
    relations = []
    for channel, suffix in enumerate(('', '1', '2')):
        harmonic = data.channels['Harmonic' + suffix].astype(int)
        voltage = data.channels['Voltage' + suffix]
        if np.any((harmonic == 0) & (voltage != 0)):
            raise ValueError(f'Channel {channel}: nonzero voltage has disabled harmonic metadata')
        starts = np.r_[0, np.flatnonzero(np.diff(harmonic) != 0) + 1]
        ends = np.r_[starts[1:] - 1, len(harmonic) - 1]
        for start, end in zip(starts, ends):
            h = int(harmonic[start])
            if h == 0 or not np.any(voltage[start:end + 1] != 0):
                continue
            key = f'{channel}:{h}'
            if key not in phase_rules:
                raise ValueError(f'Missing explicit phase rule for active channel:harmonic {key}')
            expected_frequency = h * frequency[start:end + 1]
            observed_frequency = data.channels['Frequency' + suffix][start:end + 1]
            if not np.allclose(observed_frequency, expected_frequency, rtol=5e-12, atol=1e-8):
                error = float(np.max(np.abs(observed_frequency - expected_frequency)))
                raise ValueError(f'Channel {channel}, h={h}: frequency differs from shared harmonic clock; max error {error:g} Hz')
            # Keep the nearest zero on each side to retain the exported linear
            # voltage ramp. Frequencies and phases at disabled nodes are not
            # guessed; the common harmonic clock continues through them.
            nonzero = np.flatnonzero(voltage[start:end + 1] != 0) + start
            first, last = int(nonzero[0]), int(nonzero[-1])
            lower = max(0, first - 1)
            upper = min(len(voltage) - 1, last + 1)
            if lower < start and voltage[lower] != 0 or upper > end and voltage[upper] != 0:
                raise ValueError(f'Channel {channel}: harmonic switch without a zero-voltage separator is ambiguous')
            selection = slice(lower, upper + 1)
            phase = _phase_values(data.channels, phase_rules[key])
            times = data.times[selection]
            explicit_frequency = h * frequency[selection]
            explicit = LinearProgram(explicit_frequency, times, origin=0.)
            # Both integrals have the same derivative inside this segment.
            # One constant phase preserves the old epoch and cropped prehistory.
            phase_adjustment = 2 * np.pi * np.remainder(h * clock.phase_cycles(float(times[0])) - explicit.phase_cycles(float(times[0])), 1.)
            table = tfs.TfsDataFrame({
                'TIME': times,
                'VOLTAGE': voltage[selection],
                'FREQUENCY': explicit_frequency,
                'PHASE': phase[selection] + phase_adjustment
            })
            RFComponent(times=table.TIME.tolist(), voltage=table.VOLTAGE.tolist(), phase=table.PHASE.tolist(), frequency=table.FREQUENCY.tolist())
            tables.append((channel, h, table))
            relations.append({
                'channel':
                channel,
                'harmonic':
                h,
                'maximum_frequency_error_hz':
                float(np.max(np.abs(observed_frequency - expected_frequency))),
                'phase_adjustment_rad':
                float(phase_adjustment),
                'program_domain_s':
                times[[0, -1]].tolist(),
                'frequency_construction':
                'Source base Frequency/base_harmonic multiplied by this segment harmonic, including zero-voltage boundaries.'
            })
    return tables, relations


def convert_hiaf_rf(source_directory,
                    output_directory,
                    *,
                    phase_rules=None,
                    phase_origin=None,
                    base_harmonic=4,
                    start_time=None,
                    end_time=None,
                    overwrite=False):
    """Write normalized data, provenance and optionally executable PASS tables.

    ``phase_rules`` explicitly maps ``channel:harmonic`` to a linear combination
    of exported phase columns (plus optional ``offset`` in radians). For example
    ``{'0:4': {'Phase': 1}, '1:8': {'Phase': 2, 'DeltaPhi1': 1}}`` describes one
    declared interpretation of low-energy BRing RF. It is not automatically
    identified or certified from the export. Phase inputs must be unwrapped.

    Executable output uses explicit RF FREQUENCY samples on each constant-
    harmonic source segment, with a constant phase adjustment preserving the
    original integral of Frequency/base_harmonic. ``phase_origin`` is that
    original integral's zero epoch; it does not shift the exported time column.
    No public machine-clock input is emitted. Voltages are linear between
    samples and zero outside each component's domain. No ion-mass calibration
    or adjustment to the automatically derived machine trajectory occurs.

    Omit phase_rules for a raw-only conversion, which needs no phase assumption.
    Existing files are retained unless overwrite=True is explicitly requested.
    """
    original = load_hiaf_rf(source_directory)
    data = _select_time_range(original, start_time, end_time)
    output = Path(output_directory).resolve()
    tables, relations = [], []
    if phase_rules is not None:
        tables, relations = _build_components(data, original, phase_rules, base_harmonic, phase_origin)
    manifest = {
        'format': 'PASS HIAF RF conversion v2',
        'provenance': data.provenance,
        'input_units': {
            'time': 'ms',
            'voltage': 'kV',
            'frequency': 'kHz',
            'phase': 'rad',
            'harmonic': 'dimensionless'
        },
        'output_units': {
            'time': 's',
            'voltage': 'V',
            'frequency': 'Hz',
            'phase': 'rad',
            'harmonic': 'dimensionless'
        },
        'time_range_s': data.times[[0, -1]].tolist(),
        'samples': len(data.times),
        'phase_rules': phase_rules,
        'phase_origin_s': phase_origin,
        'base_harmonic': base_harmonic,
        'frequency_relations': relations,
        'source_frequency_modified': False,
        'frequency_semantics': 'Explicit RF frequency is segment harmonic times the original base frequency/base_harmonic; '
        'disabled boundary frequencies continue that program rather than using disabled-channel zeros.',
        'phase_adjustment': 'Each segment preserves the original phase_origin and frequency prehistory modulo 2*pi; '
        'the runtime frequency integral uses physical time with origin zero.',
        'phase_semantics': 'User-declared linear combinations; source phase semantics are not inferred.',
        'interpolation': 'Analog channels are linear; harmonic labels are discrete. Adjacent zero-voltage nodes retain on/off ramps.',
        'mass_convention': 'Not supplied by these chart files; verify source frequency against the intended ion mass and lattice independently.',
        'normalized_data': str(output / 'hiaf_rf_normalized.tfs'),
        'configuration': str(output / 'rf_config.json') if phase_rules is not None else None,
        'components': [],
    }
    products = {output / 'hiaf_rf_normalized.tfs': tfs.TfsDataFrame({'TIME': data.times, **{k.upper(): v for k, v in data.channels.items()}})}
    for index, (channel, harmonic, table) in enumerate(tables, start=1):
        path = output / f'rf_channel_{channel}_h{harmonic}_{index:02d}.tfs'
        products[path] = table
        manifest['components'].append({'Program file': str(path)})
    paths = list(products) + [output / 'conversion_report.json']
    if phase_rules is not None:
        paths.append(output / 'rf_config.json')
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Output files already exist; choose a new directory or explicitly enable overwrite: {existing}')
    output.mkdir(parents=True, exist_ok=True)
    for path, table in products.items():
        tfs.write(path, table, colwidth=25, headerswidth=25)
    if phase_rules is not None:
        config = {'Components': manifest['components']}
        (output / 'rf_config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'conversion_report.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_directory', type=Path)
    parser.add_argument('output_directory', type=Path)
    parser.add_argument('--phase-rules',
                        type=Path,
                        help='JSON object declaring channel:harmonic phase-column coefficients; omit for raw-only conversion')
    parser.add_argument('--phase-origin',
                        type=float,
                        help='Original physical epoch (s) of zero accumulated source-clock phase; preserved in PHASE; required with --phase-rules')
    parser.add_argument('--base-harmonic', type=int, default=4)
    parser.add_argument('--start-time', type=float, help='First physical time (s), within the source interval')
    parser.add_argument('--end-time', type=float, help='Last physical time (s), within the source interval')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace matching generated output files')
    args = parser.parse_args()
    phase_rules = None if args.phase_rules is None else json.loads(args.phase_rules.read_text(encoding='utf-8'))
    report = convert_hiaf_rf(args.source_directory,
                             args.output_directory,
                             phase_rules=phase_rules,
                             phase_origin=args.phase_origin,
                             base_harmonic=args.base_harmonic,
                             start_time=args.start_time,
                             end_time=args.end_time,
                             overwrite=args.overwrite)
    print(
        json.dumps(
            {
                'samples': report['samples'],
                'time_range_s': report['time_range_s'],
                'components': len(report['components']),
                'report': str(args.output_directory / 'conversion_report.json')
            },
            indent=2))


if __name__ == '__main__':
    main()
