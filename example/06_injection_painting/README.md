# Two-plane injection painting

Use `generate_input.py`, `run_simulation.py`, and `analyze_results.py` as the
workflow entry points for input generation, tracking, and result analysis.

This example imports the real-magnet CISP injection case into PASS. Four Bump
elements carry independent horizontal and vertical waveforms. New particles
enter at fixed offsets; the bump magnets act on every surviving particle as it
travels around the ring. The importer reads the source files without modifying
them and records SHA-256 hashes and the translated element locations.

## Generate and run

Run these commands from the PASS repository root, replacing the example source
path with your CISP command file. Use a new output directory for each case.

The finite-length ES needs measured or deliberately specified V or VL and a
clear gap. These are not present in the source point-cut parameters.
Enter them explicitly below; the voltage is the septum potential minus the
counter-electrode potential. Do not use guessed values for physical conclusions.

```powershell
$esVoltage = [double](Read-Host "ES septum-minus-counter voltage (V)")
$esGap = [double](Read-Host "ES clear gap (m)")
python example/06_injection_painting/generate_input.py --source C:/data/thread3/cisp_cmd.txt --output tests/codex/painting_reference --turns 100 --stage aperture --clock reference --es-voltage $esVoltage --es-gap $esGap
python example/06_injection_painting/run_simulation.py tests/codex/painting_reference/beam0.json
```

Alternatively, replace `--es-voltage` with `--es-vl` to supply the longitudinal
voltage integral in V m. The integrated electric field is VL/gap. The importer
distributes the total VL between ES segments in proportion to their lengths;
it does not apply the full integral in each segment. Electrodes and the field
cover all local v, while the source apertures limit transverse acceptance.
This importer retains a positive physical ES length; the underlying ElSeparator
also supports a zero-length VL kick in directly authored input.

By default the importer retains all 60,000 particles per injection and all 65
injections in the source case. For a quick workflow check, add
`--particles-per-batch 64`. This selects the first 64 source rows per batch,
preserving their coordinates and increasing their individual weights so that
the total physical charge stays unchanged. Such a small sample checks the
workflow; it is not a statistically converged PIC simulation.

The default backend is GPU; select `--backend cpu` if needed. The runner validates
the input before tracking, propagates errors, and writes `completed.json` only
after the requested number of turns finishes. PASS creates a dated run directory
under the case's `tracking` directory. The final console line reports its path,
duration, surviving/lost counts, and remaining reserved particles.

## Comparison stages and clocks

| Option | Behavior |
| --- | --- |
| `--stage external` | External ring maps and bumps; aperture, source error multipoles, RF and PIC are omitted unless RF is explicitly supplied. |
| `--stage aperture` | Adds source apertures, source multipoles and the finite-length powered ES. Requires explicit V or VL, plus gap. Particles start at the injection plane; electrode and aperture losses are evaluated when they pass the corresponding lattice elements. This is the default. |
| `--stage pic` | Adds the source's enabled internal PIC kicks and longitudinal slicer. |
| `--clock reference` | Samples the prescribed machine clock at the current turn, by inverting its integrated revolution frequency. This clock does not follow the instantaneous energy of a tracked bunch. |
| `--clock particle` | Samples each particle's physical passage time `t = bunch.t0 - z/(beta*c)` at each kick center. Nominal bunch centers do not enter this time. |
| `--particle-clock-origin cisp` | Default: translate each CISP element's local waveform clock to the common injection clock, using a constant time offset of `-s/(beta_inj*c)`. This aligns the first ideal reference passage with CISP time zero. |
| `--particle-clock-origin injection` | Interpret raw timestamps as calibrated to the injection-point clock, without a time offset. |
| `--es-voltage V` / `--es-vl VL` | Choose exactly one for aperture/pic stages: signed interplate potential difference (V), or its total longitudinal integral (V m). Written to generated input as `V (V)` or `VL (V m)` and recorded in the manifest. |
| `--es-gap G` | Required for aperture/pic stages: clear electrode gap (m). |
| `--es-length L` | Supplied ES length before injection, default 1 m. This includes the last 0.25 m of drift049a and all 0.75 m of drift049b at the default. Tilt never rescales L or reference flight time. |
| `--rf-directory PATH` | Reads the source-named RF voltage and phase files from this directory and converts them to a PASS physical-time RF table. |
| `--rf-voltage-unit V` | Required when RF is enabled: choose `V` (current CISP source convention) or `MV` for confirmed historical programs. The output table is always in volts. |
| `--no-snapshots` | Keeps statistics but omits large particle snapshots for timing or convergence runs. |
| `--grid-width W --grid-cells N` | Sets full PIC domain width in meters and cells per axis for convergence checks; defaults are 0.5 m and the source's 128 cells. |

RF files are not included in the supplied latest CISP folder. Omitting
`--rf-directory` disables RF explicitly and records that fact in the conversion
manifest. Do not compare such a run against an RF-enabled result as an identical
physical case. Confirm the provenance of any replacement RF files first. Once
the first 100 turns are checked, use `--turns 5000` with verified RF files for the
acceleration stage.

RF input requires an explicit voltage unit as well as verified file provenance. Current CISP `rf.cpp` reads
volts and divides by 1e6 internally for its energy units. Some historical files
have values such as 0.0002 that may represent 200 V in MV units; a filename is
insufficient evidence. Select `--rf-voltage-unit MV` only after confirming that
convention. The importer records the selected unit and never guesses from scale.

The importer is intentionally specific to this CISP command dialect and rejects
unsupported active element types. It converts the source CSV's linear energy
offset into PASS momentum deviation instead of merely relabeling the sixth
column. It preserves fixed injection energy as the circulating reference gains
energy. Dipole integrals map from source `fint1`/`fint2` to PASS `Fint`/`Fintx`.
PASS retains its existing rule: positive `Fintx` sets the exit integral, while
`Fintx <= 0` inherits the entrance `Fint`. The importer rejects a nonzero source
entrance integral with a zero exit integral before writing converted files,
because that pair cannot be represented by this rule. This includes source
magnets already split into separate elements with a zero integral at the join.
The original BRing case has 48 such entrance halves (`fint1=0.489`, `fint2=0`),
starting with `BRMG41D01_000`; it currently stops at this validation. Its magnet
representation must be revised before this importer can generate that case.
Dipole body maps use at least eight Yoshida slices to resolve the source optical
matrix. Internal slicing applies one real entrance, all body slices, and one
real exit; it introduces no additional fringe maps at slice boundaries.

The external-element maps are different implementations, so exact multi-turn
particle equality is not assumed. CISP adapts its free-space PIC domain to
1.1 times the maximum absolute transverse extent at each kick; this generated
PASS input uses a fixed 0.5 m square domain and 128 by 128 cells. Compare domain,
grid, slicing and macro-particle convergence before interpreting PIC differences.
The manifest lists these differences, the clock and all source dependencies.

Check RF separately before combining it with PIC. The inspected CISP RF kernel
additionally rescales energy deviation by the ratio of new to old reference
velocity during acceleration; PASS applies its physical RF energy kick and
rescales the local time coordinate to preserve particle passage times.
Matching the reference-energy program does not validate the
six-dimensional particle motion or establish identical RF maps. Keep the CISP
executable/source version with any reference data.

CISP's automatic slice count is even for elements longer than its maximum
slice length; the importer preserves that count for PIC integration weights.
CISP applies SC before each source slice, whereas PASS uses midpoint kicks.
Aperture sampling also differs: CISP checks element/slice entrances and standard
PASS elements check exits and internal SC locations. The finite ES retains each
source vacuum aperture along its corresponding segment, including the polygon
over the final 0.75 m, and stops particles at the first segment-wall contact.
This finite chamber differs physically from a single entrance cut. Check
trajectory and loss-position convergence before comparing survival; other
elements retain their existing aperture sampling.

Clock origins matter: upstream CISP tables differ from downstream tables by
about one revolution. Their original timestamps are preserved. For particle-time
comparisons, the default constant offsets implement CISP's local first-passage
clock convention; the manifest records each offset. Hardware waveforms calibrated
against injection time should use the `injection` option. The downstream source
tables start at zero, whereas some particles in the first long bunch arrive
earlier. Those samples receive the value at time zero and one warning per Bump.
Before each plane's first supplied time the first value is held, and after its
last supplied time the last value is held. Supply negative-time samples when
the pre-injection field changes; otherwise this explicitly models a constant
leading plateau. End the ramp with zero to switch the magnet off.

Horizontal and vertical waveform files may contain different numbers of rows
and different time ranges, including disjoint ranges. The importer merges the
union of their time nodes, holding each plane's endpoints independently without
cropping either waveform. TFS headers retain each original range for warnings.
Bump aperture losses are checked once at its exit, regardless of slice count;
internal PIC nodes retain field-domain validity checks. The Bump model assumes
fixed-energy injection and ideal fields without fringe or pole-face effects.

## Particle snapshots and plots

Snapshots are taken immediately after Injection at s = 0, before subsequent ring
transport: turns 0, 1, then every fifth injection, the last injection (turn 64),
turn 65, turn 72, and the final requested turn when present. Turn indices are
zero-based. HDF5 stores every born particle, including lost particles, without
plotting subsampling. Pending reservations are excluded and counted in
`NumPending`. Three optional integer output columns identify particles and batches:

- `particle_id`: stable identity, independent of array row order;
- `injection_turn`: the turn when the particle was born;
- `injection_batch`: zero-based injection batch within the source bunch.

These are generated during export from `abs(tag)` and Injection's compact batch
history, not stored as arrays in `ParticlePool`. Injection owns a temporary
reservation-ID map while future batches remain; sorting updates that map and
the final injection releases it. The `Include injection metadata` output option
is enabled for this example's batch plots and remains off by default elsewhere.

The existing six coordinates, signed `tag`, loss turn and loss position are
also stored. Positive tags survive, negative tags have been lost, and zero tags
in memory are pending. Statistics report `numAlive`, `numInjected`, `numPending`
and loss percentage relative to born particles.

To plot one completed run, use its dated directory, not a parent containing
several runs:

```powershell
python example/06_injection_painting/analyze_results.py tests/codex/painting_reference/tracking/YYYY_MMDD/HHMM_SS
```

The analysis writes x-px, y-py and x-y panels for the injection history, a density
plot using all survivors, and seven pages covering ten batches each (five on the
last page). Other batches appear in gray. Scatter plots display at most 2,500
particles per highlighted batch in deterministic ID order; saved distributions
remain complete. All frames share axes for easier comparison. An incomplete
injection run produces evolution plots but is not labeled injection complete.
The batch population table and stacked bar chart report all injected, surviving
and lost particles per batch at injection completion.
Open `painting_plots/index.html` in a browser to select evolution frames or batch
pages, move the frame slider, or play the sequence. The gallery uses the generated
PNG files and works offline. Keep it together with those images when sharing it.
`snapshot_counts.csv` and `plot_manifest.json` accompany the figures. The script
defaults to beam 0 / bunch 0; use `--beam-id` and `--bunch-id` to select another
group. It rejects mixed runs with duplicate turns and requires injection metadata.

When comparing statistics, select the same physical location: CISP `pm2` is at
the injection point, while `pm1` lies after the downstream bumps (PASS s = 1.9 m).
Align turn numbers and the survival definition as well as input switches and
particle counts. A small-sample result must not be compared directly with the
full 60,000-particle source as a numerical agreement test.

To generate overlaid statistics and difference tables, add `--cisp-pm2 PATH`
and/or `--cisp-pm1 PATH` to the analysis command. Each path should select the
CISP monitor's `bunch1.csv`. Add `--comparison-note "..."` to state the RF, PIC,
aperture and sampling conditions; this note appears on the plot and in a JSON
record with the reference-file hash. Different conditions can be plotted for
diagnosis, but an overlay alone does not certify numerical agreement. Missing
reference turns and duplicate statistics rows are rejected.

When available, comparison plots also include RMS bunch length, RMS momentum
spread and reference kinetic energy. CISP energy columns in MeV/u are converted
to PASS eV/u before subtraction; the JSON record lists every column conversion.
CISP's energy-derived momentum spread uses its linear conversion convention,
whereas PASS tracks mechanical momentum deviation.

### Physical-time RF and slices

RF conversion now writes `rf_physical_time.tfs` with TIME, VOLTAGE, FREQUENCY and PHASE columns. The RF phase integrates the prescribed carrier frequency and adds the unwrapped phase modulation. The initial cavity passage and the common waveform epoch are distinct. PASS stores `z = beta*c*(T-t)` and samples `t = T-z/(beta*c)`; nominal bunch slots do not supply arrival corrections. SC requires the latest explicit `z_periodic` slices, formed by temporarily folding z into `[-C/2,C/2)` without changing particle coordinates. RF does not rescale saved slice intervals or automatically run Slicer.

Diagnostic tables now default to gzip-1 + shuffle HDF5 (`.h5`). The analysis scripts
accept both HDF5 and legacy TFS output; set `output_format="tfs"` on the
monitor (or initial-distribution `BunchConfig`) to request TFS explicitly.
Set `output_format="hdf5"` to write uncompressed HDF5; the default
`output_format="hdf5-gzip1"` enables gzip level 1 and shuffle. Both use `.h5` files.
StatMonitor also writes every recorded row to CSV in batches of 100 turns
by default, configurable with `write_interval_turns`. Slicer slice summaries
remain TFS/CSV. See [table output formats](../../docs/source/en/monitor/table_output.rst).
