# PASS

[English](README.md) | [中文](README-zh.md)

PASS (Particle Accelerator Simulation Studio) is a Python program for six-dimensional particle tracking and beam-dynamics analysis. It supports element-based and Twiss-map transport, injection, RF systems, space charge, wakefields, and beam diagnostics, with CPU and optional NVIDIA GPU execution.

[User manual](https://changmx.github.io/PASS/en/index.html) · [Examples](example) · [Releases](https://github.com/changmx/PASS/releases) · [Issues](https://github.com/changmx/PASS/issues)

## Install

Use Python **3.11 or newer**. Choose PyPI or GitHub, then select one of the three configurations listed under that method.
The full installation includes CPU, NVIDIA GPU, and GUI support. GPU execution requires a compatible NVIDIA GPU, driver, and CUDA environment. CPU-only and GUI installations do not require CUDA.

### Install from PyPI

**Full installation (CPU + GPU + GUI):**

```bash
python -m pip install "pass-sim[gui,cuda]"
```

**CPU computation only (command line and Python API):**

```bash
python -m pip install pass-sim
```

**GUI installation (includes CPU computation):**

```bash
python -m pip install "pass-sim[gui]"
```

After installing one configuration, run the following command to display the installed PASS version:

```bash
pass-run --version
```

If you installed the GUI, launch it with `pass-gui`.

### Install from GitHub

First clone the source and enter the repository:

```bash
git clone https://github.com/changmx/PASS.git
cd PASS
```

Alternatively, open the [GitHub repository](https://github.com/changmx/PASS), select **Code → Download ZIP**, and extract it. Enter the directory containing `pyproject.toml`.
From that directory, select one of the following configurations:

**Full installation (CPU + GPU + GUI):**

```bash
python -m pip install --editable ".[gui,cuda]"
```

**CPU computation only (command line and Python API):**

```bash
python -m pip install --editable .
```

**GUI installation (includes CPU computation):**

```bash
python -m pip install --editable ".[gui]"
```

An editable installation uses the code in this source directory directly; Python source edits take effect in subsequent runs.
After installation, run the following command to display the installed PASS version:

```bash
pass-run --version
```

If you installed the GUI, launch it with `pass-gui`.

Both installation methods also support `python -m PASS --version`. Run `pass-run --help` or `pass-run -h` to see input modes, options, path rules, and examples.

## Run a simulation

**Graphical interface:** install the `gui` extra, then launch:

```bash
pass-gui
```

`python -m PASS.gui` launches the same interface using the selected Python interpreter.

Open or create an input, apply parameter changes, select **Validate**, and start the simulation from **Run**. Use **Plot** to inspect results. A `.passproj` file stores configurations and their input dependencies. See the [GUI workflow](https://changmx.github.io/PASS/en/gui.html) and [project guide](https://changmx.github.io/PASS/en/project_files.html).

**Command line:** after installing PASS, run JSON inputs or a saved project without the GUI or Qt:

```bash
# Run one beam from an existing JSON input.
pass-run path/to/beam0.json
# Run two beams together from two JSON inputs.
pass-run path/to/beam0.json path/to/beam1.json
# Run the beam selection saved in a project.
pass-run path/to/example.passproj
# Name the inputs explicitly instead of using positional arguments.
pass-run --beam0 path/to/beam0.json
pass-run --beam0 path/to/beam0.json --beam1 path/to/beam1.json
pass-run --passproj path/to/example.passproj
# Set a different output root for this run.
pass-run path/to/beam0.json --output results
pass-run path/to/example.passproj --output "results/project run"
```

The commands accept input file paths; relative arguments are resolved from your terminal's working directory. Quote paths containing spaces. The file extension selects JSON or project loading. Projects use their saved Beam 0/Beam 1 selections, falling back to the active input when Beam 0 has not been saved. They do not run every configuration, and a project cannot be combined with a second input argument. Tracking validates and archives the selected inputs and their dependencies before execution.

Choose either positional inputs or named input options for a command; do not mix them. `--beam0` and `--beam1` accept JSON files, and `--beam1` requires `--beam0`. `--passproj` accepts one `.passproj` file and cannot be combined with either beam option.

`python -m PASS` accepts the same arguments and runs the same code using the selected Python interpreter:

```bash
python -m PASS path/to/beam0.json
python -m PASS "path with spaces/example.passproj"
pass-run --help
pass-run -h
```

**Output directories:** the normal default output folder is `output`. Its location depends on the input and launch mode:

| Launch mode | Output root | Results beneath that root |
| --- | --- | --- |
| JSON through `pass-run`, `python -m PASS`, or the Python API | The first JSON's `Output directory`, relative to that JSON's directory; if omitted or set to `default` (case-insensitive), use `output` beside that JSON | `<YYYY_MMDD>/<HHMM_SS>/` |
| Saved `.passproj` through the command line | The saved run output directory, or `output` if unset; relative paths are resolved beside the project | `<YYYY_MMDD>/<HHMM_SS>/` |
| GUI | The **Run** page's output directory, initially `output`; relative paths use the JSON/project directory (the working directory for an unsaved project) | `<YYYY_MMDD>/<HHMM_SS>/` |

`--output DIR` overrides the output root for one or two JSON inputs and for `.passproj` files. It takes precedence over the input or saved project setting and applies only to that run; relative paths are resolved from the terminal's working directory at launch. The original JSON or project file remains unchanged. `python -m PASS` supports the same option.

Each normal run saves its input snapshot inside its result directory, at `<output>/<YYYY_MMDD>/<HHMM_SS>/input/`. This folder contains the original configuration values, execution JSON, copied input dependencies, and `run.json` record. JSON, project, and GUI runs all use the same dated result layout; a suffix is added if needed to avoid reusing an existing result directory. Generated JSON inputs still use `./output`. JSON inputs that omit `Output directory` or specify `default` now use `output` beside the first JSON instead of the repository or installation directory. See the [input guide](https://changmx.github.io/PASS/en/input_generation.html) and [project guide](https://changmx.github.io/PASS/en/project_files.html) for output details.

**Inspect a project:** a `.passproj` file is a standard ZIP container. Open it with a ZIP-compatible archive application, or make a copy, change the copy's extension to `.zip`, and extract it. Read `manifest.json` for the project manifest and run settings, `configs/*.json` for simulation inputs, `assets/` for dependencies, and `recipes/` for generation settings. The GUI's **Project contents** also provides previews. Save changes through PASS so the stored checksums and dependency index stay consistent.

**Python workflow:** the [input guide](https://changmx.github.io/PASS/en/input_generation.html) covers input generation, validation, execution, and output. Example scripts are provided in the source repository. If you installed from PyPI, first download or clone the repository to obtain the examples. Then run the distribution example from the repository root:

```bash
cd example/01_generate_distribution
python generate_input.py --case longi-gaussian
pass-run --beam0 beam0_longi_gaussian.json
python analyze_results.py --case longi-gaussian
```

The generator writes the case input; `pass-run` saves results under the example's `output/` directory; the analyzer uses the latest run for that case. The distribution and RF examples also provide `run_simulation.py --case all` helpers for running their predefined cases serially through the same CLI. See the [example README](example/01_generate_distribution/README.md) for parameters, other cases, and output files.

An existing input can also be checked before execution:

```bash
python -m PASS.validation path/to/beam.json --report validation-report.json
```

## Reference

- [Coordinates and injection](https://changmx.github.io/PASS/en/injection.html)
- [Elements](https://changmx.github.io/PASS/en/element/index.html) and [Twiss maps](https://changmx.github.io/PASS/en/twiss.html)
- [Space charge](https://changmx.github.io/PASS/en/space_charge.html), [wakefields](https://changmx.github.io/PASS/en/wake_field.html) and [electron clouds: frozen kicks and prescribed-beam build-up](https://changmx.github.io/PASS/en/electron_cloud.html)
- [Intrabeam scattering](https://changmx.github.io/PASS/en/ibs.html): Gaussian growth rates, kinetic kicks and local binary collisions on CPU/GPU
- [Monitors and output formats](https://changmx.github.io/PASS/en/monitor/index.html)
- [Spectral analysis and FMA](https://changmx.github.io/PASS/en/spectral_analysis.html): standalone NumPy functions and the GUI **Analysis** workspace; CSV/TSV/TXT/DAT, TFS, HDF5, NPY and NPZ inputs.

WakeField file models accept canonical wake TFS only. Convert CSV/TXT/HEADTAIL data with `python -m PASS.tool.wake_conversion` or **Import wake…** in the GUI conversion tool; the [wakefield guide](https://changmx.github.io/PASS/en/wake_field.html#wake-tfs-en) explains units, physical conventions and examples.

With the `docs` extra installed, run this command from the repository root to build both languages:

```bash
python -m sphinx -b html -W --keep-going docs/source docs/build/html
```

Open `docs/build/html/index.html` after a successful build.

## Contributing

Follow [AGENTS.md](AGENTS.md) for physics, documentation, and formatting conventions. Install `.[dev]` and use YAPF 0.43.0 with `pyproject.toml`; embedded CUDA is checked with `python tools/format_cuda.py --check`. The `tests/` directory is maintained locally and excluded from Git, so a fresh clone does not contain the test suite.

PASS is distributed under the [Apache License 2.0](LICENSE).
