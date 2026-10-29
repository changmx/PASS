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

### Example project

The examples below assume that `/home/myprojpath` contains a valid PASS input `beam0.json` and this project description:

```text
/home/myprojpath/
├── myproject.passproj
├── beam0.json
├── quadrupole_ramp.tfs    # optional: include it if beam0.json references it
└── output/               # created when a simulation runs
```

`myproject.passproj` is an ordinary JSON text file. For a single-beam project, its contents can be:

```json
{
  "format": "pass-project",
  "format_version": 3,
  "name": "My simulation",
  "configs": ["beam0.json"],
  "active_config": "beam0.json",
  "run": {
    "beam0": "beam0.json",
    "beam1": null,
    "output_directory": "output"
  }
}
```

Project-description paths are relative to `/home/myprojpath`. File references inside `beam0.json` are relative to that JSON's directory; for example, `"Ramping file": "quadrupole_ramp.tfs"` refers to `/home/myprojpath/quadrupole_ramp.tfs`. There is no required `configs/` directory. Replace `/home/myprojpath` with your actual project path.

### Command-line execution

There are **three input modes**: one JSON for a single beam, two JSON files for two beams, and a project selecting its saved beam inputs. A project can be specified by its `.passproj` file or by its directory. Command-line execution does not require the GUI or Qt.

With the example directory above, each of the following commands can be run from **any working directory**:

```bash
# Run beam0.json directly, using the settings in that JSON.
pass-run /home/myprojpath/beam0.json

# Run the beam selection and output settings saved in myproject.passproj.
pass-run /home/myprojpath/myproject.passproj

# Open the project description in this directory, then run its saved selection.
pass-run /home/myprojpath
```

For a directory argument, PASS first uses `project.passproj` if it exists. Otherwise, it uses the only `.passproj` file directly in that directory, so the example above selects `myproject.passproj`. If there are several `.passproj` files and no `project.passproj`, specify the intended project file explicitly. PASS does not search subdirectories.

After entering the project directory, the equivalent relative-path commands are:

```bash
cd /home/myprojpath
pass-run beam0.json
pass-run myproject.passproj
pass-run .
```

These are alternative commands; choose one to launch a run. Projects use `run.beam0` and optional `run.beam1`, falling back to `active_config` when Beam 0 is unset. They do not run every configuration in `configs`. Tracking validates and archives the selected inputs and their dependencies before execution.

Named input options provide another spelling of the same commands:

```bash
pass-run --beam0 /home/myprojpath/beam0.json
pass-run --passproj /home/myprojpath/myproject.passproj
pass-run --passproj /home/myprojpath
```

For **two-beam execution**, first provide a second valid input `/home/myprojpath/beam1.json`. Then use either of these equivalent commands:

```bash
pass-run /home/myprojpath/beam0.json /home/myprojpath/beam1.json
pass-run --beam0 /home/myprojpath/beam0.json --beam1 /home/myprojpath/beam1.json
```

To run both through the project instead, add `"beam1.json"` to its `configs` list and set `run.beam1` to `"beam1.json"`, then run `pass-run /home/myprojpath/myproject.passproj`. A project supplies its own beam selection and cannot be combined with a second input argument. Choose positional arguments or named input options for each command; do not mix them. `--beam1` requires `--beam0`, and `--passproj` cannot be combined with either beam option.

`pass-run` and `python -m PASS` are **two launchers for the same execution logic**, with the same arguments. Use the module form to select a particular Python environment:

```bash
python -m PASS /home/myprojpath/beam0.json
python -m PASS /home/myprojpath/myproject.passproj
python -m PASS /home/myprojpath
```

Relative command-line paths use the terminal's working directory. Quote paths containing spaces. Run `pass-run --help` or `python -m PASS --help` for all options.

### Outputs and editing

For the project description above, results go under `/home/myprojpath/output/<YYYY_MMDD>/<HHMM_SS>/`, regardless of the directory from which you launch it. Running `beam0.json` directly uses its own `Output directory` setting; if it is omitted or set to `default` (case-insensitive), the output root is also `/home/myprojpath/output`.

| Launch mode | Output root | Results beneath that root |
| --- | --- | --- |
| JSON through `pass-run`, `python -m PASS`, or the Python API | The first JSON's `Output directory`, relative to that JSON's directory; if omitted or set to `default` (case-insensitive), use `output` beside that JSON | `<YYYY_MMDD>/<HHMM_SS>/` |
| Project file or directory through the command line | The project's `run.output_directory`, or `output` if unset; relative paths are resolved beside the project description | `<YYYY_MMDD>/<HHMM_SS>/` |
| GUI | The **Run** page's output directory, initially `output`; relative paths use the JSON/project directory (the working directory for an unsaved project) | `<YYYY_MMDD>/<HHMM_SS>/` |

`--output DIR` overrides the output root for any input mode, only for that run. An absolute path works from any working directory:

```bash
pass-run /home/myprojpath/myproject.passproj --output /home/myprojpath/results
```

The result root is `/home/myprojpath/results`. A relative override uses the terminal's working directory, as this example makes explicit:

```bash
cd /home/myprojpath
pass-run myproject.passproj --output "results/test run"
```

This writes beneath `/home/myprojpath/results/test run/<YYYY_MMDD>/<HHMM_SS>/`. `--output` takes precedence over the JSON/project settings without changing their files. `python -m PASS` supports the same option.

Each normal run saves an input snapshot in its result directory's `input/` subdirectory. It contains original configuration values, execution JSON, copied dependencies, and the `run.json` record. JSON, project, and GUI runs share the dated layout; a suffix is added when needed to avoid reusing a result directory. See the [input guide](https://changmx.github.io/PASS/en/input_generation.html) and [project guide](https://changmx.github.io/PASS/en/project_files.html) for details.

Edit `myproject.passproj`, `beam0.json`, and text dependencies with a text editor. In the GUI, use **File → Reload project** after external edits and before saving or running; external-change checks protect against overwriting or running stale inputs. Move or share the directory with all referenced files, because the `.passproj` description alone is not a complete project. **Export runnable input bundle** can package the selected inputs and dependencies for sharing.

### Graphical interface

Install the `gui` extra, then launch with either `pass-gui` or `python -m PASS.gui`. For the example above, select **File → Open project** and choose `/home/myprojpath/myproject.passproj`, or **File → Open project folder** and choose `/home/myprojpath`. To work with just the beam input, use **File → Open JSON** and choose `/home/myprojpath/beam0.json`.

Apply parameter changes, select **Validate**, and start the simulation from **Run**. Use **Plot** to inspect results. See the [GUI workflow](https://changmx.github.io/PASS/en/gui.html) and [project guide](https://changmx.github.io/PASS/en/project_files.html).

### Python workflow

The [input guide](https://changmx.github.io/PASS/en/input_generation.html) covers input generation, validation, execution, and output. Example scripts are provided in the source repository. If you installed from PyPI, first download or clone the repository to obtain the examples. Then run the distribution example from the repository root:

```bash
cd example/01_generate_distribution
python generate_input.py --case longi-gaussian
pass-run --beam0 beam0_longi_gaussian.json
python analyze_results.py --case longi-gaussian
```

The generator writes the case input; `pass-run` saves results under the example's `output/` directory; the analyzer uses the latest run for that case. The distribution and RF examples also provide `run_simulation.py --case all` helpers for running their predefined cases serially through the same CLI. See the [example README](example/01_generate_distribution/README.md) for parameters, other cases, and output files.

An existing input can also be checked before execution:

```bash
python -m PASS.validation /home/myprojpath/beam0.json --report /home/myprojpath/validation-report.json
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
