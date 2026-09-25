# PASS

[English](README.md) | [中文](README-zh.md)

PASS (Particle Accelerator Simulation Studio) is a Python program for six-dimensional particle tracking and beam-dynamics analysis. It supports element-based and Twiss-map transport, injection, RF systems, space charge, wakefields, and beam diagnostics, with CPU and optional NVIDIA GPU execution.

[User manual](https://changmx.github.io/PASS/en/index.html) · [Examples](example) · [Releases](https://github.com/changmx/PASS/releases) · [Issues](https://github.com/changmx/PASS/issues)

## Install

Use Python **3.11 or newer** in a virtual environment. Install from a source checkout:

```bash
git clone https://github.com/changmx/PASS.git
cd PASS
python -m pip install --editable .
```

Optional components can be installed together, for example `python -m pip install --editable ".[gui,docs]"`.

| Extra | Purpose |
| --- | --- |
| `gui` | Graphical configuration, execution, plotting, and calculation tools |
| `cuda` | GPU tracking; requires a compatible NVIDIA GPU, driver, and CUDA environment |
| `docs` | Build the English and Chinese manuals locally |
| `conversion` | SDDS conversion without the GUI; already included in `gui` |
| `dev` | Repository Python formatter |

CPU tracking does not require CUDA.

## Run a simulation

**Graphical interface:** install the `gui` extra, then launch:

```bash
python -m PASS.gui
```

Open or create an input, apply parameter changes, select **Validate**, and start the simulation from **Run**. Use **Plot** to inspect results. A `.passproj` file stores configurations and their input dependencies. See the [GUI workflow](https://changmx.github.io/PASS/en/gui.html) and [project guide](https://changmx.github.io/PASS/en/project_files.html).

**Python workflow:** the [input guide](https://changmx.github.io/PASS/en/input_generation.html) covers input generation, validation, execution, and output. To run the existing distribution example from the repository root:

```bash
cd example/01_generate_distribution
python generate_input.py --case longi-gaussian
python run_simulation.py --case longi-gaussian
python analyze_results.py --case longi-gaussian
```

The generator writes the case input; the runner saves results under the example's `output/` directory; the analyzer uses the latest run for that case. See the [example README](example/01_generate_distribution/README.md) for parameters, other cases, and output files.

An existing input can also be checked before execution:

```bash
python -m PASS.validation path/to/beam.json --report validation-report.json
```

## Reference

- [Coordinates and injection](https://changmx.github.io/PASS/en/injection.html)
- [Elements](https://changmx.github.io/PASS/en/element/index.html) and [Twiss maps](https://changmx.github.io/PASS/en/twiss.html)
- [Space charge](https://changmx.github.io/PASS/en/space_charge.html) and [wakefields](https://changmx.github.io/PASS/en/wake_field.html)
- [Monitors and output formats](https://changmx.github.io/PASS/en/monitor/index.html)

With the `docs` extra installed, run this command from the repository root to build both languages:

```bash
python -m sphinx -b html -W --keep-going docs/source docs/build/html
```

Open `docs/build/html/index.html` after a successful build.

## Contributing

Follow [AGENTS.md](AGENTS.md) for physics, documentation, and formatting conventions. Install `.[dev]` and use YAPF 0.43.0 with `pyproject.toml`; embedded CUDA is checked with `python tools/format_cuda.py --check`. The `tests/` directory is maintained locally and excluded from Git, so a fresh clone does not contain the test suite.

PASS is distributed under the [Apache License 2.0](LICENSE).
