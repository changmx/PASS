## 🌐 Language
[**中文**](README-zh.md) | [English](README.md)

# PASS (Particle Accelerator Simulation Studio)

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://changmx.github.io/PASS/) [![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![Release](https://img.shields.io/github/v/release/changmx/PASS)](https://github.com/changmx/PASS/releases/latest)

PASS is a versatile particle-accelerator simulation platform for Windows and Linux. It provides CPU and NVIDIA GPU execution backends, aiming to deliver high-performance, extensible, and reproducible six-dimensional particle tracking and beam-dynamics analysis.

## Documentation

The complete documentation is published at [changmx.github.io/PASS](https://changmx.github.io/PASS/), with separate [English](https://changmx.github.io/PASS/en/) and [中文](https://changmx.github.io/PASS/zh/) sections. It contains the physics model, input schema, supported elements and monitors, coordinate conventions, and worked examples. Please use the documentation site for detailed usage and reference material; this README is intentionally a short project overview.

## Installing

PASS currently installs from a source checkout. Python 3.10 or newer is required.

```bash
git clone https://github.com/changmx/PASS.git
cd PASS
python -m pip install --editable .
```

To install the optional CUDA dependencies for GPU tracking, use:

```bash
python -m pip install --editable ".[cuda]"
```

The CUDA toolkit and a compatible GPU are required for the CUDA backend. CPU tracking does not require CUDA.

## Graphical Interface

The optional cross-platform PASS interface provides separate configuration, run, and plotting pages. Install it with:

```bash
python -m pip install --editable ".[gui]"
```

Then launch it from the repository or any installed environment:

```bash
python -m PASS.gui
# or
pass-gui
```

The configuration page loads, edits, validates, and exports PASS JSON input. The run page starts the existing PASS tracking entry point in a separate process. The plotting page loads CSV or TFS result files and lets the user select a numeric column for a first-pass curve view.

## Functionality

- six-dimensional particle tracking through accelerator lattices;
- element-by-element and Twiss-based tracking workflows;
- configurable injection and multi-bunch beam distributions;
- RF cavities, magnets, collective-effect interfaces, and beam monitors;
- Python tools for generating JSON input files and analysing TFS/CSV output;
- CPU and optional CUDA execution paths.



## Development

The `tests/` directory is maintained locally and excluded from Git. Fresh clones
do not include the test suite. If you have the local test files, install the
package in editable mode and run the suite from the repository root:

```bash
python -m pytest
```

Default discovery includes `tests/unit` and `tests/integration`, including the full
space-charge simulations. For grouped runs, individual cases, saved-field analysis,
and explicit local Codex regressions, see the local guide at
`tests/integration/space_charge/README.md`.

Space charge supports `pic`, `frozen`, and `quasi-frozen` tracking through the
`Method` / `Solver` configuration. Run `python -m tests.integration.space_charge analytic`
for analytic-field, particle-kick, and parameter-evolution comparisons with plots.

Bug reports and feature requests are welcome through the [GitHub issue tracker](https://github.com/changmx/PASS/issues). Contributions should include tests and documentation updates where appropriate.

## License

PASS is distributed under the [Apache License 2.0](LICENSE).
