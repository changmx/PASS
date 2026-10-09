# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [0.6.1] - 2026-10-09

### Added

- Transverse `uniform-phase` distribution with independently uniform phase-space ellipses in both planes.

### Changed

- Renamed the transverse `uniform` distribution to `uniform-real`; old inputs must use the new name.

---

## [0.6.0] - 2026-10-08

### Added

- Slow-extraction particle collection and spill monitoring, with CPU and GPU support.
- Standalone FFT, refined FFT, and frequency-map analysis with a GUI workspace.
- Dynamic-aperture scans, particle-file injection, and analysis and export tools.
- Bunch-by-bunch transverse feedback with FIR design tools and a GUI coefficient generator.
- HIAF RF data conversion and an exciter voltage-to-kick-angle calculator.
- Unified `pass-run` command-line interface for JSON inputs and saved projects, without GUI dependencies.

### Changed

- Consolidated ParticleMonitor output into one file per monitor and beam, including injection coordinates; ParticleMonitor and Injection now default to uncompressed HDF5.
- Derived the design reference clock automatically from injection and RF settings; old inputs must remove `Reference clock`.
- Replaced exciter voltage, gap, and length inputs with `Kick angle (rad)`.
- Used tabulated ion masses in tracking and improved GPU execution, data reading, and input validation.

### Fixed

- Corrected MAD-X dispersion and chromaticity conversion to momentum deviation.
- Preserved exciter phase continuity across frequency-sweep periods.
- Protected existing output and committed history, and corrected injection snapshot handling.

### Removed

- Joint beam-beam checkpoint save/restore and stopped-run continuation.

---

## [0.5.0] - 2026-10-03

### Added

- Beam-beam, electron-cloud, intrabeam scattering, and electron-cooling simulations.
- Magnetic field and alignment errors.
- Time-dependent magnet strengths.
- HDF5/SDDS data conversion and wake-data import tools.

### Changed

- Standardized wake TFS input and default compressed HDF5 output.
- Updated RF time-window handling, GUI workflows, and wake-field performance.
- Updated dependencies and expanded documentation.

### Fixed

- Corrected periodic slicing and wake observation windows during acceleration.

---

## [0.4.0] - 2026-09-17

### Added

- Space-charge and wake-field simulations.
- GUI for configuration, validation, tracking, and analysis.
- Multi-turn injection, bump elements, and injection-painting example.
- Distribution and phase-advance monitors.

### Changed

- Improved RF timing, reference-state handling, and element tracking.
- Expanded documentation and examples.
- Raised the minimum Python version to 3.11.

---

## [0.3.0] - 2026-08-28

### Added
- Completed symplectic element-by-element and Twiss tracking for both CPU and GPU versions.

### Changed
- (None)

### Fixed
- (None)

### Removed
- (None)
