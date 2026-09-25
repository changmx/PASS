Input File Generation (Command-Line Mode)
==========================================

PASS reads simulation inputs from JSON files. Use the Python schema classes to configure global parameters, bunches and a lattice sequence, then call ``generate_input()``. For graphical configuration, see :doc:`gui`.

Schema construction checks declared field types and constraints; :doc:`input_validation` additionally checks the complete input, command ordering and external files. ``generate_input()`` writes the configuration and does not itself replace that full check.

.. _en-minimal-input-example:

Quick start
----------------------

After installing PASS, save the following script as ``input/generate_beam0.py`` under the repository root. Create the ``input`` directory if needed. This example uses 2,048 macro particles, 64 CPU turns, a smooth linear lattice and Gaussian distributions; no external lattice file is required.

.. code-block:: python

   from pathlib import Path

   from PASS.para.api import generate_input
   from PASS.para.schema.main import MainConfig
   from PASS.para.schema.bunch import BunchConfig, InjectionItem
   from PASS.para.schema.sequence import Sequence
   from PASS.para.schema.monitors import StatMonitorItem
   from PASS.para.smooth import generate_smooth_twiss
   from PASS.validation import validate_file

   main = MainConfig(
       beam_name="proton",
       num_proton=1, num_neutron=0, num_electron=1,
       gamma_t=4.8, circumference=251.327,
       num_turns=64, backend="cpu", output_dir="output", is_plot=False,
   )
   items, names, circumference = generate_smooth_twiss(
       circumference=main.circumference,
       qx=4.8, qy=4.4, num_points=17,
       longitudinal_transfer="off",
   )
   bunch = BunchConfig(
       kinetic_energy=45e6,
       num_real_particles=100_000_000_000,
       num_macro_particles=2048,
       beta_x=items[0].beta_x, beta_y=items[0].beta_y,
       alpha_x=0.0, alpha_y=0.0,
       emit_x=2e-6, emit_y=2e-6,
       sigma_z=0.1, dp=0.001,
       dist_trans="gaussian", dist_longi="gaussian",
   )
   seq = Sequence()
   seq.add("injection", InjectionItem(s=0.0, random_seed=2026, bunches=[bunch]))
   for name, item in zip(names, items):
       seq.add(name, item)
   seq.add("stat1", StatMonitorItem(s=0.0, write_interval_turns=16))

   output_path = Path(__file__).resolve().parent / "beam0.json"
   generate_input(main, seq, str(output_path))
   report = validate_file(str(output_path))
   if not report.ok:
       raise ValueError(report.text())
   print(f"Validated input: {output_path}")

Run these commands from the repository root:

.. code-block:: console

   python input/generate_beam0.py
   python -c "from PASS.main import main; main('input/beam0.json', raise_errors=True)"

The script writes ``input/beam0.json``. Its relative output directory resolves to ``input/output``; each run creates a run directory beneath it. The run directory contains a CSV file and an HDF5 file with 64 statistics rows (turns 0–63). Log output gives the exact run path. Read the newest statistics file after the run:

.. code-block:: python

   from pathlib import Path

   from PASS.utils.table_io import read_table

   files = list(Path("input/output").rglob("*stat*.h5"))
   latest = max(files, key=lambda path: path.stat().st_mtime)
   data = read_table(latest)
   print(latest)
   print(data[["turn", "sigmaX", "sigmaY", "xEmittance", "yEmittance"]].tail())

For this uncoupled linear model, transverse RMS emittances should remain constant up to numerical rounding. The finite sampled initial values need not equal the requested emittances exactly. Longitudinal transfer is disabled here; this example does not model synchrotron oscillation or collective effects.

JSON File Structure
-------------------

The abbreviated structure below shows the nesting; empty bunch objects and omitted parameters are placeholders, not a runnable input.

.. code-block:: json

   {
       "Beam Name": "proton",
       "Number of Protons": 1,
       "Number of Neutrons": 0,
       "Number of Charges": 1,
       "Transition Gamma": 4.8,
       "Circumference (m)": 251.327,
       "Number of turns": 64,
       "Backend (gpu/cpu)": "cpu",
       "Number of GPU devices": 1,
       "Device Id": [0],
       "Output directory": "./output",
       "Is plot figure": false,
       "Is beam-beam": false,
       "Sequence": {
           "injection": {
               "S (m)": 0.0,
               "Command": "Injection",
               "Harmonic Number": 1,
               "Random Seed": 2026,
               "bunch0": {}
           },
           "twiss_0000": {
               "S (m)": 0.0,
               "Command": "Twiss",
               "S previous (m)": 0.0,
               "Beta x (m)": 8.333
           },
           "stat1": {
               "S (m)": 0.0,
               "Command": "StatMonitor"
           }
       }
   }

.. note::

    JSON uses the documented field names. The schema classes map Python field names to JSON keys through pydantic aliases.

    When reading, the engine first calls ``convert_keys_to_lower()`` to convert all keys to lowercase, so the case of JSON keys does not affect reading.


Core Components
---------------

MainConfig (Global Parameters)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 24 12 13 33

   * - Python field
     - JSON key
     - Type
     - Default
     - Description
   * - ``beam_name``
     - ``Beam Name``
     - ``str``
     - ``'proton'``
     - Beam species label.
   * - ``num_proton``
     - ``Number of Protons``
     - ``int``
     - ``1``
     - Number of protons per particle; zero for electrons and positrons.
   * - ``num_neutron``
     - ``Number of Neutrons``
     - ``int``
     - ``0``
     - Number of neutrons per particle.
   * - ``num_electron``
     - ``Number of Charges``
     - ``int``
     - ``1``
     - Signed charge number Z (q=Z e), not electron count; nonzero integer.
   * - ``reference_clock``
     - ``Reference clock``
     - ``ReferenceClock | None``
     - ``None``
     - Prescribed revolution-frequency program; see the reference-clock section.
   * - ``gamma_t``
     - ``Transition Gamma``
     - ``float``
     - ``7.635``
     - Transition gamma of the lattice.
   * - ``circumference``
     - ``Circumference (m)``
     - ``float``
     - ``569.1``
     - Positive ring circumference (m).
   * - ``num_turns``
     - ``Number of turns``
     - ``int``
     - ``100``
     - Number of simulated turns; positive integer.
   * - ``backend``
     - ``Backend (gpu/cpu)``
     - ``str``
     - ``'cpu'``
     - Compute backend: cpu or gpu.
   * - ``particle_precision``
     - ``Particle Precision``
     - ``str``
     - ``'float64'``
     - Coordinate storage: float32 or float64.
   * - ``num_gpu``
     - ``Number of GPU devices``
     - ``int``
     - ``1``
     - Number of GPU devices.
   * - ``gpu_id``
     - ``Device Id``
     - ``list[int]``
     - ``[0]``
     - List of GPU device identifiers.
   * - ``output_dir``
     - ``Output directory``
     - ``str``
     - ``'./output'``
     - Output directory; relative paths resolve against the input JSON directory.
   * - ``is_plot``
     - ``Is plot figure``
     - ``bool``
     - ``False``
     - Whether to generate plots after tracking.
   * - ``timing``
     - ``Timing``
     - ``TimingConfig``
     - ``TimingConfig()``
     - TimingConfig: mode=command, log_interval=10, warmup_turns=1, include_io=True.
   * - ``is_beambeam``
     - ``Is beam-beam``
     - ``bool``
     - ``False``
     - Reserved switch; keep False because beam-beam tracking is not implemented.


Space charge is configured by the separate top-level ``Space charge`` block,
not by ``MainConfig``. See :doc:`space_charge` for its named resource schema
and sequence-command references. Each resource selects ``Method`` (``pic``,
``frozen``, ``quasi-frozen``) and ``Solver``; the latter includes the boundary
condition in its name. The command's ``Aperture type/value`` defines particle
losses and, for Dirichlet solvers, the conducting wall.  Supply a complete grid full-width or half-width pair;
omitting the command aperture selects a rectangle equal to that grid.

.. _en-reference-clock:

Prescribed machine clock and initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optional top-level ``Reference clock`` defines a positive revolution
frequency :math:`f_{rev}(t)` and an epoch :math:`t_*`:

.. math::

   \Psi(t)=\int_{t_*}^{t} f_{rev}(u)\,du.

Use ``Revolution frequency (Hz)`` (scalar or list), ``Time (s)`` for list
samples, and ``Time origin (s)`` (default 0). Samples are linearly interpolated,
end values are held, and the integral is evaluated analytically on each
segment. This prescribed program is independent of tracked bunch energies.
Without a program, PASS fixes its frequency to the initial reference
velocity of harmonic-id-zero divided by circumference; it does not follow
subsequent acceleration automatically.

Initially, :math:`T_b=\Psi^{-1}(-h_{id}/h_{group})`, unless BunchConfig supplies
``Reference arrival time (s)``. Injection on turn n uses
:math:`\Psi^{-1}(n-h_{id}/h_{group})`; an explicit initial arrival time shifts
that source schedule by its difference from the nominal initial time.
Incoming z and normalized momenta are transformed to the destination bunch
reference without changing physical arrival times or momenta.

``harmonic_id`` and ``harmonic_number`` describe nominal grouping slots.
The nominal slot position ``harmonic_id*C/harmonic_number`` is calculated
when needed for metadata output. Adding it to z does not reconstruct a
physical position or arrival time. RF harmonics are
independent of this grouping count. :doc:`reorganize` explains regrouping by
the prescribed clock phase while retaining unwrapped particle times.

InjectionItem (Injection and Grouping)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``InjectionItem`` declares ``harmonic_number`` (JSON key ``Harmonic Number``) once at the injection level. This value is the bunch-grouping count and determines:

- The number of bunch centers around the ring, separated by :math:`C/h_{\mathrm{group}}`
- The required number of ``BunchConfig`` entries in ``bunches``
- The requirement that ``harmonic_id`` values uniquely cover :math:`0,\ldots,h_{\mathrm{group}}-1`

It does not constrain ``RFComponent.harmonic``. Represent an unfilled group with a declared bunch whose ``num_macro_particles`` is zero.

Set ``random_seed`` (JSON key ``Random Seed``) to an integer when the generated particle distribution must be reproducible. Leave it unset, or use JSON ``null``, for the default non-deterministic seed. The seed belongs to the whole Injection command, so its random stream is shared by all declared bunches and injection turns.


BunchConfig (Bunch Parameters)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 30 10 35

   * - Property
     - JSON key
     - Type
     - Description
   * - ``kinetic_energy``
     - ``Kinetic Energy per Nucleon (eV/u)``
     - float
     - Kinetic energy per nucleon (eV/u)
   * - ``num_real_particles``
     - ``Number of Real Particles``
     - int
     - Number of real particles per bunch
   * - ``num_macro_particles``
     - ``Number of Macro Particles``
     - int
     - Number of macro particles per bunch
   * - ``beta_x`` / ``beta_y``
     - ``Beta x (m)`` / ``Beta y (m)``
     - float
     - Twiss β function
   * - ``alpha_x`` / ``alpha_y``
     - ``Alpha x`` / ``Alpha y``
     - float
     - Twiss α function
   * - ``emit_x`` / ``emit_y``
     - ``Emittance x (m'rad)``
     - float
     - Emittance
   * - ``sigma_z``
     - ``Sigma z (m)``
     - float
     - Bunch length
   * - ``dp``
     - ``Sigma dp/p``
     - float
     - Momentum spread
   * - ``dist_trans``
     - ``Transverse dist``
     - str
     - Transverse distribution: ``kv`` / ``gaussian`` / ``uniform`` / ``waterbag`` / ``parabolic``
   * - ``dist_longi``
     - ``Longitudinal dist``
     - str
     - Longitudinal distribution: ``gaussian`` / ``coasting`` / ``matchz`` / ``matchdp``
   * - ``rf_voltage``
     - ``RF Voltage (V)``
     - float
     - RF voltage (used in matchz/matchdp modes)
   * - ``rf_phase``
     - ``RF Phase (rad)``
     - float
     - RF phase
   * - ``harmonic_id``
     - ``Harmonic ID of this bunch``
     - int
     - Bunch-group index; its nominal slot is :math:`z_{\mathrm{center}}=h_{\mathrm{id}}C/h_{\mathrm{group}}`
   * - ``rf_s_position``
     - ``RF S Position Refer to Inj. Point (m)``
     - float
     - RF-cavity position relative to injection, used to linearly back-propagate a matched distribution to :math:`s=0`
   * - ``momentum_offset_dp``
     - ``Momentum Offset dp``
     - float
     - Mean bunch relative-momentum offset; mutually exclusive with the kinetic-energy offset
   * - ``kinetic_energy_offset``
     - ``Kinetic Energy Offset (eV)``
     - float
     - Mean bunch kinetic-energy offset, converted exactly to a relative-momentum offset internally

All generated or manually inserted ``z`` values in ``BunchConfig`` are bunch-relative coordinates :math:`z_{\mathrm{rel}}`, not absolute laboratory azimuths.

Sequence (Sequence Container)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Sequence`` is an ordered container that stores all sequence items arranged by position ``s`` . Export sorts items by ``(s, command priority)``; ties preserve insertion order. Execution additionally groups nearby positions using its position tolerance.

.. code-block:: python

   from PASS.para.schema.sequence import Sequence
   from PASS.para.schema.bunch import BunchConfig, InjectionItem
   from PASS.para.schema.elements import QuadrupoleItem
   from PASS.para.schema.monitors import StatMonitorItem

   bunch = BunchConfig(kinetic_energy=45e6, num_real_particles=100000000000,
                       num_macro_particles=2048, emit_x=2e-6, emit_y=2e-6)
   seq = Sequence()
   seq.add("injection", InjectionItem(s=0.0, bunches=[bunch]))
   seq.add("qd1", QuadrupoleItem(s=1.0, k1l=0.2, length=0.5))
   seq.add("stat1", StatMonitorItem(s=0.0))

Supported sequence item types:

- ``InjectionItem`` — injection point (must have ``s=0`` )
- ``TwissItem`` — twiss transfer point
- ``DriftItem`` , ``QuadrupoleItem`` , ``SBendItem`` , etc. — physical elements
- ``StatMonitorItem`` , ``DistMonitorItem`` , ``PhaseAdvanceMonitorItem`` — monitors


Lattice Sources
---------------

PASS supports three methods for generating lattice sequences, which can be selected or combined as needed:

Method 1: Read from MADX twiss file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reads a twiss TFS file generated by MADX, converting each element into a ``TwissItem`` transfer point. Suitable for **element-by-element twiss transport** mode.

.. code-block:: python

   from PASS.para.madx import read_madx_twiss

   items, names, circum = read_madx_twiss(
       twiss_file="lattice.tfs",
       error_file="errors.tfs",       # optional
       muz=0.001,                      # longitudinal tune
       dqx=0.0,                        # chromaticity (or "from_file")
       dqy=0.0,
       is_field_error=False,           # whether to read field errors
       insert_patterns=["QD.*"],      # regex matching, inserted as thin lens elements
   )

For absolute field-error units, instance matching and the difference between distributed element errors and Twiss exit kicks, see :ref:`en-error`.

For a uniform base grid, use the resampling reader instead:

.. code-block:: python

   from PASS.para.madx import read_madx_twiss_interpolated

   items, names, circum = read_madx_twiss_interpolated(
       twiss_file="lattice.tfs",
       num_interp_slice=101,          # 100 segments, including both 0 and C
       dqx="from_file", dqy="from_file",
       longitudinal_transfer="off",
   )

This reader uses ``interp_kind="phase_hermite"`` (the sole supported method),
``num_interp_slice`` counts **base points**, not segments; it must be an integer
at least two. DQx/DQy default to ``"from_file"`` and Mu z defaults to zero.
The longitudinal phase is used only for ``longitudinal_transfer="matrix"``.

In each source interval of length :math:`h`, define :math:`t=(s-s_i)/h` and
interpolate :math:`p(t)=\mu(s)-\mu(s_i)` with a quintic Hermite polynomial. The
endpoint constraints, in phase cycles, are:

.. math::

   p_t = \frac{h}{2\pi\beta}, \qquad
   p_{tt} = \frac{h^2\alpha}{\pi\beta^2}.

The interpolated optical functions are:

.. math::

   \beta(s)=\frac{h}{2\pi p_t},\qquad
   \alpha(s)=\frac{p_{tt}}{4\pi p_t^2}.

This preserves :math:`\beta'=-2\alpha` and :math:`\mu'=1/(2\pi\beta)` within
each interval, while matching source beta, alpha and cumulative phase at its
endpoints. The full phase derivative is checked at its endpoints and all
interior extrema to exclude nonpositive beta. DX/DPX use paired cubic Hermite
interpolation in the existing uncoupled, on-reference paraxial convention;
their TFS normalization is retained. This interpolation assumes uncoupled optics and does not transform closed-orbit coordinates. Source precision and spacing limit accuracy.

The table must include S=0 and S=LENGTH, with finite optics, positive beta and
unwrapped phases. Its full phase spans must agree with Q1/Q2 within TFS output
rounding (relative tolerance :math:`2\times10^{-8}`, absolute tolerance
:math:`2\times10^{-9}`). Output phases retain the source values rather than
being rescaled to rounded headers. Segment chromaticities are distributed in
proportion to the source phase span, preserving the requested DQx/DQy sums.

Original rows are not retained as additional output points. Matched thin
elements, field errors and repeated-S optical jumps add required split
positions. Error names are matched against the original table before
resampling. At optical jumps, a zero-length Twiss map joins the incoming and
outgoing states. Additional kicks/errors execute after the Twiss maps at that
position, consistent with command priority. Explicitly inserting design
focusing already represented by the source optics adds it again; error
insertion does not preserve the perturbed machine's tune by construction.
See :doc:`gui` for the corresponding import controls.

Method 2: Read from MADX twiss file as elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reads the twiss file, but converts each element into its corresponding physical element object ( ``QuadrupoleItem`` , ``SBendItem`` , etc.). Suitable for **element-by-element tracking** mode.

.. code-block:: python

   from PASS.para.madx import read_madx_elements

   items, names, circum = read_madx_elements(
       twiss_file="lattice.tfs",
       is_merge_drift=True,            # merge adjacent drift sections
       is_field_error=True,
       is_alignment_error=True,
       error_file="errors.tfs",
   )

The two error switches are independent and default to ``False``. Alignment
imports ``DX``, ``DY``, ``DPSI`` from the error TFS, moving only magnetic
fields while apertures and SC boundaries stay fixed. Nonzero unsupported
alignment components raise an error. Twiss transfer and resampled Twiss
imports reject alignment; use this element mode. See :ref:`en-error` for
normalization, instance matching and tracking order. ``generate_from_tfs``
also accepts ``is_alignment_error``.

Method 3: Smooth approximation twiss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No MADX file required; uses analytical formulas to generate twiss points with constant β function. :math:`\beta = C / (2\pi Q)` . Suitable for quick testing.

.. code-block:: python

   from PASS.para.smooth import generate_smooth_twiss

   items, names, circum = generate_smooth_twiss(
       circumference=569.1,
       qx=9.47, qy=9.43,
       num_points=100,
       longitudinal_transfer="off",
   )

Mixed Mode
~~~~~~~~~~

Twiss transfer points and physical elements can be mixed within the same sequence. For example, inserting an RF cavity into a twiss sequence:

.. code-block:: python

   from PASS.para.schema.elements import RFCavityItem

   from PASS.para.schema.sequence import Sequence
   from PASS.para.schema.bunch import BunchConfig, InjectionItem
   from PASS.para.schema.elements import QuadrupoleItem
   from PASS.para.schema.monitors import StatMonitorItem

   bunch = BunchConfig(kinetic_energy=45e6, num_real_particles=100000000000,
                       num_macro_particles=2048, emit_x=2e-6, emit_y=2e-6)
   seq = Sequence()
   seq.add("injection", InjectionItem(s=0.0, bunches=[bunch]))

   from PASS.para.smooth import generate_smooth_twiss
   twiss_items, twiss_names, circumference = generate_smooth_twiss(251.327, 4.8, 4.4, 17)

   # Twiss transfer points
   for i, item in enumerate(twiss_items):
       seq.add(f"twiss_{i:04d}", item)

   # insert RF cavity (at s=0)
   seq.add("rf1", RFCavityItem(s=0.0, components=[dict(voltage=100e3, harmonic=1, phase=0.5236)]))


External Data File Conversion
-----------------------------

PASS uses the **TFS format** as the unified format for all ramping/RF/exciter data files. ``tools/data_converter.py`` provides a general conversion pipeline that transforms various external files (CSV/TXT/TFS) into PASS TFS.

RF files retain physical seconds and do not use the magnet-ramping turn conversion below. RF columns are ``TIME, VOLTAGE, FREQUENCY, PHASE``; see :doc:`element/rfcavity`.

The magnet-ramping conversion helpers below prepare tables only. The current tracking engine rejects enabled magnetic-element ramping; creating a table does not enable that feature. RF physical-time tables are supported by RFCavity.

Four-Step Pipeline
~~~~~~~~~~~~~~~~~~

.. code-block:: text

   External file → load_raw_data → time_to_turn → interpolate → write_tfs

1. **load_raw_data** : Reads the external file, auto-detects turn/time columns
2. **time_to_turn** : If the external file provides time instead of turns, converts using the revolution frequency
3. **interpolate_to_continuous_turns** : Automatically interpolates when turns are non-contiguous
4. **write_tfs_ramping** : Writes to the PASS unified TFS format

One-Step Conversion
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from PASS.para.tools.data_converter import convert_external_to_tfs

   convert_external_to_tfs(
       input_path="external_ramp.csv",     # external file
       output_path="k1l_ramping.tfs",      # PASS TFS
       data_cols=["k1l", "k1sl"],          # data column names
       revolution_freq=1.76e6,             # revolution frequency (Hz)
       num_turns=5000,                     # target number of turns
       method="linear",                    # interpolation method
   )

Pre-packaged Wrappers
~~~~~~~~~~~~~~~~~~~~~

Thin wrappers for common element types:

.. code-block:: python

   from PASS.para.tools.ramping import convert_k1l_ramping, convert_k2l_ramping
   from PASS.para.tools.rf_data import convert_rf_data

   # Quadrupole ramping
   convert_k1l_ramping("external.csv", "k1l_ramping.tfs", revolution_freq=1.76e6)

   # RF data
   convert_rf_data("llrf.csv", "rf_physical_time.tfs")

Step-by-Step Invocation
~~~~~~~~~~~~~~~~~~~~~~~

When the external file format is non-standard, each function can be called step by step:

.. code-block:: python

   from PASS.para.tools.data_converter import (
       interpolate_to_continuous_turns, write_tfs_ramping,
   )
   import numpy as np

   # Prepare data manually
   turn_arr = np.array([1, 50, 100, 500, 1000])
   k2l = np.array([0.0, 0.5, 1.0, 2.5, 4.4])

   turn_cont, data_cont = interpolate_to_continuous_turns(
       turn_arr, {"K2L": k2l},
       start_turn=1, end_turn=1000, method="linear",
   )
   write_tfs_ramping("k2l_ramping.tfs", turn_cont, None, data_cont)


Parameter scans and validation
------------------------------------------------------------

``model_copy(update=...)`` does not validate updated values. Rebuild a schema object from a field dictionary, then validate the complete generated input before tracking:

.. code-block:: python

   from PASS.para.schema.main import MainConfig

   baseline = MainConfig(circumference=251.327)
   candidate = {**baseline.model_dump(), "num_turns": 128}
   scan_config = MainConfig.model_validate(candidate)

The Python field ``num_electron`` is a historical name for the signed charge number, not a bound-electron count. ``num_proton=1, num_neutron=0, num_electron=1`` therefore specifies a proton. Python field names belong in schema constructors; JSON inputs must use the aliases shown in the tables.

Architecture Overview
---------------------

The parameter modules construct, validate and serialize the input objects:

.. code-block:: text

   PASS/para/
   ├── schema/       Parameter definitions (field definitions and aliases)
   │   ├── main.py         MainConfig: global simulation parameters
   │   ├── bunch.py        BunchConfig + OffsetConfig + InjectionItem
   │   ├── twiss.py        TwissItem: twiss transfer point
   │   ├── elements.py     Element configuration classes
   │   ├── monitors.py     StatMonitor / DistMonitor / PhaseAdvanceMonitor
   │   ├── space_charge.py SpaceChargeConfig + SpaceChargeResourceConfig + SpaceCharge
   │   └── sequence.py     Sequence: ordered container + auto-sorting
   ├── madx.py        MADX TFS → schema objects (element / twiss / error)
   ├── smooth.py      Analytical smooth approximation twiss
   ├── tools/        External data → PASS TFS
   │   ├── data_converter.py General data conversion pipeline
   │   ├── ramping.py         Element ramping file generation
   │   ├── rf_data.py         RF data file generation
   │   └── exciter_data.py    Exciter data file generation
   ├── toolkit.py    sort_sequence + class_map + apply_element_settings + build_sequence
   └── api.py        High-level API (generate_input / load_input / generate_from_tfs)

The data flow is as follows:

.. code-block:: text

   MADX TFS / user parameters / external data files
              │
              ▼
        madx.py / smooth.py + tools/  → schema objects / TFS files
              │
              ▼
         schema/ (pydantic)     ← field definitions and aliases: validation + aliases
              │
              ▼
        api.py (generate_input) → beam0.json
              │
              ▼
         PASS engine (Config → Beam → CommandSequence → Executor)


API entry points
--------------------------------

``PASS.para.api`` exports ``generate_input``, ``build_sequence``, ``generate_from_tfs`` and ``load_input``. ``load_input(path)`` returns ``(MainConfig, raw_sequence_dict)``; it does not reconstruct typed commands or return the top-level Space charge/Wake field blocks. Preserve those blocks explicitly when editing a complete existing file.
