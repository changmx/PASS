Input File Generation (Command-Line Mode)
==========================================

Introduction
------------

PASS uses **JSON files** as simulation input. The engine ( ``Config`` , ``Beam`` , ``CommandSequence`` ) reads all parameters from the JSON file, including particle species, bunch distribution, lattice sequence, monitors, etc.

The parameter system ``PASS/para/`` provides a set of schema definitions based on **pydantic v2** . Users assemble parameter objects through Python scripts and call ``generate_input()`` to output an engine-compatible JSON file. Compared to hand-writing JSON, this approach offers the following advantages:

- **Type safety** : Parameter types and ranges are declared in the schema; invalid values are intercepted at generation time;
- **Alias mapping** : Python code uses concise property names (e.g., ``circumference`` ), while the JSON output automatically uses the keys expected by the engine (e.g., ``"Circumference (m)"`` );
- **Reusability** : Schema objects can be quickly derived via ``model_copy(update={...})`` , suitable for parameter scans;
- **GUI extensibility** : The schema includes JSON Schema export; future GUIs can auto-render forms.

.. note::

    This document introduces input file generation in command-line mode. GUI mode will be provided in future releases.

Architecture Overview
---------------------

The parameter system is divided into five layers, each with clear responsibilities and no inter-dependencies:

.. code-block:: text

   PASS/para/
   ├── schema/       Parameter definitions (single source of truth)
   │   ├── main.py         MainConfig: global simulation parameters
   │   ├── bunch.py        BunchConfig + OffsetConfig + InjectionItem
   │   ├── twiss.py        TwissPoint: twiss transfer point
   │   ├── elements.py     12 element types (Drift→RFCavity)
   │   ├── monitors.py     StatMonitor / DistMonitor / PhaseMonitor
   │   ├── space_charge.py SpaceChargeConfig
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
         schema/ (pydantic)     ← single source of truth: validation + aliases
              │
              ▼
        api.py (generate_input) → beam0.json
              │
              ▼
         PASS engine (Config → Beam → CommandSequence → Executor)


Quick Start
-----------

Minimal Example
~~~~~~~~~~~~~~~

The following script generates a complete input file containing injection + smooth approximation twiss + statistical monitor:

.. code-block:: python

   from PASS.para.api import generate_input
   from PASS.para.schema.main import MainConfig
   from PASS.para.schema.bunch import BunchConfig, InjectionItem
   from PASS.para.schema.sequence import Sequence
   from PASS.para.schema.monitors import StatMonitor
   from PASS.para.smooth import generate_smooth_twiss

   # 1. Global parameters
   main = MainConfig(
       beam_name="proton",
       num_proton=1, num_neutron=0, num_electron=1,
       gamma_t=4.8, circumference=251.327,
       num_turns=1000, backend="cpu",
   )

   # 2. Bunch
   bunch = BunchConfig(
       kinetic_energy=45e6,
       num_real_particles=int(1e11),
       num_macro_particles=int(1e5),
       beta_x=0.5, beta_y=0.5,
       alpha_x=-2.61, alpha_y=1.57,
       emit_x=200e-6, emit_y=100e-6,
       sigma_z=30, dp=0.005,
       dist_trans="gaussian", dist_longi="matchz",
       rf_voltage=100e3, rf_phase=0.5236,
   )

   # 3. Lattice sequence
   items, circum = generate_smooth_twiss(
       circumference=main.circumference,
       qx=4.8, qy=4.4, num_points=100,
   )
   main.circumference = circum

   seq = Sequence()
   seq.add("injection", InjectionItem(s=0.0, bunches=[bunch]))
   for i, item in enumerate(items):
       seq.add(f"twiss_{i:04d}", item)
   seq.add("stat1", StatMonitor(s=0.0))

   # 4. Generate JSON
   generate_input(main, seq, "beam0.json")

How to run:

.. code-block:: console

   cd C:\Users\changmx\Documents\PASS
   python input/generate_beam0.py

Output file: ``input/beam0.json``

JSON File Structure
-------------------

The generated JSON file has the following structure:

.. code-block:: json

   {
       "Beam Name": "proton",
       "Number of Protons": 1,
       "Number of Neutrons": 0,
       "Number of Charges": 1,
       "Transition Gamma": 4.8,
       "Circumference (m)": 251.327,
       "Number of turns": 1000,
       "Backend (gpu/cpu)": "cpu",
       "Number of GPU devices": 1,
       "Device Id": [0],
       "Output directory": "./output",
       "Is plot figure": true,
       "Is space charge": false,
       "Is beam-beam": false,
       "Sequence": {
           "injection": {
               "S (m)": 0.0,
               "Command": "Injection",
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

    The JSON key names are a hard contract of the engine. The schema layer automatically handles the mapping from Python property names to JSON keys through pydantic's ``alias`` mechanism; users do not need to write them manually.

    When reading, the engine first calls ``convert_keys_to_lower()`` to convert all keys to lowercase, so the case of JSON keys does not affect reading.


Core Components
---------------

MainConfig (Global Parameters)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 10 40

   * - Property
     - JSON key
     - Type
     - Description
   * - ``beam_name``
     - ``Beam Name``
     - str
     - Beam label
   * - ``num_proton``
     - ``Number of Protons``
     - int
     - Number of protons per particle (0 for electron/positron)
   * - ``num_neutron``
     - ``Number of Neutrons``
     - int
     - Number of neutrons per particle (>0 for ions)
   * - ``num_electron``
     - ``Number of Charges``
     - int
     - Number of charges per particle (can be negative, cannot be 0)
   * - ``gamma_t``
     - ``Transition Gamma``
     - float
     - Transition gamma
   * - ``circumference``
     - ``Circumference (m)``
     - float
     - Ring circumference (m)
   * - ``num_turns``
     - ``Number of turns``
     - int
     - Number of simulation turns
   * - ``backend``
     - ``Backend (gpu/cpu)``
     - str
     - Compute backend: ``cpu`` or ``gpu``
   * - ``num_gpu``
     - ``Number of GPU devices``
     - int
     - Number of GPUs
   * - ``gpu_id``
     - ``Device Id``
     - list[int]
     - GPU device ID list
   * - ``output_dir``
     - ``Output directory``
     - str
     - Output directory
   * - ``is_plot``
     - ``Is plot figure``
     - bool
     - Whether to generate plots
   * - ``is_space_charge``
     - ``Is space charge``
     - bool
     - Whether to enable space charge
   * - ``is_beambeam``
     - ``Is beam-beam``
     - bool
     - Whether to enable beam-beam interaction

InjectionItem (Injection and Grouping)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``InjectionItem`` declares ``harmonic_number`` (JSON key ``Harmonic Number``) once at the injection level. This value is the bunch-grouping count and determines:

- The number of bunch centers around the ring, separated by :math:`C/h_{\mathrm{group}}`
- The required number of ``BunchConfig`` entries in ``bunches``
- The requirement that ``harmonic_id`` values uniquely cover :math:`0,\ldots,h_{\mathrm{group}}-1`

It does not constrain ``RFCavityElement.harmonic``. Represent an unfilled group with a declared bunch whose ``num_macro_particles`` is zero.

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
     - Bunch-group index; its center is :math:`z_{\mathrm{center}}=h_{\mathrm{id}}C/h_{\mathrm{group}}`
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

``Sequence`` is an ordered container that stores all sequence items arranged by position ``s`` . The order of insertion does not affect the final result — items are automatically sorted by ``(s, command priority)`` upon export.

.. code-block:: python

   seq = Sequence()
   seq.add("injection", InjectionItem(s=0.0, bunches=[bunch]))
   seq.add("qd1", QuadrupoleElement(s=1.0, k1l=0.2, length=0.5))
   seq.add("stat1", StatMonitor(s=0.0))

Supported sequence item types:

- ``InjectionItem`` — injection point (must have ``s=0`` )
- ``TwissPoint`` — twiss transfer point
- ``DriftElement`` , ``QuadrupoleElement`` , ``SBendElement`` , etc. — physical elements
- ``StatMonitor`` , ``DistMonitor`` , ``PhaseMonitor`` — monitors


Lattice Sources
---------------

PASS supports three methods for generating lattice sequences, which can be selected or combined as needed:

Method 1: Read from MADX twiss file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reads a twiss TFS file generated by MADX, converting each element into a ``TwissPoint`` transfer point. Suitable for **element-by-element twiss transport** mode.

.. code-block:: python

   from PASS.para.madx import read_madx_twiss

   items, circum = read_madx_twiss(
       twiss_file="lattice.tfs",
       error_file="errors.tfs",       # optional
       muz=0.001,                      # longitudinal tune
       dqx=0.0,                        # chromaticity (or "from_file")
       dqy=0.0,
       is_field_error=False,           # whether to read field errors
       insert_patterns=["QD.*"],      # regex matching, inserted as thin lens elements
   )

Method 2: Read from MADX twiss file as elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reads the twiss file, but converts each element into its corresponding physical element object ( ``QuadrupoleElement`` , ``SBendElement`` , etc.). Suitable for **element-by-element tracking** mode.

.. code-block:: python

   from PASS.para.madx import read_madx_elements

   items, names, circum = read_madx_elements(
       twiss_file="lattice.tfs",
       is_merge_drift=True,            # merge adjacent drift sections
       is_field_error=True,
       error_file="errors.tfs",
   )

Method 3: Smooth approximation twiss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No MADX file required; uses analytical formulas to generate twiss points with constant β function. :math:`\beta = C / (2\pi Q)` . Suitable for quick testing.

.. code-block:: python

   from PASS.para.smooth import generate_smooth_twiss

   items, circum = generate_smooth_twiss(
       circumference=569.1,
       qx=9.47, qy=9.43,
       num_points=100,
       muz=0.001,
   )

Mixed Mode
~~~~~~~~~~

Twiss transfer points and physical elements can be mixed within the same sequence. For example, inserting an RF cavity into a twiss sequence:

.. code-block:: python

   from PASS.para.schema.elements import RFCavityElement

   seq = Sequence()
   seq.add("injection", InjectionItem(s=0.0, bunches=[bunch]))

   # twiss transfer points
   for i, item in enumerate(twiss_items):
       seq.add(f"twiss_{i:04d}", item)

   # insert RF cavity (at s=0)
   seq.add("rf1", RFCavityElement(s=0.0, voltage=100e3, harmonic=1, phase=0.5236))


External Data File Conversion
-----------------------------

PASS uses the **TFS format** as the unified format for all ramping/RF/exciter data files. ``tools/data_converter.py`` provides a general conversion pipeline that transforms various external files (CSV/TXT/TFS) into PASS TFS.

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
   convert_rf_data("llrf.csv", "rf_data.tfs", revolution_freq=1.76e6)

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


API Reference
-------------

.. code-block:: python

   from PASS.para.api import generate_input, load_input

   # Generate JSON
   generate_input(
       main: MainConfig,
       sequence: Sequence,
       output_path: str,
       space_charge: SpaceChargeConfig | None = None,
       extra_modules: dict | None = None,
   ) -> str

   # Load existing JSON (for modification and regeneration)
   main, seq_dict = load_input("beam0.json")

Complete Example
----------------

The built-in example script is located at ``input/generate_beam0.py`` and can be run directly:

.. code-block:: console

   cd C:\Users\changmx\Documents\PASS
   python input/generate_beam0.py

This script demonstrates the complete end-to-end workflow: global parameters → multi-bunch configuration → smooth approximation twiss → lattice sequence assembly → JSON output. The generated ``beam0.json`` can be directly read and executed by the PASS engine.
