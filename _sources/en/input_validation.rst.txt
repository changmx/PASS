JSON input validation
=====================

One-click preflight
-------------------

In **Configuration → Execution sequence**, click **Validate** to inspect the
complete current JSON, including all commands, named resources and input tables.
Pending property edits and source edits are applied first. Invalid JSON syntax
leaves the current document unchanged and highlights the source line/column.
The full check runs in a worker thread and does not generate particles, construct
Poisson matrices, start tracking, or create simulation output folders.

The report lists **errors** and **warnings**, the exact JSON path, a message,
and a stable rule code. Filter by severity, double-click to open the corresponding
command/bunch/resource, or copy/export the report. Exported reports are ordinary
UTF-8 JSON with ``valid``, ``errors``, ``warnings``, ``commands``,
``checked_files`` and ``diagnostics``. Diagnostic pointers use JSON Pointer
escaping, so punctuation in command names does not break navigation.

**Errors block execution.** Warnings describe valid but potentially unintended
behavior, such as clipped output windows, unused files/resources, RF tables that
hold their final row, or large monitor buffers. A warning does not request another
approval or prevent running. Full validation also inspects disabled configuration
contents; malformed declared parameters are errors, while unavailable or malformed
files belonging to inactive features are warnings.

The compact status during editing is a **parameter precheck**. It intentionally
does not read file contents. After edits, a previous full result is replaced by
this precheck status. Click Validate for a fresh full report. Execution repeats
full validation of every selected input; for two beams, shared turn count,
backend, precision, GPU settings and timing settings must agree. Direct
``PASS.main.main`` calls use the same preflight before initialization.

What is checked
---------------

* JSON syntax, object root, duplicate keys (including case collisions), finite
  numbers, strict numeric/boolean/list types, unknown fields and missing fields
  required by the engine. Use the current exported schema aliases and Command
  spellings; old names, Python attribute names and coercible strings are rejected.
* Global particle identity, nonzero charge, positive circumference/transition
  gamma, turn count, backend, precision, timing and device-ID constraints.
  Explicit reference clocks require positive frequency and subluminal design
  speed (revolution frequency times circumference); nested errors identify the
  clock/RF table or WakeField group/component field.
* Injection at ``Sequence.injection`` and :math:`S=0`, continuous bunch numbering,
  bunch-group count and unique harmonic IDs, positive kinetic energy, intensities,
  emittances, Twiss functions, supported distributions and mutually exclusive
  momentum/energy offsets. All nonempty bunches in one beam must have equal
  ``Number of Real Particles / Number of Macro Particles`` ratios. Empty bunches
  inherit this fixed weight. RF harmonics remain
  independent of the bunch-grouping harmonic.
* Manual particle row shape and momentum domain. With injection window ``T`` and
  interval ``I``, the event count is :math:`M=\lceil T/I\rceil`; the first block has
  :math:`\lfloor N/M\rfloor + N\bmod M` particles. Manual coordinates replace
  particles in that block. The check reports incomplete injection schedules.
* Every registered element, sorting/reorganization command and monitor: position,
  body length, thin-command constraints, integrators/models, multipole arrays,
  aperture dimensions and polygon intersections, RF acceptance, Exciter frequency
  alternatives and modulation denominators, positive optics beta functions,
  reorganization during unfinished injection, and monitor/output windows.
* Slicer models, explicit ranges, conflicting definitions of the same slice set,
  named space-charge references, and actual engine ordering (position tolerance,
  command priority and stable insertion order). A Slicer must execute before SC
  and after an intervening SortBunch/ReorganizeBunch invalidates its result.
* Explicit and internal SC: method/solver/profile compatibility, paired grid
  extents, positive grid dimensions, aperture containment, full-rectangle DST
  boundaries, active grid nodes for Dirichlet solvers, unsupported analytic
  potential output, CPU-only execution and internal thick-element requirements.
  Coverage uses the same periodic interval analysis as tracking and respects
  ``Coverage check`` and ``Coverage mode``.
* Active distribution tables (HDF5 or TFS), RF and offset TFS files: existence, parsing, required
  columns, numeric types, finite data, distribution row count and momentum domain,
  integer positive RF harmonics, monotonic offset times and integer turn indices.
  Shared files are read once per input check. Relative paths resolve against the
  containing JSON directory, both during validation and execution; UTF-8 BOM is
  accepted for input JSON.

The current engine does not implement magnetic-element ramping or a BeamBeam
command. Enabling these features is an error instead of silently ignoring the
request. RFCavity's implemented RF table remains supported. ElSeparator requires
exactly one finite ``V (V)`` or ``VL (V m)``, a positive gap and a finite septum
position. Zero strength is valid; nonzero V requires positive length, while VL
also supports a zero-length kick. The old ``Voltage (V)``, electrode-height/center,
mode and separate EX/EY/EXL/EYL parameters are rejected. Missing geometry is not inferred.

Static validation cannot establish long-term beam stability or certify future
particle-dependent quantities. For example, particles may subsequently leave a
PIC grid, a quasi-frozen beam may become degenerate, or RF tracking may change
the reference energy. Runtime guards remain in place. GPU driver availability,
actual free memory, filesystem permissions and numerical convergence still depend
on the execution environment. Large-buffer warnings are estimates, not resource
reservations. Matched-distribution checks establish necessary RF stability
conditions; sampling/convergence still has to be verified by running the model.

Command line and Python
-----------------------

No Qt installation or open GUI is needed:

.. code-block:: console

   python -m PASS.validation beam.json
   python -m PASS.validation beam0.json beam1.json --report validation-report.json

The exit status is ``0`` when there are no errors, including when warnings are
present, and ``1`` when validation fails. ``--report`` writes only the requested
report file and does not create simulation outputs.

.. code-block:: python

   from PASS.validation import validate_file, validate_input

   report = validate_file("beam.json")
   for issue in report.diagnostics:
       print(issue.severity, issue.code, issue.pointer, issue.message)
   if not report.ok:
       raise ValueError(report.text())

   # Parameter-only precheck for an in-memory editor document:
   report = validate_input(data, base_dir="inputs", check_files=False)

Validation does not mutate the supplied dictionary or rewrite the input file.
