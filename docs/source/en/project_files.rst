Single-file projects
====================

PASS supports ordinary standalone JSON inputs and single-file ``.passproj``
projects through the GUI and command line. The GUI **File** menu separates JSON,
project, and export actions.

Saving and opening
------------------

In standalone mode, **Save JSON** and **JSON Save as** write an ordinary PASS input.
In project mode, **Save project** writes all input JSON files, input dependencies,
sources, and generation settings into one container. ``Ctrl+S`` follows the active
mode. **Open JSON** opens an independent document; it does not silently import
into a project. Use **Import JSON into project** for that operation. Replacing a
document or closing the window offers to save unsaved changes.

File operations show progress and temporarily disable editing. Cancellation may
wait for an in-progress read or write to finish; a completed save is reported as
successful. Saving JSON in another directory updates relative input references,
including those used by undo/redo, so they still identify the same source files.

**Create project from current input** packages the current configuration. Import
additional JSON files and use the input selector to switch between them. Projects
copy referenced particle distributions, offset tables, RF component programs,
Bump waveforms, WakeField model files, and magnet ramping files. Original imported JSON and MAD-X TFS files and generator settings
are retained as sources. Missing inputs identify their JSON location and prevent
an incomplete project from being saved. Additional source files can be added from
**Project contents**.

Inspecting and reusing parameters
---------------------------------

**Project contents** lists input JSON, source files, and generation settings.
Select a JSON or command to view its parameters and raw text. Copy a value or the
selected command's JSON to the clipboard. **Copy command into current input**
also brings its named space-charge configurations, slicers, and file dependencies.
WakeField copies its referenced Slicer, including conflict-safe slice-set renaming.
The design clock is derived from the target input, rather than copied as a
separate resource. Importing an RFCavity can therefore change the target's
ideal RF-only trajectory and other commands using that clock. Explicit RF
frequency tables retain their physical frequency; harmonic RF follows the
target's derived clock. This is not a transfer of tracked beam or wake history.
Existing names are preserved; conflicting imported names receive numeric suffixes.
Another project can be opened read-only as a parameter source.

Copying a command adds one undo step, including imported dependencies.
Cancelled or failed copies leave the active input unchanged.

TFS and CSV files have a table preview of up to 500 rows. Text previews are bounded
to 2 MiB and parameter tables to 5,000 entries; exact files can always be exported.
Binary files display metadata and offer original-file export. Dependencies copied
into a standalone JSON are stored beside it in ``<json-name>_files`` when saved.

Generation settings describe the source of generated commands. **Load generation
settings** restores a form for preview and insertion; it does not automatically
replace manually edited commands. Plain JSON files contain the generated commands
but do not retain this additional generation metadata.

Running
-------

Select the Beam 0 input and optionally a distinct Beam 1 input on the run page.
A project can store more candidate configurations than the engine runs at once.
Relative output directories are resolved beside the JSON or saved project; an
unsaved project uses the current working directory.

After installing PASS, a saved project can also be run without Qt:

.. code-block:: console

   pass-run example.passproj
   pass-run --passproj example.passproj
   python -m PASS example.passproj

These commands are equivalent. The ``.passproj`` extension selects project
loading, including archive and checksum validation. The command line uses
``run_settings.beam0`` and the optional ``run_settings.beam1`` saved by the GUI.
When no Beam 0 selection has been saved, ``active_config_id`` supplies Beam 0;
Beam 1 is omitted unless selected. A saved selection that no longer exists, or
selecting the same configuration for both beams, is an error. Other configurations
are not run automatically. A project file cannot be combined with a second input
argument. Do not mix positional inputs with named input options. ``--passproj``
cannot be combined with ``--beam0`` or ``--beam1``; those options select standalone
JSON inputs instead. See :doc:`input_generation` for the JSON command forms.

Without an override, command-line project runs use ``run_settings.output_directory``, defaulting to
``output``. Relative paths are resolved beside the saved project, regardless of
the shell's working directory or output directories inside the selected JSON
configurations. Absolute output paths remain absolute. Results and input snapshots
use the layout below and stay outside the temporary project cache. The project
file is unchanged by execution.

To choose a different output root for this run, use ``--output DIR``:

.. code-block:: console

   pass-run example.passproj --output results
   python -m PASS example.passproj --output "results/project run"

This option takes precedence over the saved output setting and does not modify
the project. Relative override paths use the working directory at launch;
snapshots and results both use the chosen root with the same layout below.
``pass-run --help`` or ``pass-run -h`` displays the input modes, options, path
rules, and examples, including ``--stop-file``;
exit codes are documented in :doc:`input_generation`.

The GUI freezes the selected configurations, then validates them, copies their
dependencies using the same snapshot service as ``PASS.main.main``, and verifies
the snapshot in a background worker before starting the child process.
Preparation reports its current stage and can be cancelled;
duplicate starts are disabled during preparation and execution. Failed or
cancelled preparation leaves the previously displayed run intact. Individual
validation calls finish before observing cancellation.

Each GUI run and command-line project run has its own result directory, with
the input snapshot in its ``input`` subdirectory. This is the same dated layout
as standalone JSON runs:

.. code-block:: text

   <output>/<YYYY_MMDD>/<HHMM_SS>/       # simulation results for this run
       input/
           beam0.json                  # fixed runtime input; optional beam1.json
           configuration0.json         # original configuration for comparison
           assets/...                  # copied dependency bytes
           run.json                    # run record
           gui.log                     # complete process log for GUI runs

Runtime JSON uses relative paths to copied ``assets/...`` files and an absolute
path to the output root. Preparation allocates the dated result directory before
copying inputs and stores its path in ``run.json`` as ``results_directory``.
Initialization uses that directory without adding another date/time level.
A suffix is added to the time directory when needed to avoid reusing an existing
result directory. The run ID remains in ``run.json`` rather than forming another
directory level.
Tracking reads these copied inputs
without creating a second snapshot. Editing the project later does not
change the running task, and a new run does not share the preceding run's output
directory. Results and their input snapshots remain outside the temporary
project cache.

Running standalone JSON through the command-line/Python entry point uses the
same snapshot format and result layout; see
:doc:`input_generation`. Missing dependencies
of disabled resources are recorded in ``unavailable_dependencies`` with the
existing validation warnings. Missing required active inputs prevent execution.

**Stop** requests a cooperative stop at a turn boundary: the current turn finishes,
command finalizers write available output, and automatic post-run plotting is
skipped. The run is recorded as ``stopped``, with exit code 3. A long initialization
or turn must finish before this request takes effect. **Force stop** kills the
child process and is recorded separately; unwritten buffers and unfinished output
files may be lost. Closing the GUI requests cancellation or normal stopping and
waits without blocking the event loop; force stopping remains available.
An interruption such as ``KeyboardInterrupt`` is recorded as ``interrupted``
with exit code 130; it does not guarantee that the current turn finished.
Initialization, tracking and output exceptions produce ``failed`` with exit
code 1. An output failure during interruption is also a failure, not a clean
interruption. Successful completion uses exit code 0. The supervised runner
requests exception propagation instead of inferring success from log messages.
Existing history entries retain their stored status and exit code.

The log offers follow mode, search, and severity filtering. Scrolling away or
searching disables following. The visible log is bounded to 20,000 lines, while
``gui.log`` and log export retain the full unfiltered process stream. The run page
can open the result directory directly.

Run history and reproduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Run history** lists the latest 100 registered run records. It can open snapshots
or results, compare two original configurations, and rerun a selected snapshot.
Rerunning verifies the SHA-256 hashes of runtime JSON, comparison configurations,
and dependencies, then creates a new snapshot and result directory. It uses the
recorded input bytes rather than the current editor state. Moving or modifying
snapshot files can make that record unavailable for rerunning.

``run.json`` records start/end times, status, exit code, result paths, input and
dependency hashes, and Injection seeds together with their input and command
names. It also records Python, package and PASS version information, backend and
particle precision. ``source`` identifies the actual source root and its Python
source hash; Git HEAD and the dirty state of the ``PASS`` source tree are included
when Git is available. This source identity supplements the PASS version string,
which can come from installed package metadata. ``configured_device_ids`` records
the requested devices; ``observed_gpu`` is populated only after actual GPU
selection and includes device name and runtime/driver versions. It remains null
for CPU runs or before GPU initialization.

A rerun records its predecessor in ``source_run`` and starts from the initial
conditions; it does not continue a tracking checkpoint. The snapshot preserves
inputs, but does not restore the previous Python environment or source checkout.
``Random Seed: null`` retains nondeterministic sampling; even with integer seeds,
identical results are not guaranteed across changed code, libraries, precision,
or hardware.

Applied parameter changes support undo/redo; active text fields and the JSON
editor keep their own text undo/redo. Pending property and JSON edits must be
resolved before saving. Default GUI Gaussian bunches use positive transverse
emittances of 1e-6 m rad; zero-extent Gaussian generation is rejected before launch.

Exporting
---------

**Export current JSON** writes a copy of the configuration only. It does not
include referenced files and does not change which document is being edited.
**Export runnable input bundle** writes the run page's selected inputs and their
dependencies as ZIP, together with ``run.py`` and an English usage note. Extract
the entire bundle, install PASS, and execute ``python run.py``. The launcher
resolves inputs from the extracted directory. The bundle is ordinary input data
and does not require Qt to run.

Restoring existing drafts
-------------------------

The GUI does not create periodic draft backups. Save changes explicitly using
the JSON or project save actions. Existing copies in the application's
``AppLocalDataLocation/recovery`` directory remain available; they are not deleted
when the GUI starts or closes.

Use **File → Recover unsaved drafts** to look for existing copies when needed.
The GUI reads the list in a cancellable background task when this action is used;
it does not scan for drafts at startup or retain a background backup writer.
Recovery opens an unsaved document or project without overwriting its original
file. Depending on the copy's contents, it restores unapplied property fields,
raw JSON (including invalid JSON), applied configurations, project generation and
run settings, and temporary dependency assets. Switching the active project input
does not resolve the source recovery record. That record remains pending until
the recovered document is saved or explicitly discarded. Normal closing marks
handled records as resolved without deleting the recovery files.

Container format
----------------

Version 2 is a standard ZIP/ZIP64 container with UTF-8 JSON metadata:

.. code-block:: text

   manifest.json                     # format/PASS versions, input and asset index
   configs/<configuration-name>.json  # ordinary PASS input JSON
   assets/<filename>                 # original input/source bytes
   recipes/<index>.json               # generation settings and source references

To inspect the contents outside PASS, open the file with a ZIP-compatible archive
application, or copy it, rename the copy to ``.zip``, and extract it. Read
``manifest.json`` and ``configs/*.json`` with an ordinary text editor. In the GUI,
**Project contents** provides configuration and source previews without manual
extraction. Save project changes through the GUI so the dependency index and
checksums are updated together; editing archive members by hand can invalidate
the project.

Files are stored directly in ``configs/`` and ``assets/``, without UUID
subdirectories. Configurations use their display names and assets retain their
original filenames. Chinese characters and spaces are preserved; characters
unsafe in filenames are replaced. If a name already exists in the same directory,
PASS adds ``1``, ``2``, ``3``, and so on before the extension, for example
``rf.tfs``, ``rf1.tfs``, and ``rf2.tfs``. Configuration names follow the same rule,
for example ``beam.json`` and ``beam1.json``.

Stable IDs remain internal manifest references and do not determine filenames.
Configurations refer to assets by relative paths. The manifest records SHA-256
checksums, the dependency index, the active configuration, and run settings. Copying the project requires
only one file; the temporary editing cache
does not need to be transferred. The runtime input does not depend on paths on
the original computer. Source files are snapshots and are not silently refreshed
when an external file changes.

Saving writes and verifies a new archive before replacing the old one. Version 2
rejects unsupported format versions, unsafe paths, duplicate entries, and checksum
or dependency mismatches. Limits are 100,000 members, 256 GiB of uncompressed data,
and 64 MiB per JSON member. A large project requires time and additional disk
space to write a complete new archive. Large simulation outputs are not archived
automatically; files explicitly added as inputs or sources are embedded regardless
of their type, including results reused as input.
