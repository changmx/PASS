Single-file projects
====================

The GUI supports ordinary standalone JSON inputs and single-file ``.passproj``
projects. The **File** menu separates JSON, project, and export actions.

Saving and opening
------------------

In standalone mode, **Save JSON** and **JSON Save as** write an ordinary PASS input.
In project mode, **Save project** writes all input JSON files, input dependencies,
sources, and generation settings into one container. ``Ctrl+S`` follows the active
mode. **Open JSON** opens an independent document; it does not silently import
into a project. Use **Import JSON into project** for that operation. Replacing a
document or closing the window offers to save unsaved changes.

Opening, saving, importing, and exporting documents run file work in a background
worker with a modal progress dialog. The event loop remains active while document
editing is blocked. A saved candidate replaces the current document only after
the operation succeeds. Cancellation is cooperative: copying and archive writing
check for cancellation between chunks, while a project archive read must finish
before its result can be discarded. Cancellation is therefore not always immediate;
an already committed save is reported as successful.

Saving standalone JSON to another directory relocates relative input references
in both the current document and its undo/redo history. Undoing a parameter edit
therefore keeps referring to the same source file. Cached dependencies used only
by an older history entry are also preserved beside the saved JSON. Saving does
not add an undo step or remove the redo branch; failed relocation or JSON writing
does not replace the applied document or its history.

**Create project from current input** packages the current configuration. Import
additional JSON files and use the input selector to switch between them. Projects
copy referenced particle distributions, offset tables, RF component programs,
Bump waveforms, WakeField model files, and magnet ramping files. Original imported JSON and MAD-X TFS files and generator settings
are retained as sources. Missing inputs identify their JSON location and prevent
an incomplete project from being saved. Additional source files can be added from
**Project contents**.

Container format
----------------

Version 1 is a standard ZIP/ZIP64 container with UTF-8 JSON metadata:

.. code-block:: text

   manifest.json                  # format/PASS versions, input and asset index
   configs/<input-id>.json         # ordinary PASS input JSON
   assets/<asset-id>/<filename>    # original input/source bytes
   recipes/<index>.json            # generation settings and source references

Input display names are separate from their stable IDs. Configurations refer to
assets by relative paths. The manifest records SHA-256 checksums and the dependency
index. Copying the project requires only one file; the temporary editing cache
does not need to be transferred. The runtime input does not depend on paths on
the original computer. Source files are snapshots and are not silently refreshed
when an external file changes.

Saving writes and verifies a new archive before replacing the old one. Version 1
rejects unsupported format versions, unsafe paths, duplicate entries, and checksum
or dependency mismatches. Limits are 100,000 members, 256 GiB of uncompressed data,
and 64 MiB per JSON member. A large project requires time and additional disk
space to write a complete new archive. Large simulation outputs are not archived
automatically; files explicitly added as inputs or sources are embedded regardless
of their type, including results reused as input.

Inspecting and reusing parameters
----------------------------------------

**Project contents** lists input JSON, source files, and generation settings.
Select a JSON or command to view its parameters and raw text. Copy a value or the
selected command's JSON to the clipboard. **Copy command into current input**
also brings its named space-charge configurations, slicers, and file dependencies.
WakeField copies its referenced Slicer, including conflict-safe slice-set renaming.
Before copying RFCavity, Bump or WakeField, differing prescribed clocks require a
choice: retain the target clock, copy the source clock, or cancel. An implicit
source clock is resolved from its harmonic-ID-zero initial bunch and circumference.
Copying that clock changes the target's global clock and can affect its existing
commands. This is not a transfer of tracked beam or wake history.
Existing names are preserved; conflicting imported names receive numeric suffixes.
Another project can be opened read-only as a parameter source.

Copying a command into the active input adds one undo step without resetting its
earlier history. Undo/redo includes the imported named configurations, slicers and
any copied clock settings. Preparation uses a private candidate; cancellation or
failure leaves the current project unchanged. Dependency files remain available
for redo, including after saving the document to a new location.

TFS and CSV files have a table preview of up to 500 rows. Text previews are bounded
to 2 MiB and parameter tables to 5,000 entries; exact files can always be exported.
Binary files display metadata and offer original-file export. Dependencies copied
into a standalone JSON are stored beside it in ``<json-name>_files`` when saved.

Generation settings describe the source of generated commands. **Load generation
settings** restores a form for preview and insertion; it does not automatically
replace manually edited commands. Plain JSON files contain the generated commands
but do not retain this additional generation metadata.

Exporting
---------

**Export current JSON** writes a copy of the configuration only. It does not
include referenced files and does not change which document is being edited.
**Export runnable input bundle** writes the run page's selected inputs and their
dependencies as ZIP, together with ``run.py`` and an English usage note. Extract
the entire bundle, install PASS, and execute ``python run.py``. The launcher
resolves inputs from the extracted directory. The bundle is ordinary input data
and does not require Qt to run.

Running
-------

Select the Beam 0 input and optionally a distinct Beam 1 input on the run page.
A project can store more candidate configurations than the engine runs at once.
Relative output directories are resolved beside the JSON or saved project; an
unsaved project uses the current working directory.

The GUI freezes the selected configurations, then validates them, copies their
dependencies, and verifies the snapshot in a background worker before starting
the child process. Preparation reports its current stage and can be cancelled;
duplicate starts are disabled during preparation and execution. Failed or
cancelled preparation leaves the previously displayed run intact. Individual
validation calls finish before observing cancellation.

Each run has separate snapshot and result directories:

.. code-block:: text

   <output>/input_snapshots/<run-id>/
       beam0.json                  # fixed runtime input; optional beam1.json
       configuration0.json         # original configuration for comparison
       assets/...                  # copied dependency bytes
       run.json                    # run record
       gui.log                     # complete process log
   <output>/runs/<run-id>/          # simulation results for this run

Runtime JSON uses absolute paths to copied assets and the dedicated result
directory. Editing the project later does not change the running task, and a new
run does not share the preceding run's output directory. Both directories remain
outside the temporary project cache.

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

A rerun records its predecessor in ``source_run``. The snapshot preserves inputs,
but does not restore the previous Python environment or source checkout.
``Random Seed: null`` retains nondeterministic sampling; even with integer seeds,
identical results are not guaranteed across changed code, libraries, precision,
or hardware.

Applied parameter changes support undo/redo; active text fields and the JSON
editor keep their own text undo/redo. Pending property and JSON edits must be
resolved before saving. Default GUI Gaussian bunches use positive transverse
emittances of 1e-6 m rad; zero-extent Gaussian generation is rejected before launch.

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
