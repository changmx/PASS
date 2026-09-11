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

**Create project from current input** packages the current configuration. Import
additional JSON files and use the input selector to switch between them. Projects
copy referenced particle distributions, offset tables, RF tables, and magnet
ramping files. Original imported JSON and MAD-X TFS files and generator settings
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
Existing names are preserved; conflicting imported names receive numeric suffixes.
Another project can be opened read-only as a parameter source.

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

Before starting the child process, the GUI validates all selected inputs and
writes a fixed snapshot under ``<output>/input_snapshots/<run-id>``. Snapshot JSON
uses absolute paths to copied assets; output remains outside the temporary project
cache. Editing the project later does not change an already running task. Stop
terminates the child process. Initialization and tracking errors produce a failed
run, including errors logged by the engine without re-raising an exception.

Applied parameter changes support undo/redo; active text fields and the JSON
editor keep their own text undo/redo. Pending property and JSON edits must be
resolved before saving. Default GUI Gaussian bunches use positive transverse
emittances of 1e-6 m rad; zero-extent Gaussian generation is rejected before launch.
