Spectral analysis and frequency maps
====================================

The **Analysis** workspace provides basic FFT, refined FFT and frequency-map
analysis (FMA). It uses the same array functions as Python scripts. Analysis
does not require a tracking input or an open project. BPM reconstruction,
OMC3 integration and optics reconstruction are outside this workflow.

Independent Python functions
----------------------------

The public functions are ``compute_fft`` in ``PASS/analysis/fft.py``,
``compute_refined_fft`` in ``PASS/analysis/refined_fft.py`` and
``compute_fma`` in ``PASS/analysis/fma.py``. Each complete function definition
can be copied into another Python file and used with NumPy installed: imports
and implementation helpers are inside the function. There are no PASS, Qt,
file-reader or plotting dependencies in these three functions. Inputs are not
modified; results are dictionaries containing arrays and metadata.

.. code-block:: python

   import numpy as np
   from PASS.analysis import compute_fft, compute_refined_fft, compute_fma

   turns = np.arange(2048)
   x = 0.002 * np.cos(2 * np.pi * 0.2317 * turns + 0.3)
   y = 0.001 * np.cos(2 * np.pi * 0.3172 * turns - 0.2)

   spectrum = compute_fft(x, sample_spacing=1.0)
   peaks = compute_refined_fft(
       x, window="hann", padding_factor=8,
       frequency_range=(0.20, 0.27), n_peaks=1,
   )
   print(peaks["peak_frequency"], peaks["peak_amplitude"])

   # Rows identify particles; the last axis contains consecutive samples.
   fma = compute_fma(
       x[None, :], y[None, :],
       windows=((0, 1024), (1024, 2048)),
       frequency_range_x=(0.20, 0.27),
       frequency_range_y=(0.29, 0.35),
   )
   print(fma["qx_first"], fma["qx_second"], fma["drift"])

FFT conventions
---------------

``sample_spacing`` is finite and positive. One sample per turn with spacing
1 gives cycles/turn; spacing in seconds gives Hz. Sampling every several
turns requires the actual turn spacing, with the corresponding smaller
Nyquist band. The FFT does not recover an integer tune or remove aliasing.
For real turn-by-turn signals, the positive-frequency spectrum alone cannot
distinguish fractional tunes ``Q`` and ``1-Q``.

``compute_fft`` accepts a real or complex array, ``axis`` (default -1), and
``remove_mean`` (default false). Output arrays put frequency on the last axis.
Real signals use a one-sided spectrum. Complex signals use a sorted signed
two-sided spectrum. ``coefficients`` are amplitude-normalized complex
coefficients, ``amplitude`` is their magnitude and ``phase`` is in radians,
referenced to the first sample of the analyzed interval. Real-signal positive
frequencies are doubled except DC and the even-length Nyquist bin. These are
amplitude spectra, not power spectral densities.

Refined FFT accepts ``window`` (``rectangle``, ``hann``, ``hamming`` or
``blackman``), integer ``padding_factor``, ``interpolation`` (``none`` or
``parabolic``), ``frequency_range`` and ``n_peaks``. Defaults are Hann, factor
8, parabolic interpolation and one peak. It corrects the window's coherent
gain, locates spectral peaks and evaluates their complex coefficients at the
estimated frequencies. Zero-padding changes the frequency sampling grid,
not the number of observed turns or the physical resolving power. Accuracy
depends on record length, window, noise, nearby lines and time variation.
The function reports peak validity and quality; an invalid result is not a
measured zero frequency. ``return_spectrum=False`` avoids returning spectrum
arrays when only peak estimates are needed.

FMA conventions
---------------

FMA compares frequencies in two time windows; it is not another FFT method.
``x`` and ``y`` must have the same shape, with samples on the last axis.
An individual trajectory is one-dimensional; particle ensembles have leading
object dimensions. Explicit windows are ``((start1, end1), (start2, end2))``,
with exclusive end indices. They must be equal-length, ordered and disjoint.
The default uses equal adjacent halves and leaves an odd final sample unused.
Returned metadata records the exact windows.

``method`` selects ``fft`` or ``refined_fft``. Mean removal defaults to true
for FMA. The standalone function contains its own default estimator; the
optional ``frequency_estimator`` hook permits an explicitly supplied custom
estimator without making one necessary. See its docstring for the callback
contract. The default and standalone frequency estimators use the same
numerical conventions.

For the two transverse signals the reported quantities are

.. math::

   \Delta Q_x = Q_{x,2}-Q_{x,1},\qquad
   \Delta Q_y = Q_{y,2}-Q_{y,1},\qquad
   d_Q=\sqrt{(\Delta Q_x)^2+(\Delta Q_y)^2},\qquad
   D=\log_{10}d_Q.

The keys are ``qx_first``, ``qy_first``, ``qx_second``, ``qy_second``,
``delta_qx``, ``delta_qy``, ``drift`` and ``diffusion_log10``. Exact zero
drift retains ``D=-inf``; plotting limits do not change the numeric value.
Differences are not implicitly wrapped. Use consistent frequency branches
and search bands in both windows. Nonfinite samples or invalid frequency
estimates mark that object's result invalid through ``valid`` and ``quality``.
Time-dependent settings, noise and collective evolution can also cause
frequency drift; this diagnostic alone does not establish chaotic motion.

Files and sampling
------------------

``PASS.analysis.data_io.inspect_data`` lists selectable numeric arrays or
columns. ``load_signal`` loads an explicit selection and returns the signal,
sampling coordinates, spacing and source metadata. The numeric functions
themselves accept arrays, independently of file formats.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Format
     - Selection
   * - CSV / TSV
     - Named numeric columns; configurable header and skipped rows.
   * - TXT / DAT
     - Delimited numeric text; select delimiter, header handling and skipped rows.
   * - TFS
     - Numeric columns with TFS header metadata.
   * - HDF5 (``.h5``, ``.hdf5``)
     - Explicit numeric dataset path and sample axis; file/dataset attributes are retained.
   * - NPY
     - The ``data`` array and its explicit sample axis.
   * - NPZ
     - A named numeric array and its explicit sample axis.

Object arrays and pickle payloads are not loaded. For multidimensional arrays,
choose the sampling axis explicitly. ``sample_range`` and ``object_range``
use half-open index ranges. An explicit ``object_range`` flattens the leading
object dimensions in C order; omitting it preserves those dimensions.
Returned ``object_ids`` are positions in that original array, not physical
particle identifiers. The caller must ensure that paired X/Y arrays follow
the same particle order.
A supplied coordinate column/dataset must be
strictly increasing and uniformly spaced; its spacing is used instead of a
manually entered value. Missing turns, repeated coordinates and irregular
sampling are not silently dropped, sorted, interpolated or zero-filled.
Identified PASS ParticleMonitor files automatically use their ``turn`` column.
Otherwise, without coordinates the caller must specify the spacing.

CSV defaults to commas, TSV to tabs, and TXT/DAT to whitespace. Text headers
can be ``auto``, ``present`` or ``absent``; headerless columns are named
``column_0``, ``column_1``, and so on. HDF5/NPY selections are sliced before
copying; NPZ loads the complete selected member and text loads the table.

PASS ParticleMonitor output freezes lost-particle records. The adapter uses
the signed tag for identified ParticleMonitor files, including files converted
with PASS metadata. It rejects a selection containing lost or nonfinite samples.
For external files, an explicit alive/status selection can declare positive
values live. Analyze an intact live interval; do not treat a frozen
tail as an oscillation. A generic table containing repeated turns for
different particles must be organized into separate trajectories first.

GUI workflow
------------

1. Open **Analysis**, choose **Spectrum** or **FMA**, then select a data file.
2. Inspect the columns/datasets and choose signal(s), sample axis, coordinate
   column or spacing, and the desired sample/object ranges.
3. For a spectrum choose basic or refined FFT. Refined settings include the
   window, padding, interpolation and peak search band.
4. For FMA select the paired signals, two equal-length windows and frequency
   estimation settings. Window indices refer to the selected signal interval.
5. Run the calculation, inspect the plotted and numeric results, then export
   numerical data or the figure. **Copy Python call** reproduces the selected
   data and calculation parameters in a script.

Reading and calculation run in a background worker. Display previews may be
bounded, but calculation uses the selected full-resolution arrays. Cancellation
waits for the current numerical operation to finish and discards its result.
Ordinary file opening does not change the tracking configuration.

NPZ export retains result arrays, selected input arrays, coordinates and
metadata. CSV exports complete numeric results with a metadata comment;
spectral rows use ``record_type=spectrum`` and refined peak rows use
``record_type=peak`` with validity and quality fields. PNG/SVG export saves
the displayed figure, including its preview point limit.
