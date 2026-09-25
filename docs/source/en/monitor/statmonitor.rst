StatMonitor
======================

``StatMonitor`` records each bunch's centroid, standard deviations, RMS emittances, derived Twiss parameters and particle losses once per turn at a specified lattice position. Moments use live particles (``tag > 0``). Each monitor writes a per-bunch history in CSV and the selected table format.

Configuration example
------------------------------------------

.. code-block:: python

   from PASS.para.schema.monitors import StatMonitorItem
   from PASS.para.schema.sequence import Sequence

   sequence = Sequence()
   sequence.add("stat1", StatMonitorItem(s=0.0, write_interval_turns=100))

Add this item to the complete sequence. ``write_interval_turns`` controls disk writes; statistics are still recorded on every turn. The command name comes from the sequence key.

Interface Parameters
--------------------

.. list-table::
  :header-rows: 1
  :widths: 20 20 10 10 40

  * - Python field
    - JSON key
    - Type
    - Default
    - Description
  * - ``s``
    - ``"S (m)"``
    - float
    - Required
    - Longitudinal position of the monitor in the beamline
  * - ``command``
    - ``"Command"``
    - str
    - ``"StatMonitor"``
    - Command type identifier

  * - ``output_format``
    - ``"Output format"``
    - str
    - ``"hdf5-gzip1"``
    - ``"hdf5-gzip1"`` (gzip-1 + shuffle), ``"hdf5"`` (uncompressed), or ``"tfs"``; CSV is always provided
  * - ``write_interval_turns``
    - ``"Write interval (turns)"``
    - positive int
    - 100
    - Batch write interval in turns; all intervening rows are retained

.. note::

  The statistics target all surviving particles in the bunch at that position (``tag > 0``); no particle indices need to be specified.


Output Files
------------

A pair of files is generated for each bunch at each monitor position:

- **CSV** (appended in batches): ``{hms}_stat_beam{bid}_bunch{bid}_Np_{Np}_s_{s:.4f}.csv``
- **HDF5** (default, appended with CSV): ``{hms}_stat_beam{bid}_bunch{bid}_Np_{Np}_s_{s:.4f}.h5``
- **TFS** (instead of HDF5 when selected, generated at finalization): ``{hms}_stat_beam{bid}_bunch{bid}_Np_{Np}_s_{s:.4f}.tfs``

The output directory is ``output_dir_stat``.

Example metadata (HDF5 attributes, or TFS headers in text mode):

::

   @ Name             PASS Statistic Data
   @ Time             2026-07-14 00:11:03

Output columns:

.. list-table::
  :header-rows: 1
  :widths: 25 15 60

  * - Column name
    - Group
    - Description
  * - ``turn``
    - Basic
    - Turn number
  * - ``xAverage``
    - Centroid
    - Horizontal position mean :math:`\langle x \rangle`
  * - ``pxAverage``
    - Centroid
    - Horizontal momentum mean :math:`\langle p_x \rangle`
  * - ``sigmaX``
    - Beam size
    - Horizontal position standard deviation :math:`\sigma_x`
  * - ``sigmaPx``
    - Beam size
    - Horizontal momentum standard deviation :math:`\sigma_{p_x}`
  * - ``yAverage``
    - Centroid
    - Vertical position mean
  * - ``pyAverage``
    - Centroid
    - Vertical momentum mean
  * - ``sigmaY``
    - Beam size
    - Vertical position standard deviation
  * - ``sigmaPy``
    - Beam size
    - Vertical momentum standard deviation
  * - ``zAverage``
    - Centroid
    - Mean of the folded bunch-relative coordinate :math:`\langle z_{\mathrm{rel}}\rangle`
  * - ``dpAverage``
    - Centroid
    - Momentum deviation mean
  * - ``sigmaZ``
    - Beam size
    - Standard deviation of the folded bunch-relative coordinate
  * - ``sigmadp``
    - Beam size
    - Momentum deviation standard deviation
  * - ``xEmittance``
    - Emittance
    - Horizontal 2D emittance :math:`\varepsilon_x`
  * - ``yEmittance``
    - Emittance
    - Vertical 2D emittance :math:`\varepsilon_y`
  * - ``betax``
    - Twiss
    - Horizontal beta function
  * - ``betay``
    - Twiss
    - Vertical beta function
  * - ``alphax``
    - Twiss
    - Horizontal alpha function
  * - ``alphay``
    - Twiss
    - Vertical alpha function
  * - ``gammax``
    - Twiss
    - Horizontal gamma function
  * - ``gammay``
    - Twiss
    - Vertical gamma function
  * - ``invariantx``
    - Verification
    - Horizontal invariant :math:`\gamma_x \beta_x - \alpha_x^2` (should equal 1)
  * - ``invarianty``
    - Verification
    - Vertical invariant (should equal 1)
  * - ``zCenter``
    - Longitudinal reference
    - Nominal bunch-grouping slot (not the physical centroid), :math:`z_{\mathrm{center}}`
  * - ``referenceTime``
    - Reference
    - Reference passage time at this observation (s)
  * - ``referenceBeta``
    - Reference
    - Reference velocity divided by c at this observation
  * - ``referenceMomentum``
    - Reference
    - Reference mechanical momentum at this observation (eV/c per nucleon for ions)
  * - ``sigmaTime``
    - Beam size
    - Passage-time standard deviation from continuous z (s)
  * - ``xzAverage``
    - Correlation
    - :math:`\langle x \, z \rangle`
  * - ``xyAverage``
    - Correlation
    - :math:`\langle x \, y \rangle`
  * - ``yzAverage``
    - Correlation
    - :math:`\langle y \, z \rangle`
  * - ``xzDevideSigmaxSigmaz``
    - Correlation
    - :math:`\langle x \, z \rangle / (\sigma_x \, \sigma_z)` normalized raw cross moment, not a centered Pearson coefficient
  * - ``beamLossTotal``
    - Loss
    - Number of lost particles
  * - ``lossPercent``
    - Loss
    - Loss percentage
  * - ``xSkewness``
    - Higher-order moments
    - Horizontal skewness
  * - ``xKurtosis``
    - Higher-order moments
    - Horizontal kurtosis
  * - ``ySkewness``
    - Higher-order moments
    - Vertical skewness
  * - ``yKurtosis``
    - Higher-order moments
    - Vertical kurtosis
  * - ``Ek``
    - Energy
    - Bunch kinetic energy


Longitudinal coordinate in output
-------------------------------------------

Longitudinal moments use a temporary full-ring representative of bunch-relative
z in ``[-C/2,C/2)`` on both CPU and GPU. This never changes stored particle z.
HDF5 attributes and TFS headers record ``ZCoordinate=z_rel_folded_by_ring`` and ``ZInterval``.
These are moments of the chosen representative, not unwrapped slip statistics
or circular moments; a distribution crossing the interval cut can have a large
reported width. ParticleMonitor and Distribution retain the continuous z_rel
for analyses that need accumulated longitudinal slip.

Injection populations
---------------------

numAlive, numInjected and numPending record live, born (live plus lost), and
reserved macro-particle counts. beamLossTotal excludes pending slots;
lossPercent uses the injected population as its denominator. An allocated
population with no survivors produces zero moments and explicit zero-survival
counts on both CPU and GPU; an empty declared bunch retains the previous no-row behavior.

Batching, final partial batches, live CSV inspection and HDF5 layout are
described in :doc:`table_output`.
Working Principle
-----------------

Statistics Computation
~~~~~~~~~~~~~~~~~~~~~~

For :math:`N` surviving particles in the bunch (:math:`\text{tag} > 0`), the moments of each order are defined as:

First-order moment (centroid):

.. math::

   \langle x \rangle = \frac{1}{N} \sum_{i=1}^{N} x_i

Second-order moment:

.. math::

   \langle x^2 \rangle = \frac{1}{N} \sum_{i=1}^{N} x_i^2

Uncentered cross moment:

.. math::

   \langle x \, p_x \rangle = \frac{1}{N} \sum_{i=1}^{N} x_i \, p_{x,i}

Beam size (standard deviation):

.. math::

   \sigma_x = \sqrt{\langle x^2 \rangle - \langle x \rangle^2}

Similarly, :math:`\sigma_{p_x}`, :math:`\sigma_y`, :math:`\sigma_{p_y}`, :math:`\sigma_z`, :math:`\sigma_{\delta}` are computed.

The implementation evaluates the equivalent centered moments instead of subtracting
large raw moments. CPU and GPU use FP64 accumulators for both FP32 and FP64
particles: first find the centroid relative to a surviving particle, then accumulate
moments about that centroid. This also stabilizes covariance, emittance, skewness
and kurtosis for narrow or displaced bunches. Particle storage precision is unchanged.
The temporary ring projection for z statistics is evaluated in FP64.

``sigmaZ`` and z moments retain a temporary ring-period projection, without changing stored continuous z. ``sigmaTime`` uses the standard deviation of unwrapped z divided by :math:`\beta_b c`, giving the physical passage-time spread. Rows include ``referenceTime``, ``referenceBeta`` and ``referenceMomentum``. Nominal zCenter cannot reconstruct a laboratory centroid.

Emittance and Twiss Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 2D emittance is derived from second-order moments:

.. math::

   \varepsilon_x = \sqrt{\sigma_x^2 \, \sigma_{p_x}^2 - \sigma_{x,p_x}^2}

where :math:`\sigma_{x,p_x} = \langle x \, p_x \rangle - \langle x \rangle \langle p_x \rangle` is the covariance.

Twiss parameters:

.. math::

   \beta_x = \frac{\sigma_x^2}{\varepsilon_x}

.. math::

   \alpha_x = -\frac{\sigma_{x,p_x}}{\varepsilon_x}

.. math::

   \gamma_x = \frac{\sigma_{p_x}^2}{\varepsilon_x}

Invariant verification:

.. math::

   \gamma_x \, \beta_x - \alpha_x^2 = 1

The formulas for the vertical direction (y) are identical in form; simply replace the subscript x with y.

Higher-Order Moments
~~~~~~~~~~~~~~~~~~~~

Skewness (third standardized moment):

.. math::

   S_x = \frac{\langle x^3 \rangle - 3 \langle x \rangle \sigma_x^2 - \langle x \rangle^3}{\sigma_x^3}

Kurtosis (fourth standardized moment):

.. math::

   K_x = \frac{\langle x^4 \rangle - 4 \langle x \rangle \langle x^3 \rangle + 2 \langle x \rangle^2 \langle x^2 \rangle + 4 \langle x \rangle^2 \sigma_x^2 + \langle x \rangle^4}{\sigma_x^4}

Beam Loss
~~~~~~~~~

.. math::

   N_{\text{loss}} = N_{\text{injected}} - N_{\text{alive}}

.. math::

   \text{loss\%} = \frac{N_{\text{loss}}}{N_{\text{injected}}} \times 100\%

where :math:`N_{\text{injected}}` is the number of already injected macro particles (live plus lost), and :math:`N_{\text{alive}}` is the current number of surviving particles.

Numerical precision and interpretation
----------------------------------------------------------------------------

CPU and GPU accumulate centered moments in float64, including for float32 particle storage. The stored particle coordinates retain their configured precision. This reduces cancellation in narrow or displaced bunches, but does not eliminate finite-sampling or tracking errors. GPU records are transferred in batches at the configured write interval.

When an emittance is zero, the corresponding derived Twiss parameters and invariant are reported as zero. For nonzero emittance the identity gamma*beta-alpha²=1 follows from the definitions; it is not an independent validation of tracking. Skewness and kurtosis are centered standardized moments; kurtosis is not excess kurtosis. A nonzero centroid contributes to xzAverage and its normalized output, so neither is a centered covariance by itself.

Reference time and z definitions follow :ref:`en-longitudinal-reference`. Batch finalization and live inspection are described in :doc:`table_output`.
