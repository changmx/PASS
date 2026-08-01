StatMonitor
==========================

Introduction
------------

``StatMonitor`` is a beam statistics monitor that records bunch statistics turn-by-turn at a specified longitudinal position, including centroid position, beam size, emittance, Twiss parameters, higher-order moments, and beam loss. It is the core tool for evaluating beam quality evolution and diagnosing beam dynamics behavior.

- **Code location**: ``PASS/commands/monitor/statistic.py``
- **Class name**: ``StatMonitor``, registered name ``"statmonitor"``
- **Key features**:

  - Computes bunch statistics in 6D phase space turn-by-turn (first through fourth order moments);
  - Derives emittance and Twiss parameters (beta, alpha, gamma) from second-order moments;
  - Records beam loss count and loss percentage;
  - CPU uses numpy vectorized computation, GPU uses CUDA kernel functions + warp reduction;
  - Appends data to CSV each turn, converts to TFS format uniformly on the final turn;
  - Only surviving particles (``tag > 0``) are counted; lost particles are excluded.


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

Covariance:

.. math::

   \langle x \, p_x \rangle = \frac{1}{N} \sum_{i=1}^{N} x_i \, p_{x,i}

Beam size (standard deviation):

.. math::

   \sigma_x = \sqrt{\langle x^2 \rangle - \langle x \rangle^2}

Similarly, :math:`\sigma_{p_x}`, :math:`\sigma_y`, :math:`\sigma_{p_y}`, :math:`\sigma_z`, :math:`\sigma_{\delta}` are computed.

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

   N_{\text{loss}} = N_{\text{total}} - N_{\text{alive}}

.. math::

   \text{loss\%} = \frac{N_{\text{loss}}}{N_{\text{total}}} \times 100\%

where :math:`N_{\text{total}}` is the initial number of macro particles in the bunch, and :math:`N_{\text{alive}}` is the current number of surviving particles.

GPU Implementation
~~~~~~~~~~~~~~~~~~

The GPU version uses the CUDA kernel function ``calc_all_stats``, employing a grid stride loop to traverse particles. Each thread accumulates 22 statistics in registers, then through warp reduction (``__shfl_down_sync``) and block reduction, writes the global result via ``atomicAdd``. The maximum number of blocks is 512 (due to ``atomicAdd`` contention overhead).


Interface Parameters
--------------------

.. list-table::
  :header-rows: 1
  :widths: 20 20 10 10 40

  * - Property
    - JSON key
    - Type
    - Default
    - Description
  * - ``s``
    - ``"S (m)"``
    - float
    - Required
    - Longitudinal position of the monitor in the beamline
  * - ``cmd_name``
    - ``"name"``
    - str
    - Required
    - Monitor name (automatically filled from the sequence key name)
  * - ``command``
    - ``"Command"``
    - str
    - ``"StatMonitor"``
    - Command type identifier

.. note::

  ``StatMonitor`` has no additional configuration parameters. The statistics target all surviving particles in the bunch at that position (``tag > 0``); no particle indices need to be specified.


Output Files
------------

A pair of files is generated for each bunch at each monitor position:

- **CSV** (appended turn-by-turn): ``{hms}_stat_beam{bid}_bunch{bid}_Np_{Np}_s_{s:.4f}.csv``
- **TFS** (converted from CSV on the final turn): ``{hms}_stat_beam{bid}_bunch{bid}_Np_{Np}_s_{s:.4f}.tfs``

The output directory is ``output_dir_stat``.

TFS file header:

::

   @ Name             PASS Statistic Data
   @ Time             2026-07-14 00:11:03

Output columns (34 columns total):

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
    - Longitudinal position mean
  * - ``dpAverage``
    - Centroid
    - Momentum deviation mean
  * - ``sigmaZ``
    - Beam size
    - Longitudinal position standard deviation
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
    - :math:`\langle x \, z \rangle / (\sigma_x \, \sigma_z)` normalized correlation
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


Usage Example
-------------

The following JSON snippet places a statistics monitor at :math:`s = 0.0` m:

.. code-block:: json

   "SM1": {
       "S (m)": 0.0,
       "Command": "StatMonitor"
   }

The statistics monitor requires no additional parameters; only the position and command type need to be specified. During simulation, the bunch statistics at that position are recorded turn-by-turn.

Multi-position Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~

Multiple statistics monitors can be placed at different positions to compare the variation of bunch statistics along the beamline:

.. code-block:: json

   "SM_start": {
       "S (m)": 0.0,
       "Command": "StatMonitor"
   },
   "SM_mid": {
       "S (m)": 100.0,
       "Command": "StatMonitor"
   },
   "SM_end": {
       "S (m)": 250.0,
       "Command": "StatMonitor"
   }


Application Scenarios
---------------------

- **Beam quality assessment**: Monitor the evolution of emittance, beam size, and centroid position turn-by-turn to evaluate whether beam quality is stable or degrading
- **Emittance measurement**: Compute emittance and Twiss parameters from second-order moments, and compare with design values for verification
- **Beam loss diagnostics**: Monitor beam loss rate through ``beamLossTotal`` and ``lossPercent``, identifying the turn and position where losses occur
- **Nonlinear effect identification**: Use higher-order moment information (skewness and kurtosis) to determine the degree to which the beam distribution deviates from Gaussian, identifying nonlinear resonances or dispersion coupling
- **Momentum spread monitoring**: ``sigmadp`` and ``sigmaZ`` reflect longitudinal beam quality, supporting longitudinal dynamics studies
- **Correlation diagnostics**: Correlation quantities such as ``xzAverage`` can be used to diagnose dispersion coupling or transverse-longitudinal coupling
