Bunch Regrouping (ReorganizeBunch)
==================================

This page describes the PASS **ReorganizeBunch** command. At a selected turn, the command changes the beam bunch-grouping count and rebuilds the bunch structure from the physical arrival phases of the particles.

**Code location**

- Source: ``PASS/commands/reorganize.py``
- Regrouping algorithm: ``PASS/commands/sort_bunch.py``
- Class: ``ReorganizeBunch`` (inherits from ``Command``)
- Registered name: ``reorganizebunch``
- Schema: ``ReorganizeBunchElement`` in ``PASS/para/schema/elements.py``


Operation and group boundaries
------------------------------

At ``Start turn``, the new grouping count is h. With the prescribed reference
clock phase :math:`\Psi(t)=\int_{t_*}^t f_{rev}(u)du`, the sorting key at s is

.. math::

   k_i=\left[-\Psi(t_i)+s/C+1/(2h)\right]\bmod1,\qquad
   t_i=T_b-z_i/(\beta_b c).

Group j owns :math:`j/h\le k_i<(j+1)/h`. The half-slot shift treats odd and
even grouping counts identically. All particle arrays are permuted together.
The periodic key is used only for grouping; the unwrapped time is retained.

New reference events are chosen from the prescribed clock for the current
passage and new slot IDs. Their reference velocities are :math:`C f_{rev}(T_b)`
and must be subluminal. The reference energy follows from this velocity and
rest mass. A different reference energy is a coordinate choice, not an RF kick:

.. math::

   z_i'=\beta_b'c(T_b'-t_i),\qquad
   p_{x,y}'=p_{x,y}P_{0,b}/P_{0,b}',\qquad
   1+\delta_i'=(1+\delta_i)P_{0,b}/P_{0,b}'.

Physical time, energy and mechanical momenta remain unchanged. ``SortBunch``
uses the same key and transformation while retaining the existing bunch
references. Both commands invalidate old SliceSets because their local particle
indices change. Users explicitly execute Slicer afterwards. No automatic
centroid recentering, physical debunching, merging or compression is performed.

Interface Parameters
--------------------

.. list-table::
  :header-rows: 1
  :widths: 22 30 12 12 24

  * - Property
    - JSON key
    - Type
    - Default
    - Description
  * - ``s``
    - ``S (m)``
    - float
    - Required
    - Longitudinal position of the command in the ring
  * - ``name``
    - ``name``
    - str
    - Auto-filled
    - Command name
  * - ``start_turn``
    - ``Start turn``
    - int
    - 0
    - Execution turn (inclusive, 0-based); the command runs only once
  * - ``new_harmonic``
    - ``New harmonic number``
    - int
    - Required
    - New bunch-grouping count, must be :math:`\ge 1`


Usage Example
-------------

The following example switches the beam to one longitudinal group at turn 500:

.. code-block:: json

  {
      "ReorganizeBunch1": {
          "S (m)": 0.0,
          "Command": "ReorganizeBunch",
          "Start turn": 500,
          "New harmonic number": 1
      }
  }


Applications
------------

- Update diagnostic grouping after RF manipulations have changed the longitudinal distribution
- Change the bunch-grouping count between simulation stages
- Reclassify particles that have crossed old group boundaries according to their current machine-clock phase

.. note::

   ReorganizeBunch changes the PASS bunch-reference grouping only. It does not replace the physical debunching, capture, merging, or bunch-compression process produced by RF elements. First create the intended longitudinal distribution with the appropriate physical elements, then regroup at the selected turn.


See :ref:`en-longitudinal-reference` and :doc:`slicer`。
