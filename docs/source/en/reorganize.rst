Bunch Regrouping (ReorganizeBunch)
==================================

This page describes the PASS **ReorganizeBunch** command. At a selected turn, the command changes the beam bunch-grouping count and rebuilds the bunch structure from the laboratory longitudinal positions of the particles.

**Code location**

- Source: ``PASS/commands/reorganize.py``
- Regrouping algorithm: ``PASS/commands/sort_bunch.py``
- Class: ``ReorganizeBunch`` (inherits from ``Command``)
- Registered name: ``reorganizebunch``
- Schema: ``ReorganizeBunchElement`` in ``PASS/para/schema/elements.py``


Operation
---------

Let the old and new grouping counts be :math:`h_{\mathrm{old}}` and :math:`h_{\mathrm{new}}`. The command runs once at ``Start turn`` and performs the following operations:

1. Recover each particle's laboratory longitudinal position from its old bunch reference:

   .. math::

      z_{\mathrm{lab}} = z_{\mathrm{rel}} + z_{\mathrm{center,old}}.

2. Build a new grid of bunch centers separated by :math:`C/h_{\mathrm{new}}`:

   .. math::

      z_{\mathrm{center},k} = k\frac{C}{h_{\mathrm{new}}},
      \qquad k=0,1,\ldots,h_{\mathrm{new}}-1.

3. Assign particles to the nearest new group center around the ring and reorder every particle array so that each new bunch occupies a contiguous index range.
4. Convert laboratory positions back to coordinates relative to the new bunch center:

   .. math::

      z_{\mathrm{rel,new}}
      = \operatorname{fold}_C
        \left(z_{\mathrm{lab}}-z_{\mathrm{center,new}}\right).

5. Update the beam ``harmonic_number`` and each bunch's ``harmonic_id``, ``z_center``, particle count, and index range.
6. If a new bunch inherits a different reference momentum, rebase :math:`p_x`, :math:`p_y`, and :math:`\delta` so that each particle's absolute mechanical momentum is preserved.

ReorganizeBunch is therefore more than an index edit, but it is not itself a physical debunching, merging, capture, or compression process. Laboratory positions are preserved, while bunch reference centers, relative longitudinal coordinates, and normalized momenta may change.


Group Boundaries
----------------

The algorithm uses the ring-azimuth sorting key

.. math::

   k_z = \left(z_{\mathrm{lab}}+\frac{C}{2h_{\mathrm{new}}}\right)\bmod C.

Group :math:`j` contains particles satisfying

.. math::

   j\frac{C}{h_{\mathrm{new}}}
   \le k_z
   < (j+1)\frac{C}{h_{\mathrm{new}}}.

The half-group-width shift places each boundary midway between adjacent centers. The same rule applies to odd and even grouping counts.


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
- Reclassify particles that have crossed old group boundaries according to their current laboratory azimuth

.. note::

   ReorganizeBunch changes the PASS bunch-reference grouping only. It does not replace the physical debunching, capture, merging, or bunch-compression process produced by RF elements. First create the intended longitudinal distribution with the appropriate physical elements, then regroup at the selected turn.
