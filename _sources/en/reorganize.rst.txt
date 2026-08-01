Bunch Reorganization (ReorganizeBunch)
=======================================

This module describes the **ReorganizeBunch** command in PASS, used to dynamically adjust the grouping of bunches during simulation.

**Function Description**

The ReorganizeBunch command **only modifies the index ranges of bunches** ( ``start_idx`` and ``end_idx`` ) and **does not modify any particle coordinates** (x, px, y, py, z, dp, tag, etc.).

.. note::

   This command only reorganizes the index allocation of bunches and does not involve physical debunching or bunch compression. True debunching (bunch lengthening) and bunch compression are physical processes achieved by adjusting parameters such as RF cavity voltage and phase, not index reallocation.

Applicable scenarios:

  - **Merge** : Merge the indices of multiple bunches into fewer bunches. The z-coordinate of particles remains unchanged; only the index ranges are reallocated. For example, at injection with harmonic number :math:`h=4` producing 4 bunches, merge them into 1 bunch
  - **Split** : Regroup the indices of existing bunches into more bunches

**Code Location**

  - Source file: ``PASS/commands/reorganize.py``
  - Class name: ``ReorganizeBunch`` (inherits from ``Command`` )
  - Registered name: ``reorganizebunch``
  - Schema class: ``ReorganizeBunchElement`` ( ``PASS/para/schema/elements.py`` )


Interface Parameters
--------------------

.. list-table::
  :header-rows: 1
  :widths: 20 30 10 10 30

  * - Property
    - JSON key
    - Type
    - Unit
    - Description
  * - ``s``
    - ``S (m)``
    - float
    - m
    - s-position of the command in the ring
  * - ``name``
    - ``name``
    - str
    - -
    - Command name, automatically filled from the sequence key name
  * - ``mode``
    - ``Mode``
    - str
    - -
    - Operation mode, options: ``merge`` (merge bunches), ``split`` (split bunches)
  * - ``start_turn``
    - ``Start turn``
    - int
    - -
    - Effective starting turn (inclusive, counted from 0)
  * - ``end_turn``
    - ``End turn``
    - int
    - -
    - Effective ending turn (exclusive). Set to -1 for no upper limit, continuing until the end of the simulation
  * - ``new_num_bunch``
    - ``New num bunch``
    - int
    - -
    - New number of bunches (must be :math:`\ge 1` )


Physics Description
-------------------

The design philosophy of the ReorganizeBunch command is: **the physical position of particles is determined by the tracking process, and bunch identity is merely an index label.**

The z-coordinate of particles evolves naturally with tracking (through energy modulation in RF cavities and the slipping effect in drifts) and does not need manual adjustment. ReorganizeBunch only handles the reallocation of index ranges, enabling subsequent diagnostics, slicing, and other operations to correctly identify the new bunch structure.

Index Allocation Method
~~~~~~~~~~~~~~~~~~~~~~~

The total number of particles :math:`N_{\text{total}}` is distributed as evenly as possible among ``new_num_bunch`` bunches:

.. math::

  N_k = \left\lfloor \frac{N_{\text{total}}}{n} \right\rfloor + \begin{cases} 1 & k < N_{\text{total}} \bmod n \\ 0 & \text{otherwise} \end{cases}

where :math:`n` is the new number of bunches and :math:`k = 0, 1, \ldots, n-1` . The first :math:`N_{\text{total}} \bmod n` bunches each receive one extra particle.


Usage Example
-------------

The following example shows how to merge 4 bunches into 1 bunch at turn 500:

.. code-block:: json

  {
      "ReorganizeBunch1": {
          "S (m)": 0.0,
          "Command": "ReorganizeBunch",
          "Mode": "merge",
          "Start turn": 500,
          "End turn": -1,
          "New num bunch": 1
      }
  }


Application Scenarios
---------------------

  - **Merging indices after debunching** : At injection, a high harmonic number (e.g., :math:`h=4` ) is used to produce multiple bunches, then the RF voltage is turned off or reduced to let the bunches naturally debunch. After debunching is complete, ReorganizeBunch is used to merge the indices into 1 bunch
  - **Regrouping after rebunching** : After debunching, RF voltage is reapplied to rebunch the particles, and ReorganizeBunch is used to regroup the indices
  - **Bunch filling scheme adjustment** : Using different bunch grouping schemes at different simulation stages
