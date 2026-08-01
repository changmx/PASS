Marker
====================

This module introduces the marker element **Marker** in PASS, used to mark a specific longitudinal position in the beamline without altering any particle coordinates.

The marker code is located in ``PASS/commands/element/marker.py``, with class name ``Marker`` and registration name ``marker``. The core features of the marker are as follows:

- **Zero-length element** (``length = 0.0``, not configurable), occupies no physical beamline space
- **Performs no particle coordinate transformation**; all phase space coordinates remain unchanged when particles pass through the marker
- **Supports aperture checking** via ``aperture_type`` and ``aperture_value`` parameters; ``execute_cpu`` only calls the ``check_aperture_cpu`` function
- **GPU tracking is a no-op** (``execute_gpu`` is ``pass``), performing no computation

The main purposes of the marker include:

- Marking key positions in the beamline sequence (e.g., measurement points, interaction points, injection points) for subsequent analysis
- Serving as sorting reference points; other elements can be laid out relative to marker positions
- Identifying physical positions in output and logs without participating in particle tracking


Physical Description
--------------------

The marker produces no electromagnetic field, exerts no force, and does not change particle state. When a particle passes through the marker, all six phase space coordinates (:math:`x, p_x, y, p_y, z, \delta`) remain unchanged:

.. math::

  x \leftarrow x

.. math::

  p_x \leftarrow p_x

.. math::

  y \leftarrow y

.. math::

  p_y \leftarrow p_y

.. math::

  z \leftarrow z

.. math::

  \delta \leftarrow \delta

The marker only records its longitudinal position :math:`s` in the beamline for sequence sorting and position annotation.

When aperture checking is configured (``aperture_type`` is not ``off``), the marker calls the ``check_aperture_cpu`` function in ``execute_cpu`` to perform aperture boundary checking on the transverse coordinates (:math:`(x, y)`) of alive particles (:math:`\text{tag} > 0`). Particles exceeding the aperture boundary are marked as lost (``tag`` set to negative), and the loss position and turn number are recorded. The detailed principles and type definitions of aperture checking are described in the ``Aperture`` chapter.


Interface Parameters
--------------------

All interface parameters of the marker are shown in the table below:

.. list-table::
  :header-rows: 1
  :widths: 20 25 10 10 35

  * - Property
    - JSON key
    - Type
    - Unit
    - Description
  * - ``s``
    - ``S (m)``
    - float
    - m
    - Longitudinal position of the element in the beamline
  * - ``length``
    - - (fixed 0.0, not configurable)
    - float
    - m
    - Element length (always 0, not configurable)
  * - ``name``
    - ``name``
    - str
    - -
    - Element name (automatically filled from the key name of the sequence JSON)
  * - ``aperture_type``
    - ``Aperture Type``
    - str
    - -
    - Aperture type, default ``off``, case-insensitive
  * - ``aperture_value``
    - ``Aperture Value``
    - list
    - -
    - Aperture parameter values, default ``[]``, meaning varies by type


Aperture Type Options
~~~~~~~~~~~~~~~~~~~~~

``aperture_type`` is case-insensitive and is internally converted to lowercase before matching. The available values are:

.. list-table::
  :header-rows: 1
  :widths: 15 25 60

  * - Type
    - aperture_value
    - Description
  * - ``off``
    - ignored
    - No aperture checking (default)
  * - ``default``
    - ignored
    - Default ±1m rectangular aperture
  * - ``circle``
    - ``[r]``
    - Circle, :math:`r` is the radius
  * - ``rectangle``
    - ``[w, h]``
    - Rectangle, :math:`w` is the half-width, :math:`h` is the half-height
  * - ``ellipse``
    - ``[a, b]``
    - Ellipse, :math:`a` is the semi-major axis, :math:`b` is the semi-minor axis
  * - ``rectcircle``
    - ``[w, h, r]``
    - Intersection of rectangle and circle
  * - ``rectellipse``
    - ``[w, h, a, b]``
    - Intersection of rectangle and ellipse
  * - ``racetrack``
    - ``[w, h, a, b]``
    - Racetrack shape (rectangle + elliptical ends)
  * - ``octagon``
    - ``[w, h, d]``
    - Octagon (rectangle with 45° corners cut)
  * - ``polygon``
    - ``[[x1,y1], ...]``
    - Polygon vertex list, automatically closed

.. note::

  The marker does not require a ``Length (m)`` field; the length is fixed at 0 in the code.

  The ``Command`` field should be set to ``Marker``.

  The ``off`` and ``default`` types ignore ``aperture_value``. The detailed physical descriptions and conditions for each aperture type are in the ``Aperture`` chapter.


Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

The following example places a marker at :math:`s = 12.5` m to mark the interaction point position, without enabling aperture checking:

.. code-block:: json

  {
      "IP": {
          "S (m)": 12.5,
          "Command": "Marker"
      }
  }

``"IP"`` is the element name, automatically read by ``CommandSequence`` and assigned to the ``name`` property.

Marker with Aperture Checking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The marker can also be configured with aperture checking to detect particles exceeding the beam pipe boundary at that position. The following example places a marker with a circular aperture (radius 0.05 m) at :math:`s = 12.5` m:

.. code-block:: json

  {
      "IP": {
          "S (m)": 12.5,
          "Command": "Marker",
          "Aperture Type": "circle",
          "Aperture Value": [0.05]
      }
  }

Multiple Marker Combination
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiple markers can be used repeatedly in the same beamline, for example to mark the interaction point, injection point, and measurement point:

.. code-block:: json

  {
      "IP": {
          "S (m)": 12.5,
          "Command": "Marker"
      },
      "Injection_Point": {
          "S (m)": 0.0,
          "Command": "Marker"
      },
      "Measurement_Point": {
          "S (m)": 105.3,
          "Command": "Marker",
          "Aperture Type": "rectangle",
          "Aperture Value": [0.05, 0.03]
      }
  }


Application Scenarios
---------------------

- **Interaction point marking**: Marks the beam collision position in colliders, facilitating luminosity calculation and beam-beam interaction element positioning
- **Injection point marking**: Marks the position of injection elements, facilitating the interface between the injector and the main ring
- **Measurement point marking**: Marks the location of monitors, facilitating data analysis and physics quantity extraction
- **Sequence sorting reference**: Uses the marker's ``s`` position as a reference to organize the arrangement of other elements in the beamline
- **Aperture monitoring**: Configures aperture checking at key positions (e.g., interaction point, injection point) to monitor beam loss without adding additional physical elements
