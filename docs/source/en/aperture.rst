Aperture
========

This module describes the **aperture checking system** (Aperture) in PASS, used to check whether particles exceed the transverse aperture boundaries of the beam pipe during particle tracking. Aperture checking is a core component of beam loss simulation, capable of identifying and recording particles lost due to transverse coordinates exceeding physical pipe limits.

The aperture module is located at ``PASS/utils/aperture.py`` and provides both CPU and GPU implementations (called via the ``check_aperture_cpu`` and ``check_aperture_gpu`` functions, respectively), automatically invoked after each element tracking. The current element's longitudinal position :math:`s` and the current turn number are passed in at call time.

Aperture checking is performed only on the transverse coordinates :math:`(x, y)` of particles and does not involve longitudinal coordinates. Each element can independently set its aperture type and parameters, supporting 10 aperture geometries.


Interface Parameters
--------------------

The aperture system is controlled by two parameters:

.. list-table::
  :header-rows: 1
  :widths: 20 25 15 40

  * - Property
    - JSON key
    - Type
    - Description
  * - ``aperture_type``
    - ``Aperture Type``
    - str
    - Aperture type, case-insensitive; available values are listed below
  * - ``aperture_value``
    - ``Aperture Value``
    - list
    - Aperture parameter values; meaning varies by type

.. note::

  ``aperture_type`` is case-insensitive and is internally converted to lowercase before matching. The ``off`` and ``default`` types ignore ``aperture_value`` .


Lost Particle Handling
----------------------

When a particle is determined to be lost, the system performs the following operations:

- **tag negation** : :math:`\text{tag} \leftarrow -|\text{tag}|` , preserving the particle ID information and only negating the sign to mark it as lost
- **lost_position** : records the longitudinal coordinate :math:`s` of the loss location, i.e., the longitudinal position of the current element
- **lost_turn** : records the turn number at the time of loss

Particles already lost ( :math:`\text{tag} < 0` ) are skipped in subsequent aperture checks and are not marked again. Aperture checking is performed only on surviving particles ( :math:`\text{tag} > 0` ).


Detailed Aperture Types
-----------------------

The following describes the parameter definitions and loss conditions for each of the 10 aperture types.

off (Disabled)
~~~~~~~~~~~~~~

**Parameters** : none ( ``aperture_value`` is ignored)

**Description** : No aperture checking is performed; all particles are retained.

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <circle cx="150" cy="150" r="60" stroke="#e94560" stroke-width="3" fill="none"/>
    <line x1="108" y1="108" x2="192" y2="192" stroke="#e94560" stroke-width="3"/>
    <text x="132" y="248" fill="#e94560" font-size="16" font-weight="bold" font-family="monospace">OFF</text>
  </svg>
  </div>


default (Default Rectangle)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Parameters** : none ( ``aperture_value`` is ignored)

**Description** : Uses the default ±1m rectangular aperture, equivalent to the ``rectangle`` type with ``aperture_value = [1.0, 1.0]`` .

**Loss condition** :

.. math::

  |x| > 1.0 \quad \text{or} \quad |y| > 1.0

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <rect x="90" y="90" width="120" height="120" stroke="#00d2ff" stroke-width="2" fill="#00d2ff22"/>
    <text x="125" y="84" fill="#00d2ff" font-size="11" font-family="monospace">+1m</text>
    <text x="125" y="228" fill="#00d2ff" font-size="11" font-family="monospace">-1m</text>
  </svg>
  </div>


circle (Circular)
~~~~~~~~~~~~~~~~~

**Parameters** : ``aperture_value = [r]`` , where :math:`r` is the circle radius.

**Loss condition** :

.. math::

  x^2 + y^2 > r^2

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <circle cx="150" cy="150" r="108" stroke="#00d2ff" stroke-width="2" fill="#00d2ff22"/>
    <line x1="150" y1="150" x2="258" y2="150" stroke="#f5a623" stroke-width="1.5" stroke-dasharray="3,3"/>
    <text x="195" y="142" fill="#f5a623" font-size="13" font-style="italic" font-family="monospace">r</text>
  </svg>
  </div>


rectangle (Rectangular)
~~~~~~~~~~~~~~~~~~~~~~~

**Parameters** : ``aperture_value = [w, h]`` , where :math:`w` is the half-width and :math:`h` is the half-height.

**Loss condition** (particle is lost if either condition is met):

.. math::

  |x| > w

.. math::

  |y| > h

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <rect x="30" y="66" width="240" height="168" stroke="#00d2ff" stroke-width="2" fill="#00d2ff22"/>
    <line x1="150" y1="150" x2="270" y2="150" stroke="#f5a623" stroke-width="1.5" stroke-dasharray="3,3"/>
    <text x="200" y="142" fill="#f5a623" font-size="13" font-style="italic" font-family="monospace">w</text>
    <line x1="150" y1="150" x2="150" y2="66" stroke="#f5a623" stroke-width="1.5" stroke-dasharray="3,3"/>
    <text x="156" y="112" fill="#f5a623" font-size="13" font-style="italic" font-family="monospace">h</text>
  </svg>
  </div>


ellipse (Elliptical)
~~~~~~~~~~~~~~~~~~~~

**Parameters** : ``aperture_value = [a, b]`` , where :math:`a` is the semi-major axis (x direction) and :math:`b` is the semi-minor axis (y direction).

**Loss condition** :

.. math::

  \left(\frac{x}{a}\right)^2 + \left(\frac{y}{b}\right)^2 > 1

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <ellipse cx="150" cy="150" rx="120" ry="84" stroke="#00d2ff" stroke-width="2" fill="#00d2ff22"/>
    <line x1="150" y1="150" x2="270" y2="150" stroke="#f5a623" stroke-width="1.5" stroke-dasharray="3,3"/>
    <text x="200" y="142" fill="#f5a623" font-size="13" font-style="italic" font-family="monospace">a</text>
    <line x1="150" y1="150" x2="150" y2="66" stroke="#f5a623" stroke-width="1.5" stroke-dasharray="3,3"/>
    <text x="156" y="112" fill="#f5a623" font-size="13" font-style="italic" font-family="monospace">b</text>
  </svg>
  </div>


rectcircle (Rectangle Inscribed Circle)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Parameters** : ``aperture_value = [w, h, r]`` , where :math:`w` is the rectangle half-width, :math:`h` is the rectangle half-height, and :math:`r` is the circle radius.

The aperture region is the **intersection** of the rectangle and the circle (particles must be inside both the rectangle and the circle to survive).

**Loss condition** (particle is lost if either condition is met):

.. math::

  |x| > w \quad \text{or} \quad |y| > h

.. math::

  x^2 + y^2 > r^2

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <rect x="90" y="60" width="120" height="180" stroke="#e94560" stroke-width="1.5" fill="none" stroke-dasharray="5,3"/>
    <circle cx="150" cy="150" r="78" stroke="#00d2ff" stroke-width="2" fill="#00d2ff22"/>
    <line x1="150" y1="150" x2="210" y2="150" stroke="#e94560" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="172" y="145" fill="#e94560" font-size="11" font-style="italic" font-family="monospace">w</text>
    <line x1="150" y1="150" x2="150" y2="60" stroke="#e94560" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="135" y="108" fill="#e94560" font-size="11" font-style="italic" font-family="monospace">h</text>
    <line x1="150" y1="150" x2="205" y2="107" stroke="#f5a623" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="165" y="128" fill="#f5a623" font-size="11" font-style="italic" font-family="monospace">r</text>
  </svg>
  </div>

.. note::

  The red dashed line is the rectangle boundary, and the blue solid line is the circle boundary. The aperture region is the intersection of the two.


rectellipse (Rectangle Inscribed Ellipse)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Parameters** : ``aperture_value = [w, h, a, b]`` , where :math:`w` is the rectangle half-width, :math:`h` is the rectangle half-height, :math:`a` is the ellipse semi-major axis (x direction), and :math:`b` is the ellipse semi-minor axis (y direction).

The aperture region is the **intersection** of the rectangle and the ellipse (particles must be inside both the rectangle and the ellipse to survive).

**Loss condition** (particle is lost if either condition is met):

.. math::

  |x| > w \quad \text{or} \quad |y| > h

.. math::

  \left(\frac{x}{a}\right)^2 + \left(\frac{y}{b}\right)^2 > 1

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <rect x="90" y="60" width="120" height="180" stroke="#e94560" stroke-width="1.5" fill="none" stroke-dasharray="5,3"/>
    <ellipse cx="150" cy="150" rx="90" ry="78" stroke="#00d2ff" stroke-width="2" fill="#00d2ff22"/>
    <line x1="150" y1="150" x2="210" y2="150" stroke="#e94560" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="172" y="145" fill="#e94560" font-size="11" font-style="italic" font-family="monospace">w</text>
    <line x1="150" y1="150" x2="150" y2="60" stroke="#e94560" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="135" y="108" fill="#e94560" font-size="11" font-style="italic" font-family="monospace">h</text>
    <line x1="150" y1="155" x2="240" y2="155" stroke="#f5a623" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="188" y="170" fill="#f5a623" font-size="11" font-style="italic" font-family="monospace">a</text>
    <line x1="155" y1="150" x2="155" y2="72" stroke="#f5a623" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="160" y="90" fill="#f5a623" font-size="11" font-style="italic" font-family="monospace">b</text>
  </svg>
  </div>

.. note::

  The red dashed line is the rectangle boundary, and the blue solid line is the ellipse boundary. The aperture region is the intersection of the two.


racetrack (Racetrack)
~~~~~~~~~~~~~~~~~~~~~

**Parameters** : ``aperture_value = [w, h, a, b]`` , where :math:`w` is the rectangle half-width, :math:`h` is the rectangle half-height, :math:`a` is the x-direction semi-axis of the elliptical ends, and :math:`b` is the y-direction semi-axis of the elliptical ends.

The racetrack aperture consists of a central rectangle and two semi-elliptical ends. The centers of the elliptical ends are located at :math:`(\pm w, 0)` .

**Survival condition** (particle survives if either condition is met):

Inside the rectangular region:

.. math::

  |x| \le w \quad \text{and} \quad |y| \le h

Inside the elliptical end region (when :math:`|x| > w` ):

.. math::

  \left(\frac{|x| - w}{a}\right)^2 + \left(\frac{y}{b}\right)^2 \le 1

**Loss condition** : Not inside the rectangle and not inside the elliptical end.

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <path d="M 102 60 L 198 60 A 60 72 0 0 1 198 240 L 102 240 A 60 72 0 0 1 102 60 Z" stroke="#00d2ff" stroke-width="2" fill="#00d2ff22"/>
    <circle cx="198" cy="150" r="3" fill="#f5a623"/>
    <circle cx="102" cy="150" r="3" fill="#f5a623"/>
    <line x1="150" y1="150" x2="198" y2="150" stroke="#f5a623" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="168" y="145" fill="#f5a623" font-size="11" font-style="italic" font-family="monospace">w</text>
    <line x1="150" y1="150" x2="150" y2="60" stroke="#f5a623" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="135" y="108" fill="#f5a623" font-size="11" font-style="italic" font-family="monospace">h</text>
    <line x1="198" y1="150" x2="258" y2="150" stroke="#f5a623" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="220" y="165" fill="#f5a623" font-size="11" font-style="italic" font-family="monospace">a</text>
    <line x1="198" y1="150" x2="198" y2="78" stroke="#f5a623" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="203" y="118" fill="#f5a623" font-size="11" font-style="italic" font-family="monospace">b</text>
  </svg>
  </div>

.. note::

  The orange dots mark the centers of the elliptical ends at :math:`(\pm w, 0)` .


octagon (Octagonal)
~~~~~~~~~~~~~~~~~~~

**Parameters** : ``aperture_value = [w, h, d]`` , where :math:`w` is the half-width, :math:`h` is the half-height, and :math:`d` is the half-diagonal clearance (chamfer distance).

The octagon is the shape obtained by cutting 45° corners off a rectangle. The larger :math:`d` , the larger the chamfer; when :math:`d = 0` , it degenerates into a rectangle.

**Loss condition** (particle is lost if either condition is met):

.. math::

  |x| > w \quad \text{or} \quad |y| > h

.. math::

  |x| + |y| > w + h - d

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <polygon points="60,60 240,60 270,90 270,210 240,240 60,240 30,210 30,90" stroke="#00d2ff" stroke-width="2" fill="#00d2ff22"/>
    <line x1="150" y1="150" x2="270" y2="150" stroke="#f5a623" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="200" y="145" fill="#f5a623" font-size="11" font-style="italic" font-family="monospace">w</text>
    <line x1="150" y1="150" x2="150" y2="60" stroke="#f5a623" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="156" y="108" fill="#f5a623" font-size="11" font-style="italic" font-family="monospace">h</text>
    <line x1="240" y1="60" x2="240" y2="90" stroke="#e94560" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="244" y="80" fill="#e94560" font-size="11" font-style="italic" font-family="monospace">d</text>
  </svg>
  </div>


polygon (Polygon)
~~~~~~~~~~~~~~~~~

**Parameters** : ``aperture_value = [[x1, y1], [x2, y2], ...]`` , a list of vertices, automatically closed (the last vertex connects back to the first vertex).

**Loss condition** : The point is outside the polygon.

The **ray casting** method is used to determine whether a point is inside the polygon: a ray is cast from the test point in the horizontal direction, and the number of intersections with the polygon edges is counted:

- Odd number of intersections → the point is inside the polygon (survives)
- Even number of intersections → the point is outside the polygon (lost)

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <polygon points="270,150 210,46 90,46 30,150 90,254 210,254" stroke="#00d2ff" stroke-width="2" fill="#00d2ff22"/>
    <circle cx="270" cy="150" r="4" fill="#f5a623"/>
    <circle cx="210" cy="46" r="4" fill="#f5a623"/>
    <circle cx="90" cy="46" r="4" fill="#f5a623"/>
    <circle cx="30" cy="150" r="4" fill="#f5a623"/>
    <circle cx="90" cy="254" r="4" fill="#f5a623"/>
    <circle cx="210" cy="254" r="4" fill="#f5a623"/>
    <text x="248" y="170" fill="#f5a623" font-size="10" font-family="monospace">P1</text>
    <text x="215" y="40" fill="#f5a623" font-size="10" font-family="monospace">P2</text>
    <text x="72" y="40" fill="#f5a623" font-size="10" font-family="monospace">P3</text>
    <text x="10" y="170" fill="#f5a623" font-size="10" font-family="monospace">P4</text>
    <text x="72" y="270" fill="#f5a623" font-size="10" font-family="monospace">P5</text>
    <text x="215" y="270" fill="#f5a623" font-size="10" font-family="monospace">P6</text>
  </svg>
  </div>


Parameter Summary Table
------------------------

.. list-table::
  :header-rows: 1
  :widths: 15 25 60

  * - Type
    - aperture_value
    - Description
  * - ``off``
    - ignored
    - No aperture checking
  * - ``default``
    - ignored
    - Default ±1m rectangular aperture
  * - ``circle``
    - ``[r]``
    - Circular, :math:`r` is the radius
  * - ``rectangle``
    - ``[w, h]``
    - Rectangular, :math:`w` is the half-width, :math:`h` is the half-height
  * - ``ellipse``
    - ``[a, b]``
    - Elliptical, :math:`a` is the semi-major axis, :math:`b` is the semi-minor axis
  * - ``rectcircle``
    - ``[w, h, r]``
    - Intersection of rectangle and circle
  * - ``rectellipse``
    - ``[w, h, a, b]``
    - Intersection of rectangle and ellipse
  * - ``racetrack``
    - ``[w, h, a, b]``
    - Racetrack (rectangle + elliptical ends)
  * - ``octagon``
    - ``[w, h, d]``
    - Octagonal (rectangle with 45° chamfers)
  * - ``polygon``
    - ``[[x1,y1], ...]``
    - Polygon vertex list, automatically closed


Usage Example
-------------

The following JSON snippets show the configuration of each aperture type. Aperture parameters are element properties,
placed alongside fields such as ``S (m)`` , ``Command`` , ``Length (m)`` , etc.:

**Circular aperture** :

.. code-block:: json

  "Drift1": {
      "S (m)": 10.0,
      "Command": "Drift",
      "Length (m)": 0.5,
      "Aperture Type": "circle",
      "Aperture Value": [0.1]
  }

**Rectangular aperture** :

.. code-block:: json

  "Drift2": {
      "S (m)": 10.5,
      "Command": "Drift",
      "Length (m)": 0.3,
      "Aperture Type": "rectangle",
      "Aperture Value": [0.06, 0.04]
  }

**Elliptical aperture** :

.. code-block:: json

  "Drift3": {
      "S (m)": 11.0,
      "Command": "Drift",
      "Length (m)": 0.2,
      "Aperture Type": "ellipse",
      "Aperture Value": [0.06, 0.04]
  }

**Racetrack aperture** :

.. code-block:: json

  "Drift4": {
      "S (m)": 11.5,
      "Command": "Drift",
      "Length (m)": 0.4,
      "Aperture Type": "racetrack",
      "Aperture Value": [0.03, 0.05, 0.02, 0.05]
  }

**Octagonal aperture** :

.. code-block:: json

  "Drift5": {
      "S (m)": 12.0,
      "Command": "Drift",
      "Length (m)": 0.3,
      "Aperture Type": "octagon",
      "Aperture Value": [0.05, 0.03, 0.01]
  }

**Polygon aperture** :

.. code-block:: json

  "Drift6": {
      "S (m)": 12.5,
      "Command": "Drift",
      "Length (m)": 0.2,
      "Aperture Type": "polygon",
      "Aperture Value": [[0.05, 0.0], [0.025, 0.043], [-0.025, 0.043], [-0.05, 0.0], [-0.025, -0.043], [0.025, -0.043]]
  }

**Disable aperture checking** :

.. code-block:: json

  "Drift7": {
      "S (m)": 13.0,
      "Command": "Drift",
      "Length (m)": 0.5,
      "Aperture Type": "off"
  }
