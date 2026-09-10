Transverse Field Solvers
========================

Introduction
------------

``PASS.commands.solver`` provides the numerical layer used by ``SpaceCharge``.
It is independent of ``Simulation`` and PASS particle classes: PIC callers supply
transverse particle arrays, slice IDs, a uniform grid, and charge per
macroparticle; analytic fields use coordinates, charge, and profile parameters.  The same API can therefore be used by commands, tests, and
standalone field studies.

- **Code location**: ``PASS/commands/solver/``
- **PIC entry point**: ``PASS/commands/solver/pic.py``
- **Low-level PIC solver identifiers**: ``fd``, ``dst_rectangle``, ``fft_free_space``
- **Analytic tracking entry point**: ``PASS/commands/solver/analytic.py``
- **Array order**: ``(slice, y, x)`` for batched grid data
- **Execution backend**: NumPy/SciPy on CPU; CuPy and CUDA libraries on GPU

The numerical package solves only the transverse field problem.  It does not
know the interaction length, bunch rigidity, relativistic kick factor, or
simulation turn.  Those responsibilities belong to :doc:`space_charge`.

Public configuration uses ``Method`` and ``Solver``. For ``Method="pic"``,
``fd_dirichlet``, ``dst_dirichlet`` and ``fft_free_space`` dispatch to the
low-level identifiers ``fd``, ``dst_rectangle`` and ``fft_free_space`` respectively.
The identifiers in the numerical API sections below are arguments to
``build_pic_resources``. ``fft_free_space`` is shared with the public JSON
``Solver`` value; the FD and DST identifiers differ between these interfaces.
``frozen`` and ``quasi-frozen`` select the analytic profiles described below.

PIC Data Flow
-------------

One call follows this sequence:

.. code-block:: text

   particle x, y, tag and slice_id
                |
                v
     CIC or TSC charge deposition
                |
                v
      Sigma[slice, y, x] in C/m^2
                |
                v
       batched Poisson field solve
                |
                v
      Psi in V m, integrated Ex/Ey in V
                |
                v
     matching CIC or TSC field gather

All longitudinal slices are stored as leading right-hand sides and are solved
together.  Geometry-dependent masks, sparse factorizations, spectral
eigenvalues, or FFT kernels are built once in ``PICResources`` and reused.

Grid and Source Definition
--------------------------

Uniform nodal grid
~~~~~~~~~~~~~~~~~~

``GridGeometry`` describes a uniform node-centered rectangle.  For the
top-level space-charge input, full widths :math:`W_x` and :math:`W_y` define

.. math::

   x_{\min}=-\frac{W_x}{2},\quad x_{\max}=\frac{W_x}{2},
   \qquad
   \Delta x=\frac{W_x}{N_x-1},

.. math::

   y_{\min}=-\frac{W_y}{2},\quad y_{\max}=\frac{W_y}{2},
   \qquad
   \Delta y=\frac{W_y}{N_y-1}.

``Nx`` and ``Ny`` are node counts, not cell counts, and must each be at least
3. Both configuration inputs and ``build_grid_geometry`` accept a complete
full-width pair or a complete ``Grid Half Width X/Y (m)`` pair. Half widths
:math:`H_x,H_y` give :math:`W_x=2H_x,W_y=2H_y`. Mixing pairs, spacing inputs
``Dx``/``Dy``, and explicit-bound mapping inputs are rejected. Direct
``GridGeometry(...)`` construction remains available for numerical code.

Charge deposition
~~~~~~~~~~~~~~~~~

For each live particle with a valid slice ID, the deposition method distributes
the signed charge onto active field nodes:

.. list-table::
   :widths: 18 20 25 37
   :header-rows: 1

   * - Method
     - Nodes per particle
     - Matching gather
     - Characteristics
   * - ``CIC``
     - 2 x 2
     - bilinear
     - Piecewise-linear weights from the particle's containing grid cell.
   * - ``TSC``
     - 3 x 3
     - quadratic
     - Wider quadratic stencil with smoother particle-grid coupling.

Conductor and boundary nodes are removed from a deposition stencil.  Remaining
weights are normalized per particle so the retained stencil conserves that
particle's charge.  The same active-node normalization is used during gather.
A warning is emitted when a boundary changes the stencil.  If an in-domain
particle has no active node, it is ignored by that PIC call and gathers zero
field; its PASS particle tag is not changed.

Particles with ``tag <= 0``, invalid slice IDs, or coordinates outside the grid
or physical aperture are also ignored.  Grid and aperture dimensions should
therefore cover the intended tracked distribution.
These standalone PIC functions do not modify particle tags. ``SpaceCharge``
first applies its command aperture using the shared particle-loss function;
wall and outside particles are lost before deposition. Surviving participating
particles outside the grid, or without active stencil nodes, raise an error.
See :doc:`space_charge` for initialization checks and the input contract.

Poisson Equation and Units
--------------------------

Each solver consumes the deposited surface density :math:`\Sigma_k` in
C/m\ :sup:`2` and solves

.. math::

   -\nabla_\perp^2\Psi_k=\frac{\Sigma_k}{\epsilon_0},
   \qquad
   \mathcal E_{x,k}=-\frac{\partial\Psi_k}{\partial x},
   \qquad
   \mathcal E_{y,k}=-\frac{\partial\Psi_k}{\partial y}.

Because the slice charge has already been integrated longitudinally,
:math:`\Psi` has units V m and :math:`\mathcal E_x,\mathcal E_y` have
units V.  These are integrated fields, not V/m average fields.
``SpaceCharge`` divides a gathered field by that slice's ``delta_z`` before
calculating the kick.

Field-Solver Selection
----------------------

.. list-table::
   :widths: 18 22 25 35
   :header-rows: 1

   * - Low-level ``field_solver``
     - Boundary model
     - Allowed aperture
     - Numerical method and intended use
   * - ``fd``
     - Zero Dirichlet conductor
     - Full grid rectangle or any supported continuous aperture
     - Cached sparse LU solve.  Uses the regular five-point stencil on a full
       rectangle and Shortley--Weller distances near a curved or oblique wall.
   * - ``dst_rectangle``
     - Zero Dirichlet conductor
     - Complete grid-aligned rectangle only
     - Direct type-I discrete sine transform using cached eigenvalues.  It is
       a specialized rectangular-chamber alternative to ``fd``.
   * - ``fft_free_space``
     - Open free space
     - Complete grid only; no conducting aperture
     - Zero-padded Hockney-style convolution with cached Green-function
       kernels.  Use when image charges from a conducting chamber are not
       wanted.

Finite difference: ``fd``
~~~~~~~~~~~~~~~~~~~~~~~~~

For a full rectangular domain, the regular five-point discretization is

.. math::

   \left(\frac{2}{\Delta x^2}+\frac{2}{\Delta y^2}\right)\Psi_{i,j}
   -\frac{\Psi_{i-1,j}+\Psi_{i+1,j}}{\Delta x^2}
   -\frac{\Psi_{i,j-1}+\Psi_{i,j+1}}{\Delta y^2}
   =\frac{\Sigma_{i,j}}{\epsilon_0}.

The outer grid nodes are held at :math:`\Psi=0`. An explicit aperture equal
to the full grid uses the same rectangular solver. Other continuous apertures
use the Shortley--Weller solver.  Where a neighboring
node lies outside the aperture, the regular spacing is replaced by the actual
distance from the active node to the grid-line/wall intersection.  The
physical wall is therefore not approximated merely by the visible stair-step
node mask.

The sparse matrix and LU factorization are constructed once.  Every slice is
passed to the same factorization as one dense multi-column right-hand side.

Sine transform: ``dst_rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``dst_rectangle`` imposes zero potential on all four outer grid edges.  A
type-I discrete sine transform diagonalizes the same rectangular
finite-difference operator.  For horizontal mode :math:`m` and vertical mode
:math:`n`, its eigenvalue is

.. math::

   \lambda_{m,n}
   =\frac{4}{\Delta x^2}\sin^2\!\left(
      \frac{\pi m}{2(N_x-1)}\right)
    +\frac{4}{\Delta y^2}\sin^2\!\left(
      \frac{\pi n}{2(N_y-1)}\right).

The solver transforms only the two transverse axes, preserving the leading
slice axis.  It cannot represent a curved or smaller internal conductor.

Free-space Green function: ``fft_free_space``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``fft_free_space`` performs a zero-padded linear convolution, avoiding the periodic
wrap-around of an unpadded FFT.  Away from the self cell, the kernels are

.. math::

   G_\Psi(\mathbf r)
   =-\frac{1}{2\pi\epsilon_0}\ln\!\left(\frac{r}{r_0}\right),
   \qquad r_0=\sqrt{\Delta x\Delta y},

.. math::

   G_x(\mathbf r)=\frac{x}{2\pi\epsilon_0 r^2},
   \qquad
   G_y(\mathbf r)=\frac{y}{2\pi\epsilon_0 r^2}.

The self-cell entries are set to zero.  The potential therefore uses a kernel
reference and is meaningful only up to an additive constant; the transverse
fields are the physical outputs.  This solver models open free space, not a
grounded beam pipe.

The source uses one forward real FFT. Each requested output uses one inverse
FFT: two for the electric fields and a third for the potential. By default,
``FFTFreeSpaceSolver.solve`` returns all three outputs. With
``compute_potential=False``, it returns ``potential=None`` and skips the
potential transform. The potential kernel is built lazily on its first use.
The inverse transforms share a scratch spectrum; returned arrays own only
the physical grid, releasing the larger padded arrays.

The same optional keyword is available in ``solve_pic``, ``pic_cpu``, and
``solve_poisson_fft_free_space``. FD and DST still calculate potential because
their fields require its gradient. The ``SpaceCharge`` command requests FFT
potential only on selected turns when ``Save potential`` is enabled.

Field gradients
~~~~~~~~~~~~~~~

``fd`` on a full rectangle and ``dst_rectangle`` obtain fields with
``E = -grad(Psi)`` using grid finite differences.  Shortley--Weller ``fd``
uses unequal-distance derivative coefficients at active nodes near the wall.
``fft_free_space`` convolves directly with the analytic field kernels.

Aperture Interface
------------------

The SC command owns both loss geometry and, for FD/DST, the conducting wall.
The configuration has no ``Chamber`` field. For example, a command contains:

.. code-block:: json

   "Aperture type": "ellipse",
   "Aperture value": [0.04, 0.02]

Internally, FD receives ``{"Type": "ellipse", "Value": [0.04, 0.02]}`` as its
``aperture`` argument. FFT receives no conducting aperture; its command aperture
is applied only to particle losses. The following table describes the shared
low-level geometry builder, with all dimensions in metres. Its generic
``default`` differs from SC: SC resolves default to the actual grid rectangle
before calling the builder and rejects ``off`` for Dirichlet solvers.

.. list-table::
   :widths: 18 27 55
   :header-rows: 1

   * - Type
     - ``Aperture Value``
     - Geometry
   * - ``off``
     - omitted
     - No separate physical aperture.  For ``fd``, the outer grid still acts
       as the zero-potential boundary.
   * - ``default``
     - omitted
     - Default tracking rectangle :math:`|x|\leq1`, :math:`|y|\leq1`.
   * - ``circle``
     - ``[R]``
     - Circle of radius :math:`R`.
   * - ``rectangle``
     - ``[A, B]``
     - Rectangle :math:`|x|\leq A`, :math:`|y|\leq B`.
   * - ``ellipse``
     - ``[A, B]``
     - Ellipse :math:`x^2/A^2+y^2/B^2\leq1`.
   * - ``rectcircle``
     - ``[W, H, R]``
     - Intersection of the rectangle :math:`(W,H)` and circle :math:`R`.
   * - ``rectellipse``
     - ``[W, H, A, B]``
     - Intersection of the rectangle :math:`(W,H)` and ellipse
       :math:`(A,B)`.
   * - ``racetrack``
     - ``[W, H, A, B]``
     - Central half-width/half-height :math:`(W,H)` with horizontal elliptic
       end caps of semi-axes :math:`(A,B)`.
   * - ``octagon``
     - ``[W, H, D]``
     - Symmetric octagon satisfying :math:`|x|\leq W`,
       :math:`|y|\leq H`, and :math:`|x|+|y|\leq W+H-D`.
   * - ``polygon``
     - ``[[x1,y1], ...]``
     - Polygon with at least three finite vertices and nonzero area.

``circular``, ``elliptic`` and ``rectangular`` are accepted lower-level
aliases.  Named parameters are also supported by the Python aperture builder,
but generated input files should use the table above.

For ``dst_rectangle`` and ``fft_free_space``, the aperture must resolve exactly to
the full grid-aligned rectangle; ``null`` is the normal input.  For ``fd``, a
physical aperture should be resolved within the selected grid. The high-level
SC initializer rejects any finite PIC aperture extending beyond the grid,
and rejects a DST aperture different from the full rectangle. Thus an oversized
command aperture never silently becomes a truncated conductor.

Python Interfaces
-----------------

Geometry and PIC pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 26 27 47
   :header-rows: 1

   * - Interface
     - Main arguments
     - Result and behavior
   * - ``GridGeometry(...)``
     - ``nx, ny, x_min, x_max, y_min, y_max``
     - Immutable uniform-grid description with ``dx``, ``dy``, ``x`` and ``y``
       properties.
   * - ``build_grid_geometry(config, **kwargs)``
     - node counts plus one complete full-width or half-width pair
     - Builds a ``GridGeometry`` centered at zero.
   * - ``build_aperture_mask(geometry, aperture)``
     - grid and continuous aperture mapping
     - Returns the boolean nodal membership mask.
   * - ``build_pic_resources(...)``
     - ``geometry``, optional ``aperture``, ``field_solver``
     - Returns reusable ``PICResources`` containing geometry, aperture,
       active mask, and a cached solver.
   * - ``deposit_particles(...)``
     - particles, ``slice_id``, geometry, resources, method, charge
     - Dispatches to CIC or TSC and returns ``DepositResult``.
   * - ``solve_pic(...)``
     - particles, ``slice_id``, geometry, resources, method,
       ``charge_per_macro``, ``num_slices``, ``compute_potential=True``
     - Deposits and solves all slices, returning ``PICResult``.
   * - ``gather_bilinear(...)``
     - field, particles, geometry, resources, ``slice_id``
     - Gathers one or many fields with CIC weights.
   * - ``gather_quadratic(...)``
     - field, particles, geometry, resources, ``slice_id``
     - Gathers one or many fields with TSC weights.
   * - ``pic_cpu(...)``
     - arrays ``x``, ``y``, ``slice_id``, charge, geometry or mesh,
       ``compute_potential=True``
     - Convenience array API around ``solve_pic``.  ``delta_z`` is accepted
       only to infer the number of slices and does not rescale the field.

``particles`` may be a mapping or an object exposing equal-shaped ``x`` and
``y`` arrays and, optionally, ``tag``.  ``charge_per_macro`` may be a finite
scalar or an array broadcastable to the particle shape.  ``slice_id`` must be
an integer array with one entry per particle.  Supplying ``num_slices`` keeps
trailing empty slices in the output.

Solver builders and one-shot wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 34 30 36
   :header-rows: 1

   * - Builder
     - Reusable solver
     - One-shot wrapper
   * - ``build_fd_resources(geometry)``
     - ``FDSolver``
     - ``solve_poisson_fd(...)``
   * - ``build_fd_arbitrary_resources(geometry, aperture)``
     - ``ArbitraryFDSolver``
     - ``solve_poisson_fd_arbitrary(...)``
   * - ``build_dst_rectangle_resources(geometry)``
     - ``DSTRectangleSolver``
     - call ``solver.solve(density)``
   * - ``build_fft_free_space_resources(geometry)``
     - ``FFTFreeSpaceSolver``
     - ``solve_poisson_fft_free_space(...)``

For repeated calculations, prefer a builder plus ``solver.solve`` so cached
resources are reused.

Result Objects
--------------

.. list-table::
   :widths: 22 22 18 38
   :header-rows: 1

   * - Object / field
     - Shape
     - Unit
     - Description
   * - ``DepositResult.density``
     - ``(n_slice, ny, nx)``
     - C/m\ :sup:`2`
     - Deposited charge density.
   * - ``DepositResult.deposited_charge``
     - ``(n_slice,)``
     - C
     - Charge retained in each slice.
   * - ``DepositResult.deposited_count``
     - ``(n_slice,)``
     - -
     - Number of deposited macroparticles per slice.
   * - ``FieldResult.potential``
     - 2-D, 3-D, or ``None``
     - V m
     - Integrated potential; ``None`` when an FFT field-only solve omits it.
   * - ``FieldResult.integrated_ex``, ``integrated_ey``
     - 2-D or 3-D
     - V
     - Integrated transverse fields; ``ex`` and ``ey`` are aliases.
   * - ``PICResult``
     - batched
     - mixed
     - Combines density, potential, fields, geometry, deposited charge, and
       deposition diagnostics.

A field solver accepts either ``(ny, nx)`` for one slice or
``(n_slice, ny, nx)`` for a batch.  All values must be finite and the trailing
dimensions must match the solver geometry.

Analytic Tracking and Reference Fields
------------------------------------------

The ``formula_*`` modules provide free-space analytic integrated fields. They
are used by the ``frozen`` and ``quasi-frozen`` tracking methods and remain
available for reference calculations. Their public solver names are
``gaussian_round_free_space``, ``gaussian_ellipse_free_space``,
``uniform_round_free_space`` and ``uniform_ellipse_free_space``. These formulas
are evaluated directly at particle positions, outside the PIC pipeline.

Source charge, coordinates, and units
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All four profiles solve the transverse free-space problem for one charge slice.
Let Q denote its signed total charge in C, and let (u, v) be coordinates in the
source's principal frame. The command obtains Q from the current live,
assigned population after aperture losses, then translates and rotates:

.. math::

   Q_k=N_k R Z e,\qquad
   \begin{pmatrix}u\\v\end{pmatrix}
   =\begin{pmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{pmatrix}
   \begin{pmatrix}x-c_x\\y-c_y\end{pmatrix},\qquad
   \begin{pmatrix}\mathcal E_x\\\mathcal E_y\end{pmatrix}
   =\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}
   \begin{pmatrix}\mathcal E_u\\\mathcal E_v\end{pmatrix}.

Here R is ``bunch.ratio``, Z the signed charge number, and N_k the slice's
live macroparticle count. The densities below are longitudinally integrated
surface densities in C/m\ :sup:`2`; their integral over the entire transverse
plane is Q. The fields are integrated fields in V. ``SpaceCharge`` divides
them by ``delta_z`` to obtain V/m and applies the relativistic kick separately.
The formula functions themselves apply neither ``delta_z`` nor 1/gamma².

The source center and profile sizes describe the beam, not the vacuum chamber.
These free-space formulas contain no conducting-wall image fields. A command
aperture handles particle losses only; its default is the grid rectangle.
Diagnostic sampling does not clip or renormalize the analytic model. Even when
losses occur, a frozen Gaussian retains its specified full Gaussian shape with
the updated Q; it is not an exact field of a Gaussian truncated by the aperture.

Round Gaussian: gaussian_round_free_space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``formula_gaussian_round.gaussian_round_field`` uses the single-axis RMS
size sigma (``Sigma (m)`` in a frozen configuration):

.. math::

   r^2=u^2+v^2,\qquad
   \Sigma(u,v)=\frac{Q}{2\pi\sigma^2}\exp\!\left(-\frac{r^2}{2\sigma^2}\right),
   \qquad
   \begin{pmatrix}\mathcal E_u\\\mathcal E_v\end{pmatrix}
   =\frac{Q}{2\pi\epsilon_0}
   \frac{1-\exp[-r^2/(2\sigma^2)]}{r^2}
   \begin{pmatrix}u\\v\end{pmatrix}.

At the origin both components are zero. The scalar factor multiplying (u, v)
has the limit Q/(4 pi epsilon_0 sigma²), so the central force is linear.
The implementation uses ``-expm1(-r²/(2 sigma²))`` to avoid subtracting nearly
equal numbers. Far from the source the signed radial field tends to
Q/(2 pi epsilon_0 r), with its sign set by Q.

Elliptic Gaussian: gaussian_ellipse_free_space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``formula_gaussian_ellipse.gaussian_elliptic_field`` uses the principal
single-axis RMS widths sigma_u and sigma_v (frozen ``Sigma X/Y (m)``):

.. math::

   \Sigma(u,v)=\frac{Q}{2\pi\sigma_u\sigma_v}
   \exp\!\left(-\frac{u^2}{2\sigma_u^2}-\frac{v^2}{2\sigma_v^2}\right).

For sigma_u > sigma_v, PASS evaluates the Bassetti--Erskine formula in the
first quadrant using the Faddeeva function ``scipy.special.wofz``:

.. math::

   D=\sigma_u^2-\sigma_v^2,\quad
   U=|u|,\quad V=|v|,\quad
   z_1=\frac{U+iV}{\sqrt{2D}},\quad
   z_2=\frac{U\sigma_v/\sigma_u+iV\sigma_u/\sigma_v}{\sqrt{2D}},


.. math::

   g=\exp\!\left(-\frac{u^2}{2\sigma_u^2}-\frac{v^2}{2\sigma_v^2}\right),\qquad
   F=\frac{iQ}{2\epsilon_0\sqrt{2\pi D}}\,[w(z_1)-g\,w(z_2)],
   \qquad w(z)=e^{-z^2}\operatorname{erfc}(-iz),


.. math::

   \mathcal E_u=-\operatorname{sgn}(u)\operatorname{Re}F,\qquad
   \mathcal E_v=\operatorname{sgn}(v)\operatorname{Im}F.

When sigma_u < sigma_v, coordinates, widths and returned components are
exchanged. Equal widths reduce to the round Gaussian; numerically PASS uses
``np.isclose(sigma_u, sigma_v, rtol=const.eps, atol=0)``. At small amplitude the
linear terms are

.. math::

   \mathcal E_u\simeq\frac{Q u}{2\pi\epsilon_0\sigma_u(\sigma_u+\sigma_v)},\qquad
   \mathcal E_v\simeq\frac{Q v}{2\pi\epsilon_0\sigma_v(\sigma_u+\sigma_v)}.

The implementation includes cubic corrections near the center; numerical
stability details follow the Python interface table below.

Uniform disk: uniform_round_free_space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``formula_uniform_round.uniform_round_field`` uses the physical outer beam
radius R_b (``Radius (m)``), not an RMS size:

.. math::

   \Sigma(u,v)=
   \begin{cases}Q/(\pi R_b^2),&r\le R_b,\\0,&r>R_b,\end{cases}
   \qquad
   \begin{pmatrix}\mathcal E_u\\\mathcal E_v\end{pmatrix}
   =\frac{Q}{2\pi\epsilon_0}
   \begin{cases}
   R_b^{-2}\begin{pmatrix}u\\v\end{pmatrix},&r\le R_b,\\
   r^{-2}\begin{pmatrix}u\\v\end{pmatrix},&r>R_b.
   \end{cases}

The interior field is linear, the exterior field decreases as 1/r, and the
field is continuous at r=R_b. The per-axis RMS size of this uniform disk is
R_b/2. A source edge is distinct from a command's physical aperture wall;
evaluating the formula on the source edge does not itself mark a particle lost.

Uniform ellipse: uniform_ellipse_free_space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``formula_uniform_ellipse.uniform_elliptic_field`` uses beam semi-axes a and b
(``Semi-axis A/B (m)``), with per-axis RMS sizes a/2 and b/2:

.. math::

   \eta=\frac{u^2}{a^2}+\frac{v^2}{b^2},\qquad
   \Sigma(u,v)=\begin{cases}Q/(\pi ab),&\eta\le1,\\0,&\eta>1.\end{cases}


.. math::

   \lambda=0\quad(\eta\le1),\qquad
   \frac{u^2}{a^2+\lambda}+\frac{v^2}{b^2+\lambda}=1,\quad\lambda>0\quad(\eta>1),


.. math::

   A=\sqrt{a^2+\lambda},\quad B=\sqrt{b^2+\lambda},\qquad
   \mathcal E_u=\frac{Q u}{\pi\epsilon_0 A(A+B)},\qquad
   \mathcal E_v=\frac{Q v}{\pi\epsilon_0 B(A+B)}.

Inside the source A=a and B=b, giving linear fields. Outside, lambda is the
positive root of the confocal-ellipse equation. The implementation computes
that root with a cancellation-resistant quadratic expression and uses the
rationalized field above to remain stable near a=b. Fields are continuous
across the source edge; a=b reduces to the uniform disk.

Frozen and quasi-frozen parameter selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``frozen`` uses one fixed center, orientation and set of profile sizes for
all slices referencing a configuration. Center and angle default to zero;
solver-specific sizes are required. Q and ``delta_z`` remain current, so
freezing the transverse profile does not freeze its field amplitude.

``quasi-frozen`` recomputes every slice's population moments at each kick:

.. math::

   \mathbf c_k=\frac{1}{N_k}\sum_{n\in k}\mathbf r_n,\qquad
   C_k=\frac{1}{N_k}\sum_{n\in k}(\mathbf r_n-\mathbf c_k)
   (\mathbf r_n-\mathbf c_k)^{\mathsf T}.


.. math::

   \sigma_u=\sqrt{\nu_1},\quad\sigma_v=\sqrt{\nu_2}
   \quad\text{(Gaussian ellipse)},\qquad
   a=2\sqrt{\nu_1},\quad b=2\sqrt{\nu_2}\quad\text{(uniform ellipse)},


.. math::

   \sigma=\sqrt{\frac{\operatorname{tr}C_k}{2}},\qquad
   R_b=2\sigma\quad\text{(round profiles)}.

The eigenvalues satisfy nu_1 >= nu_2; the eigenvector for nu_1 determines the
major-axis angle. Moments use denominator N_k, not N_k-1. The round rule
preserves the radial second moment even if the particles are not round; it
does not reconstruct a non-round source's exact field. Uniform profiles
approximate a uniform projected density, rather than making the tracked
population a KV distribution.

Empty slices return zero field. Nonempty quasi-frozen round slices require
at least two particles and positive radial variance. Elliptic slices require
at least three particles and ``nu_2 > 64 * float64_epsilon * nu_1``.
Invalid moments raise an error. The supplied slice IDs and widths remain
under user control; no Slicer execution or turn-history check is added.

Direct formula example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This standalone example evaluates the integrated field and then converts it
to the slice-average field. Coordinates are already in the source frame:

.. code-block:: python

   import numpy as np
   from PASS.commands.solver.formula_gaussian_ellipse import gaussian_elliptic_field

   x = np.linspace(-0.02, 0.02, 201)  # m, relative to the source center
   ex_integrated, ey_integrated = gaussian_elliptic_field(
       x, np.zeros_like(x), slice_charge=1e-9,
       sigma_x=0.003, sigma_y=0.002,
   )
   delta_z = 0.01  # m
   ex_average = ex_integrated / delta_z  # V/m

For tracking, select the corresponding public ``Solver`` together with
``Method="frozen"`` or ``"quasi-frozen"``; see :doc:`space_charge` for JSON
examples and command aperture defaults. Validation cases in
``tests/integration/space_charge/test_analytic_free_space_tracking.py`` compare
actual particle kicks against independent field integrals. Run
``python -m tests.integration.space_charge analytic`` for these comparisons
and repeated-kick parameter-evolution checks, including generated plots.

Python interfaces and numerical stability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``solve_analytic(x, y, slice_id, valid, num_slices, charge_per_macro,
configuration)`` groups assigned live particles by slice. Frozen parameters
come from the configuration; quasi-frozen parameters come from each slice's
current population moments. ``AnalyticResult`` contains particle-sized
``integrated_ex``/``integrated_ey``, slice charges and counts, and a
``(n_slice, 5)`` parameter array with columns center-x, center-y, size-x, size-y,
angle. Sizes are Gaussian RMS widths or uniform semi-axes; empty-slice
parameters are NaN and their field/charge is zero. No simulation turn or Slicer
execution metadata is read. See :doc:`space_charge` for the exact moment rules.

``sample_analytic_grid(result, configuration, geometry)`` evaluates diagnostic
density and fields on a grid after particle fields have been obtained.
It returns a grid result with ``potential=None``; analytic potential output is
currently rejected by the command. The sampling grid does not determine
particle kicks or truncate the analytic charge distribution.

.. list-table::
   :widths: 32 34 34
   :header-rows: 1

   * - Function
     - Distribution parameters
     - Return value
   * - ``gaussian_round_field``
     - ``x, y, slice_charge, sigma``
     - Round-Gaussian integrated ``(Ex, Ey)`` in V.
   * - ``gaussian_elliptic_field``
     - ``x, y, slice_charge, sigma_x, sigma_y``
     - Bassetti--Erskine integrated field in V;
       ``gaussian_ellipse_field`` is an alias.
   * - ``uniform_round_field``
     - ``x, y, slice_charge, radius``
     - Uniform round-slice field inside and outside the beam.
   * - ``uniform_elliptic_field``
     - ``x, y, slice_charge, a, b``
     - Uniform elliptic-slice field inside and outside the beam.
   * - ``macro_charge_to_physical``
     - real-particle count, signed charge number
     - Signed physical charge in C.

All analytic functions accept scalar or broadcastable coordinate arrays,
require positive finite size parameters, and use ``epsilon_0`` from PASS
constants unless explicitly overridden.

The uniform-ellipse field uses confocal semi-axes
:math:`A=\sqrt{a^2+\lambda}`, :math:`B=\sqrt{b^2+\lambda}` and the equivalent
expressions :math:`\mathcal E_x=Qx/[\pi\epsilon_0 A(A+B)]`,
:math:`\mathcal E_y=Qy/[\pi\epsilon_0 B(A+B)]`. Here lambda is zero inside
the ellipse and the nonnegative confocal parameter outside. This rationalized
form avoids cancellation as the two source semi-axes approach equality,
including almost isotropic quasi-frozen slices.

The elliptic Gaussian formula is evaluated in the first quadrant and its
field signs are restored by reflection symmetry. This avoids subtracting
exponentially large Faddeeva-function values in the lower half-plane,
particularly for nearly round beams. The width difference is factored as
``(sigma_x - sigma_y) * (sigma_x + sigma_y)``; the existing round-beam limit
and axis-exchange convention are retained.
Near the beam center, where the stable-half-plane terms also nearly cancel,
a cubic field expansion is used when
``(x/sigma_x)**2 + (y/sigma_y)**2 <= 1e-6``. Its relative truncation error is
of order the square of this normalized squared radius.

Efficient Grid Sizes
--------------------

Here ``N`` counts nodes, including both endpoints: a width ``W`` has spacing
``h = W/(N-1)``. Choose the physical extent and required resolution first.
The following are convenient starting sizes near each nominal scale, not
hardware-independent timing optima; apply the rule separately to each axis.

.. list-table:: Node-count recommendations
   :header-rows: 1
   :widths: 16 24 20 20 20

   * - Nominal scale
     - FD baseline
     - DST nodes
     - FFT nodes
     - FFT padded size
   * - 128
     - About 128
     - 129
     - 128
     - 256
   * - 256
     - About 256
     - 257
     - 256
     - 512
   * - 512
     - About 512
     - 513
     - 512
     - 1024
   * - 1024
     - About 1024
     - 1025
     - 1024
     - 2048
   * - 2048
     - About 2048
     - 2049
     - 2048
     - 4096

For DST-I the interior length is ``N-2`` and the logical transform length is
``2*(N-1)``. Therefore ``N = 2**k + 1`` is a convenient family. More generally,
small prime factors in ``N-1`` are favorable; powers of two are not the only
fast lengths.

For FFT Green convolution each axis is padded to
``P = scipy.fft.next_fast_len(2*N-1)``. Choosing ``N = 2**k`` gives
``P = 2**(k+1)`` at the listed sizes. Nearby node counts can also be fast;
benchmark candidates at comparable resolution. Padding prevents circular
wrap-around and does not increase the physical grid extent.

FD uses sparse factorization and has no special power-of-two advantage.
Choose the smallest size satisfying geometry and convergence requirements,
for example ``N >= ceil(W/h_max) + 1``. An odd size can be useful to place a
node on the centerline. The FD column is only a resolution baseline: a
2048-by-2048 sparse factorization can require substantial memory. For a full
grounded rectangle, DST solves the same discrete Poisson system without
sparse LU factors. PASS keeps the explicitly configured node counts.

Selection Guidance and Limitations
----------------------------------

- Use ``fd`` for a grounded chamber, especially a curved, polygonal, or
  compound aperture.
- Use ``dst_rectangle`` for a grounded chamber exactly aligned with the full
  rectangular grid.
- Use ``fft_free_space`` for an open-boundary approximation without image charges.
- Increase the grid extent until an open-boundary field is insensitive to
  truncation, and increase resolution until field and kick observables
  converge.
- CIC is cheaper and more local; TSC gives smoother coupling but uses a wider
  stencil.  The deposition and gather methods must remain paired.
- CPU and GPU implement the same boundary models and units. Floating-point
  reductions and sparse factorizations need not be bitwise identical.

GPU resources and execution
---------------------------

Install ``python -m pip install --editable ".[cuda]"`` with a compatible CUDA
toolkit and driver. GPU imports are lazy; CPU use does not require CuPy.
The implementation uses CuPy device arrays and ``RawKernel`` for deposition,
gather and field processing. Host-side Python calls the cuDSS bindings and
cuFFT plans; these host library APIs are not called inside a CUDA kernel.

CPU and GPU implementations live in the same module for each function. CUDA
source is embedded in that module and compiled lazily; there are no separate
``gpu_*.py`` implementations or external ``.cu`` source files for these solvers.
The previous GPU module import paths have been removed.

.. list-table:: Source modules under ``PASS.commands.solver``
   :header-rows: 1
   :widths: 25 75

   * - Module
     - CPU and GPU entry points
   * - ``pic.py``
     - ``pic_cpu`` / ``pic_gpu``, resource builders, deposition and gather.
   * - ``fd_rectangle.py``
     - ``FDSolver`` / ``GPUFDSolver(geometry, dtype="float64")``.
   * - ``fd_arbitrary.py``
     - ``ArbitraryFDSolver`` / ``GPUArbitraryFDSolver(geometry, aperture, dtype="float64")``.
   * - ``dst_rectangle.py``
     - ``DSTRectangleSolver`` / ``GPUDSTRectangleSolver``, including cuFFTDx compilation.
   * - ``fft_free_space.py``
     - ``FFTFreeSpaceSolver`` / ``GPUFFTFreeSpaceSolver``.
   * - ``analytic.py``
     - ``solve_analytic`` / ``solve_analytic_gpu``.

``field_result.py`` provides the shared result type and GPU buffer/compilation
utilities. Both SpaceCharge execution paths live in ``PASS.commands.space_charge``;
internal-element scheduling lives in ``PASS.utils.slicing``. GPU libraries are
imported only inside GPU entry points, so co-location does not add a CUDA
requirement to CPU execution.

.. list-table:: GPU solver implementations
   :header-rows: 1
   :widths: 22 78

   * - Solver
     - Resident computation
   * - ``fd``
     - cuDSS factorization at initialization and batched dense right-hand-side
       solves during tracking. Full rectangles use SPD mode; Shortley--Weller
       matrices use general mode because unequal boundary distances can break symmetry.
   * - ``dst_rectangle``
     - General DST-I uses odd extension, real cuFFT transforms and fused
       packing, transpose and normalization kernels. Suitable power-of-two
       extensions also support a cuFFTDx implementation that fuses both
       transverse-y transforms with the eigenvalue division.
   * - ``fft_free_space``
     - Batched real cuFFT linear convolution with cached Green-function spectra,
       small-prime padding and bounded slice chunks. One source transform is
       reused for both field components and optional potential.

``build_pic_resources_gpu`` accepts the same geometry and low-level solver
names as ``build_pic_resources``, plus ``dtype``, ``num_slices``,
``dst_implementation``, ``fft_batch_size`` and ``deposition_strategy``.
The defaults are ``float64``, automatic DST selection, 16-slice FFT chunks and
direct atomic deposition. ``num_slices`` should be supplied at initialization.
For example, given device arrays ``x``, ``y`` and integer ``slice_id``:

.. code-block:: python

   from PASS.commands.solver import (
       GridGeometry, build_pic_resources_gpu, pic_gpu, gather_fields_gpu,
   )

   grid = GridGeometry(513, 513, -0.02, 0.02, -0.02, 0.02)
   resources = build_pic_resources_gpu(
       grid, field_solver="dst_rectangle", dtype=x.dtype, num_slices=100,
   )
   result = pic_gpu(x, y, slice_id, 1.0e-15, geometry=grid,
                    resources=resources, num_slices=100, method="CIC")
   ex, ey = gather_fields_gpu(result.ex, result.ey, {"x": x, "y": y},
                             grid, resources, slice_id, method="CIC")
   resources.close()

Arrays and reduction diagnostics remain on the GPU. Results own their grid
arrays by default. ``copy=False`` borrows grid buffers until the next call on
the same resources. ``validate=False`` skips finite-value checks for validated
inputs; metadata checks still apply. Supplying ``num_slices`` avoids a device
maximum read. Reuse requires the creation device and stream, serialized calls,
unchanged geometry, boundary operator and precision. ``close()`` releases
workspaces and cuDSS handles; further use raises an error. A different slice
count rebuilds the batch workspace, retaining the FD factorization.

Automatic DST selection first checks numerical agreement and then interleaves
CUDA-event timings of cuFFT and cuFFTDx. It selects cuFFTDx only when its median
is at least 5 percent lower. The decision is cached in the process by device,
precision, grid dimensions and slice count. Tuning and first-time NVRTC
compilation belong to initialization. Explicit choices are ``cufft``,
``cufftdx`` and ``fused`` (the ordinary CUDA radix-two implementation).
cuFFTDx headers come from ``nvidia-mathdx``. Automatic mode warns and retains
cuFFT if this optional implementation cannot compile; a numerical discrepancy
raises an error. Explicit cuFFTDx requests report dependency/compiler failures.

Node counts include boundary nodes: 513 nodes give 511 interior nodes and a
1024-point DST-I extension; 512 nodes give a 1022-point extension. Counts
``2**k + 1`` are useful candidates. The fused implementations currently accept
extension lengths 8 through 2048; all other legal sizes use cuFFT without
changing the configured grid. Benchmark on the actual target GPU and precision.

``deposition_strategy="warp"`` combines same-node contributions within a warp;
``sorted_warp`` first sorts temporary particle indices by slice and cell, then
uses the same aggregation. Neither changes the particle pool ordering. Sorting
cost must be included in comparisons: clustered input alone does not guarantee
a gain. Both CIC and TSC normalize retained wall stencils and use matching
gather weights. Geometry and aperture comparisons use double intermediates,
including with FP32 particles, to avoid rounding a near-wall point onto a
different node. Density, potential, fields and kicks retain the selected precision.

GPU analytic tracking is provided by ``analytic.solve_analytic_gpu`` for
all four round/elliptical Gaussian/uniform profiles in frozen and quasi-frozen
mode. Centered slice statistics and special-function intermediates use FP64;
particle fields follow the configured dtype. Diagnostic analytic-grid sampling
may use the CPU on selected output turns.
