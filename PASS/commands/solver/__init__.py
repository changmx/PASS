"""Numerical solvers used by collective-effect commands."""

from .pic import (
    DepositResult,
    GridGeometry,
    PICResources,
    PICResult,
    build_aperture_mask,
    build_grid_geometry,
    build_pic_resources,
    deposit_cic,
    deposit_particles,
    deposit_tsc,
    gather_bilinear,
    gather_quadratic,
    pic_cpu,
    solve_pic,
)
from .field_result import FieldResult
from .fd_rectangle import (
    FDSolver,
    build_fd_rectangle_resources,
    build_fd_resources,
    solve_poisson_fd,
)
from .fd_arbitrary import (
    AllSpaceAperture,
    ArbitraryFDSolver,
    EllipticAperture,
    IntersectionAperture,
    OctagonAperture,
    PolygonAperture,
    RacetrackAperture,
    RectangleAperture,
    build_aperture,
    build_fd_arbitrary_resources,
    solve_poisson_fd_arbitrary,
)
from .dst_rectangle import DSTRectangleSolver, build_dst_rectangle_resources
from .fft_free_space import FFTFreeSpaceSolver, build_fft_free_space_resources, solve_poisson_fft_free_space
from .formula_common import macro_charge_to_physical
from .formula_gaussian_round import gaussian_round_field
from .formula_gaussian_ellipse import gaussian_ellipse_field, gaussian_elliptic_field
from .formula_uniform_ellipse import uniform_elliptic_field
from .formula_uniform_round import uniform_round_field
