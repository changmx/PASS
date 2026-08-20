from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo

import numpy as np
import logging

logger = logging.getLogger(__name__)

_VALID_TYPES = {"off", "default", "circle", "rectangle", "ellipse", "rectcircle", "rectellipse", "racetrack", "octagon", "polygon"}

# ------------------------------------------------------------------
# CPU
# ------------------------------------------------------------------


def _mark_lost_cpu(tag, lost_position, lost_turn, mask, s_position, turn):
    """Mark particles selected by mask as lost."""
    tag[mask] = -np.abs(tag[mask])
    lost_position[mask] = s_position
    lost_turn[mask] = turn


def _check_rect_cpu(beam, bunch, half_width, half_height, s_position, turn):
    start = bunch.start_idx
    end = bunch.end_idx
    p = beam.particles
    x, y = p.x[start:end], p.y[start:end]
    tag, lost_position, lost_turn = p.tag[start:end], p.lost_position[start:end], p.lost_turn[start:end]

    alive = tag > 0
    out = alive & ((np.abs(x) > half_width) | (np.abs(y) > half_height))
    _mark_lost_cpu(tag, lost_position, lost_turn, out, s_position, turn)


def _check_circle_cpu(beam, bunch, radius, s_position, turn):
    start = bunch.start_idx
    end = bunch.end_idx
    p = beam.particles
    x, y = p.x[start:end], p.y[start:end]
    tag, lost_position, lost_turn = p.tag[start:end], p.lost_position[start:end], p.lost_turn[start:end]

    alive = tag > 0
    out = alive & ((x * x + y * y) > (radius * radius))
    _mark_lost_cpu(tag, lost_position, lost_turn, out, s_position, turn)


def _check_ellipse_cpu(beam, bunch, a, b, s_position, turn):
    start = bunch.start_idx
    end = bunch.end_idx
    p = beam.particles
    x, y = p.x[start:end], p.y[start:end]
    tag, lost_position, lost_turn = p.tag[start:end], p.lost_position[start:end], p.lost_turn[start:end]

    alive = tag > 0
    out = alive & (((x / a)**2 + (y / b)**2) > 1.0)
    _mark_lost_cpu(tag, lost_position, lost_turn, out, s_position, turn)


def _check_rectcircle_cpu(beam, bunch, half_width, half_height, radius, s_position, turn):
    start = bunch.start_idx
    end = bunch.end_idx
    p = beam.particles
    x, y = p.x[start:end], p.y[start:end]
    tag, lost_position, lost_turn = p.tag[start:end], p.lost_position[start:end], p.lost_turn[start:end]

    alive = tag > 0
    out = alive & ((np.abs(x) > half_width) | (np.abs(y) > half_height) | ((x * x + y * y) > (radius * radius)))
    _mark_lost_cpu(tag, lost_position, lost_turn, out, s_position, turn)


def _check_rectellipse_cpu(beam, bunch, w, h, a, b, s_position, turn):
    start = bunch.start_idx
    end = bunch.end_idx
    p = beam.particles
    x, y = p.x[start:end], p.y[start:end]
    tag, lost_position, lost_turn = p.tag[start:end], p.lost_position[start:end], p.lost_turn[start:end]

    alive = tag > 0
    out = alive & ((np.abs(x) > w) | (np.abs(y) > h) | (((x / a)**2 + (y / b)**2) > 1.0))
    _mark_lost_cpu(tag, lost_position, lost_turn, out, s_position, turn)


def _check_racetrack_cpu(beam, bunch, w, h, a, b, s_position, turn):
    """Racetrack = rectangle (half-width w, half-height h) ∪ semi-ellipses
    (half-axes a in x, b in y) centered at (±w, 0).

    Particle survives if inside rectangle OR inside ellipse end.
    """
    start = bunch.start_idx
    end = bunch.end_idx
    p = beam.particles
    x, y = p.x[start:end], p.y[start:end]
    tag, lost_position, lost_turn = p.tag[start:end], p.lost_position[start:end], p.lost_turn[start:end]

    alive = tag > 0
    ax = np.abs(x)
    ay = np.abs(y)

    in_rect = (ax <= w) & (ay <= h)
    in_ellipse = (ax > w) & ((((ax - w) / a)**2 + (y / b)**2) <= 1.0)
    out = alive & ~(in_rect | in_ellipse)
    _mark_lost_cpu(tag, lost_position, lost_turn, out, s_position, turn)


def _check_octagon_cpu(beam, bunch, w, h, d, s_position, turn):
    """Octagon = rectangle (half-width w, half-height h) with 45° chamfered corners.
    d is the half-diagonal gap (chamfer distance).
    Loss: |x| > w OR |y| > h OR |x|+|y| > w+h-d
    """
    start = bunch.start_idx
    end = bunch.end_idx
    p = beam.particles
    x, y = p.x[start:end], p.y[start:end]
    tag, lost_position, lost_turn = p.tag[start:end], p.lost_position[start:end], p.lost_turn[start:end]

    alive = tag > 0
    out = alive & ((np.abs(x) > w) | (np.abs(y) > h) | ((np.abs(x) + np.abs(y)) > (w + h - d)))
    _mark_lost_cpu(tag, lost_position, lost_turn, out, s_position, turn)


def _check_polygon_cpu(beam, bunch, vertices, s_position, turn):
    """Polygon aperture via ray-casting point-in-polygon test.

    vertices: list of [x, y] pairs, auto-closed (last vertex connects to first).
    """
    start = bunch.start_idx
    end = bunch.end_idx
    p = beam.particles
    x, y = p.x[start:end], p.y[start:end]
    tag, lost_position, lost_turn = p.tag[start:end], p.lost_position[start:end], p.lost_turn[start:end]

    nvert = len(vertices)
    vertx = np.array([v[0] for v in vertices], dtype=np.float64)
    verty = np.array([v[1] for v in vertices], dtype=np.float64)

    inside = np.zeros(len(x), dtype=bool)
    for i in range(nvert):
        j = nvert - 1 if i == 0 else i - 1
        yi, yj = verty[i], verty[j]
        xi, xj = vertx[i], vertx[j]

        cond1 = (yi > y) != (yj > y)
        denom = yj - yi
        # avoid division by zero: when yi == yj, cond1 is always False
        x_intersect = np.where(denom != 0.0, (xj - xi) * (y - yi) / denom + xi, np.inf)
        cond2 = x < x_intersect

        inside ^= (cond1 & cond2)

    alive = tag > 0
    out = alive & ~inside
    _mark_lost_cpu(tag, lost_position, lost_turn, out, s_position, turn)


def check_aperture_cpu(beam: Beam, bunch: BunchInfo, aperture_type: str, aperture_value: list, s_position: float, turn: int):
    """Dispatch aperture check based on type (CPU).

    aperture_type: "off", "default", "circle", "rectangle", "ellipse",
                   "rectcircle", "rectellipse", "racetrack", "octagon", "polygon"
    aperture_value:
        circle:       [radius]
        rectangle:    [half_width, half_height]
        ellipse:      [a, b]                          (half-major axis a, half-minor axis b)
        rectcircle:   [half_width, half_height, radius]
        rectellipse:  [w, h, a, b]                    (half-width w, half-height h, half-major axis a, half-minor axis b)
        racetrack:    [w, h, a, b]                    (half-width w, half-height h, ellipse-corner half-axes a, b)
        octagon:      [w, h, d]                       (half-width w, half-height h, half-diagonal gap d)
        polygon:      [[x1,y1], [x2,y2], ...]
        off/default:  ignored
    """
    aperture_type = aperture_type.lower()

    if aperture_type == "off":
        return
    elif aperture_type == "default":
        _check_rect_cpu(beam, bunch, 1.0, 1.0, s_position, turn)
    elif aperture_type == "circle":
        _check_circle_cpu(beam, bunch, aperture_value[0], s_position, turn)
    elif aperture_type == "rectangle":
        _check_rect_cpu(beam, bunch, aperture_value[0], aperture_value[1], s_position, turn)
    elif aperture_type == "ellipse":
        _check_ellipse_cpu(beam, bunch, aperture_value[0], aperture_value[1], s_position, turn)
    elif aperture_type == "rectcircle":
        _check_rectcircle_cpu(beam, bunch, aperture_value[0], aperture_value[1], aperture_value[2], s_position, turn)
    elif aperture_type == "rectellipse":
        _check_rectellipse_cpu(beam, bunch, aperture_value[0], aperture_value[1], aperture_value[2], aperture_value[3], s_position, turn)
    elif aperture_type == "racetrack":
        _check_racetrack_cpu(beam, bunch, aperture_value[0], aperture_value[1], aperture_value[2], aperture_value[3], s_position, turn)
    elif aperture_type == "octagon":
        _check_octagon_cpu(beam, bunch, aperture_value[0], aperture_value[1], aperture_value[2], s_position, turn)
    elif aperture_type == "polygon":
        _check_polygon_cpu(beam, bunch, aperture_value, s_position, turn)
    else:
        raise ValueError(f"Unknown aperture type: {aperture_type}. Must be one of {sorted(_VALID_TYPES)}")


# ------------------------------------------------------------------
# GPU
# ------------------------------------------------------------------

kernel_code = r'''
extern "C" __global__
void check_aperture_rect(
    double* __restrict__ x, double* __restrict__ y,
    int* __restrict__ tag, double* __restrict__ lost_position, int* __restrict__ lost_turn,
    int start_index, int end_index,
    double half_width, double half_height, double s_position, int turn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;
    if (tag[i] > 0)
    {
        if (fabs(x[i]) > half_width || fabs(y[i]) > half_height)
        {
            tag[i] = -tag[i];
            lost_position[i] = s_position;
            lost_turn[i] = turn;
        }
    }
}

extern "C" __global__
void check_aperture_circle(
    double* __restrict__ x, double* __restrict__ y,
    int* __restrict__ tag, double* __restrict__ lost_position, int* __restrict__ lost_turn,
    int start_index, int end_index,
    double radius, double s_position, int turn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;
    if (tag[i] > 0)
    {
        if ((x[i] * x[i] + y[i] * y[i]) > (radius * radius))
        {
            tag[i] = -tag[i];
            lost_position[i] = s_position;
            lost_turn[i] = turn;
        }
    }
}

extern "C" __global__
void check_aperture_ellipse(
    double* __restrict__ x, double* __restrict__ y,
    int* __restrict__ tag, double* __restrict__ lost_position, int* __restrict__ lost_turn,
    int start_index, int end_index,
    double a, double b, double s_position, int turn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;
    if (tag[i] > 0)
    {
        double tx = x[i] / a;
        double ty = y[i] / b;
        if ((tx * tx + ty * ty) > 1.0)
        {
            tag[i] = -tag[i];
            lost_position[i] = s_position;
            lost_turn[i] = turn;
        }
    }
}

extern "C" __global__
void check_aperture_rectcircle(
    double* __restrict__ x, double* __restrict__ y,
    int* __restrict__ tag, double* __restrict__ lost_position, int* __restrict__ lost_turn,
    int start_index, int end_index,
    double half_width, double half_height, double radius, double s_position, int turn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;
    if (tag[i] > 0)
    {
        if (fabs(x[i]) > half_width || fabs(y[i]) > half_height ||
            (x[i] * x[i] + y[i] * y[i]) > (radius * radius))
        {
            tag[i] = -tag[i];
            lost_position[i] = s_position;
            lost_turn[i] = turn;
        }
    }
}

extern "C" __global__
void check_aperture_rectellipse(
    double* __restrict__ x, double* __restrict__ y,
    int* __restrict__ tag, double* __restrict__ lost_position, int* __restrict__ lost_turn,
    int start_index, int end_index,
    double w, double h, double a, double b,
    double s_position, int turn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;
    if (tag[i] > 0)
    {
        double tx = x[i] / a;
        double ty = y[i] / b;
        if (fabs(x[i]) > w || fabs(y[i]) > h || (tx * tx + ty * ty) > 1.0)
        {
            tag[i] = -tag[i];
            lost_position[i] = s_position;
            lost_turn[i] = turn;
        }
    }
}

extern "C" __global__
void check_aperture_racetrack(
    double* __restrict__ x, double* __restrict__ y,
    int* __restrict__ tag, double* __restrict__ lost_position, int* __restrict__ lost_turn,
    int start_index, int end_index,
    double w, double h, double a, double b,
    double s_position, int turn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;
    if (tag[i] > 0)
    {
        double ax = fabs(x[i]);
        double ay = fabs(y[i]);

        bool in_rect = (ax <= w) && (ay <= h);
        bool in_ellipse = false;
        if (ax > w)
        {
            double dx = (ax - w) / a;
            double ty = y[i] / b;
            in_ellipse = (dx * dx + ty * ty) <= 1.0;
        }

        if (!in_rect && !in_ellipse)
        {
            tag[i] = -tag[i];
            lost_position[i] = s_position;
            lost_turn[i] = turn;
        }
    }
}

extern "C" __global__
void check_aperture_octagon(
    double* __restrict__ x, double* __restrict__ y,
    int* __restrict__ tag, double* __restrict__ lost_position, int* __restrict__ lost_turn,
    int start_index, int end_index,
    double w, double h, double d, double s_position, int turn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;
    if (tag[i] > 0)
    {
        double ax = fabs(x[i]);
        double ay = fabs(y[i]);
        if (ax > w || ay > h || (ax + ay) > (w + h - d))
        {
            tag[i] = -tag[i];
            lost_position[i] = s_position;
            lost_turn[i] = turn;
        }
    }
}

extern "C" __global__
void check_aperture_polygon(
    double* __restrict__ x, double* __restrict__ y,
    int* __restrict__ tag, double* __restrict__ lost_position, int* __restrict__ lost_turn,
    int start_index, int end_index,
    int nvert, const double* __restrict__ vertx, const double* __restrict__ verty,
    double s_position, int turn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;
    if (tag[i] > 0)
    {
        bool inside = false;
        for (int k = 0, j = nvert - 1; k < nvert; j = k++)
        {
            if ((verty[k] > y[i]) != (verty[j] > y[i]))
            {
                double x_intersect = (vertx[j] - vertx[k]) * (y[i] - verty[k]) / (verty[j] - verty[k]) + vertx[k];
                if (x[i] < x_intersect)
                    inside = !inside;
            }
        }
        if (!inside)
        {
            tag[i] = -tag[i];
            lost_position[i] = s_position;
            lost_turn[i] = turn;
        }
    }
}
'''

_kernel_cache = {}


def _get_kernel(name):
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "GPU aperture checks require the optional 'cuda' dependencies "
            "(install PASS with the [cuda] extra)."
        ) from exc
    if name not in _kernel_cache:
        _kernel_cache[name] = cp.RawKernel(kernel_code, name)
    return _kernel_cache[name]


def _launch_gpu(kernel, beam, bunch, *args):
    start = bunch.start_idx
    end = bunch.end_idx
    p = beam.particles
    N = end - start
    threads = 256
    blocks = (N + threads - 1) // threads
    kernel((blocks, ), (threads, ), (p.x, p.y, p.tag, p.lost_position, p.lost_turn, start, end, *args))


def check_aperture_gpu(beam: Beam, bunch: BunchInfo, aperture_type: str, aperture_value: list, s_position: float, turn: int):
    """Dispatch aperture check based on type (GPU)."""
    aperture_type = aperture_type.lower()

    if aperture_type == "off":
        return
    elif aperture_type == "default":
        _launch_gpu(_get_kernel("check_aperture_rect"), beam, bunch, 1.0, 1.0, s_position, turn)
    elif aperture_type == "circle":
        _launch_gpu(_get_kernel("check_aperture_circle"), beam, bunch, aperture_value[0], s_position, turn)
    elif aperture_type == "rectangle":
        _launch_gpu(_get_kernel("check_aperture_rect"), beam, bunch, aperture_value[0], aperture_value[1], s_position, turn)
    elif aperture_type == "ellipse":
        _launch_gpu(_get_kernel("check_aperture_ellipse"), beam, bunch, aperture_value[0], aperture_value[1], s_position, turn)
    elif aperture_type == "rectcircle":
        _launch_gpu(_get_kernel("check_aperture_rectcircle"), beam, bunch, aperture_value[0], aperture_value[1], aperture_value[2], s_position, turn)
    elif aperture_type == "rectellipse":
        _launch_gpu(_get_kernel("check_aperture_rectellipse"), beam, bunch, aperture_value[0], aperture_value[1], aperture_value[2],
                    aperture_value[3], s_position, turn)
    elif aperture_type == "racetrack":
        _launch_gpu(_get_kernel("check_aperture_racetrack"), beam, bunch, aperture_value[0], aperture_value[1], aperture_value[2], aperture_value[3],
                    s_position, turn)
    elif aperture_type == "octagon":
        _launch_gpu(_get_kernel("check_aperture_octagon"), beam, bunch, aperture_value[0], aperture_value[1], aperture_value[2], s_position, turn)
    elif aperture_type == "polygon":
        try:
            import cupy as cp
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "GPU aperture checks require the optional 'cuda' dependencies "
                "(install PASS with the [cuda] extra)."
            ) from exc
        vertx = cp.asarray([v[0] for v in aperture_value], dtype=cp.float64)
        verty = cp.asarray([v[1] for v in aperture_value], dtype=cp.float64)
        nvert = len(aperture_value)
        _launch_gpu(_get_kernel("check_aperture_polygon"), beam, bunch, nvert, vertx, verty, s_position, turn)
    else:
        raise ValueError(f"Unknown aperture type: {aperture_type}. Must be one of {sorted(_VALID_TYPES)}")
