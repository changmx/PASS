"""Fixed-energy painting dipole with endpoint-held waveforms and SC/exit apertures."""
import logging
import numpy as np

from PASS.commands.command import Command
from PASS.utils.constants import const
from PASS.utils.aperture import check_aperture_cpu, check_aperture_gpu
from PASS.utils.slicing import configure_element_slicing, run_body_slices
from PASS.utils.bump_waveform import read_bump_waveform

logger = logging.getLogger(__name__)


def _drift(p, bunch, length, s0, turn):
    """Exact drift with a scalar length; zero length checks momentum only.

    The caller advances bunch.t0 separately for the reference particle.
    """
    xp = p.xp
    sl = slice(bunch.start_idx, bunch.end_idx)
    px, py, dp = p.px[sl], p.py[sl], p.dp[sl]
    tag = p.tag[sl]
    transverse = px * px + py * py
    longitudinal = (1 + dp)**2 - transverse
    lost = (tag > 0) & ((longitudinal <= 0) | (dp <= -1) | ~xp.isfinite(longitudinal))
    tag[lost] = -xp.abs(tag[lost])
    p.lost_turn[sl][lost] = turn
    p.lost_position[sl][lost] = s0
    if length == 0.0:
        return
    active = tag > 0
    px, py, dp = (xp.where(active, v, 0) for v in (px, py, dp))
    transverse = px * px + py * py
    pz = xp.sqrt(xp.where(active, longitudinal, 1))
    distance = xp.where(tag > 0, length, 0)
    p.x[sl] += distance * px / pz
    p.y[sl] += distance * py / pz
    inv_gamma2 = 1 / bunch.gamma**2
    energy = xp.sqrt(inv_gamma2 + (1 - inv_gamma2) * (1 + dp)**2)
    slip = (dp * (2 + dp) * inv_gamma2 - transverse) / (pz * (pz + energy))
    p.z[sl] += distance * slip


@Command.register("bump")
class Bump(Command):

    def __init__(self, beam_id, sim, **kwargs):
        values = {k.lower(): value for k, value in kwargs.items()}
        self.beam_id, self.cmd_type = beam_id, "Bump"
        self.cmd_name, self.s = values["name"], float(values["s (m)"])
        self.length = float(values.get("length (m)", 0.0))
        self.num_slice = values.get("num slices", 1)
        self.time_mode = values.get("time mode", "particle")
        self.time_offset = float(values.get("time offset (s)", 0.0))
        self.enabled = values.get("enable", True)
        if self.time_mode not in {"reference", "particle"}:
            raise ValueError("Bump Time mode must be reference or particle")
        if not np.isfinite(self.s) or not np.isfinite(self.time_offset):
            raise ValueError("Bump position and time offset must be finite")
        self.waveform, self.waveform_bounds = read_bump_waveform(values["waveform file"])
        # Outside either original plane range warrants a warning. These are
        # warning limits only: interpolation retains the full union of times.
        self._warning_start = float(self.waveform_bounds[:, 0].max())
        self._warning_end = float(self.waveform_bounds[:, 1].min())
        self.aperture_type = values.get("aperture type", "off").lower()
        self.aperture_value = values.get("aperture value", [])
        configure_element_slicing(self, sim, values)
        self._tables = {}
        self._gpu_cache = {}
        self._warned = False
        super().__init__()

    def print(self):
        logger.info(f"Bump {self.cmd_name}: S={self.s} L={self.length} slices={self.slice_plan.num_slices} clock={self.time_mode}")

    def execute_cpu(self, sim):
        return self._execute(sim)

    def execute_gpu(self, sim):
        """Fuse DKD steps between SC nodes; check apertures at SC entry and exit."""
        import cupy as cp

        beam, turn = sim.beams[self.beam_id], int(sim.state.turn)
        p = beam.particles
        key = (cp.cuda.runtime.getDevice(), p.dtype)
        if key not in self._gpu_cache:
            self._gpu_cache[key] = (
                cp.RawKernel(_bump_kernel_source(), "track_bump",
                    options=("--std=c++17", "--fmad=false",
                             f"-DPASS_USE_FLOAT={int(p.dtype == np.float32)}")),
                # Raw pointer indexing requires a known layout: TFS/pandas
                # tables can be Fortran-contiguous rather than row-major.
                cp.asarray(np.ascontiguousarray(self.waveform)),
                cp.zeros(1, dtype=cp.int32),
            )
        kernel, table, outside = self._gpu_cache[key]
        reference_time = (beam.reference_program.inverse_integral(float(turn))
                          if self.time_mode == "reference" and self.enabled else 0.0)
        reference_time += self.time_offset
        # One interpolation per invocation, independent of the particle count.
        h_ref, v_ref = (tuple(np.interp(reference_time, self.waveform[:, 0], self.waveform[:, j])
                             for j in (1, 2))
                        if self.time_mode == "reference" and self.enabled else (0., 0.))
        for bunch in beam.bunches:
            offset_s = 0.0
            count = bunch.end_idx - bunch.start_idx

            def launch(before, fraction, after, *, repeats=1):
                nonlocal offset_s
                if count:
                    kernel(((count + 255) // 256,), (256,), (
                        p.x, p.px, p.y, p.py, p.z, p.dp, p.tag,
                        p.lost_position, p.lost_turn, table, outside,
                        np.int32(bunch.start_idx), np.int32(bunch.end_idx), np.int32(len(self.waveform)),
                        np.int32(turn), np.int32(repeats),
                        np.int32(self.enabled and fraction != 0.), np.int32(self.time_mode == "particle"),
                        np.int32(not self._warned),
                        np.float64(before), np.float64(after), np.float64(fraction),
                        np.float64(self.s - self.length + offset_s), np.float64(bunch.t0),
                        np.float64(bunch.beta * const.c), p.real(1. / bunch.gamma**2),
                        np.float64(self.time_offset), np.float64(reference_time),
                        np.float64(self._warning_start), np.float64(self._warning_end),
                        np.float64(h_ref), np.float64(v_ref)))
                # Match the CPU sequence of reference half-drift advances,
                # including empty bunches. No particle correction state.
                for _ in range(repeats):
                    offset_s += before
                    bunch.t0 += before / (bunch.beta * const.c)
                    offset_s += after
                    bunch.t0 += after / (bunch.beta * const.c)

            def transport(ds, on_center):
                if on_center is None:
                    launch(ds / 2, ds / self.length, ds / 2)
                else:
                    launch(ds / 2, ds / self.length, 0.)
                    on_center()
                    launch(0., 0., ds / 2)

            if self.length == 0.:
                launch(0., 1., 0.)
            elif self._sc_nodes:
                run_body_slices(self, beam, bunch, turn, transport, gpu=True)
            else:
                # With only an exit aperture, all DKD slices stay in registers.
                ds = self.slice_plan.slice_length
                launch(ds / 2, ds / self.length, ds / 2, repeats=self.slice_plan.num_slices)
            check_aperture_gpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
        # Aggregate over bunches and slices, with at most one scalar readback
        # per invocation until the warning has actually been emitted.
        if self.enabled and not self._warned and bool(outside.get()[0]):
            self._warn_outside()
        return True

    def _warn_outside(self):
        logger.warning(
            "Bump %s: time outside supplied waveform range; holding nearest endpoint values "
            "(HKICK [%g, %g] s; VKICK [%g, %g] s)",
            self.cmd_name, *self.waveform_bounds.ravel())
        self._warned = True

    def _execute(self, sim):
        beam, turn = sim.beams[self.beam_id], int(sim.state.turn)
        p, xp = beam.particles, beam.particles.xp
        table = self._tables.get(xp)
        if table is None:
            table = self._tables[xp] = xp.asarray(self.waveform)
        check_aperture = check_aperture_cpu if xp is np else check_aperture_gpu
        for bunch in beam.bunches:
            sl = slice(bunch.start_idx, bunch.end_idx)
            reference_time = (beam.reference_program.inverse_integral(float(turn))
                              if self.time_mode == "reference" and self.enabled else 0.0)
            offset_s = 0.0

            def kick(fraction):
                if not self.enabled:
                    return
                alive = p.tag[sl] > 0
                if self.time_mode == "particle":
                    times = bunch.t0 - p.z[sl].astype(xp.float64) / (bunch.beta * const.c)
                    times = times[alive] + self.time_offset
                else:
                    times = xp.full(int(alive.sum()), reference_time + self.time_offset)
                if not self._warned and bool(xp.any(
                        (times < self._warning_start) | (times > self._warning_end))):
                    self._warn_outside()
                for column, name in ((1, "px"), (2, "py")):
                    values = xp.interp(times, table[:, 0], table[:, column])
                    getattr(p, name)[sl][alive] += (fraction * values).astype(p.dtype)

            def advance(ds):
                nonlocal offset_s
                _drift(p, bunch, ds, self.s - self.length + offset_s, turn)
                offset_s += ds
                bunch.t0 += ds / (bunch.beta * const.c)

            def transport(ds, on_center):
                advance(ds / 2)
                kick(ds / self.length)
                if on_center:
                    # Magnetic momentum losses must not enter the SC source.
                    _drift(p, bunch, 0.0, self.s - self.length + offset_s, turn)
                    on_center()
                advance(ds / 2)

            if self.length == 0:
                _drift(p, bunch, 0.0, self.s, turn)
                kick(1.0)
                # A large thin kick may immediately violate positive p_s.
                _drift(p, bunch, 0.0, self.s, turn)
            else:
                run_body_slices(self, beam, bunch, turn, transport, gpu=xp is not np)
            check_aperture(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
        return True


def _bump_kernel_source():
    """CUDA source colocated with the CPU map; only the instance caches state."""
    return r'''
#if PASS_USE_FLOAT
using R = float;
#else
using R = double;
#endif

__device__ bool bump_drift(R& x, R& y, R& z, R px, R py, R dp,
    int& tag, float& lost_s, int& lost_turn, R inv_gamma2,
    double length, double s0, int turn)
{
    if (tag <= 0) return false;
    R transverse = px*px + py*py;
    R one = (R)1 + dp;
    R longitudinal = one*one - transverse;
    if (longitudinal <= (R)0 || dp <= (R)-1 || !isfinite(longitudinal)) {
        tag = -abs(tag); lost_s = (float)s0; lost_turn = turn;
        return false;
    }
    if (length == 0.) return true;
    R ps = sqrt(longitudinal);
    R energy = sqrt(inv_gamma2 + ((R)1-inv_gamma2)*(one*one));
    R slip = (dp*((R)2+dp)*inv_gamma2-transverse)/(ps*(ps+energy));
    R distance = (R)length;
    x += distance*px/ps;
    y += distance*py/ps;
    z += distance*slip;
    return true;
}

extern "C" __global__ void track_bump(
    R* x, R* px, R* y, R* py, R* z, const R* dp, int* tag,
    float* lost_position, int* lost_turn, const double* table, int* outside,
    int start, int end, int rows, int turn, int repeats,
    int enabled, int particle_time, int check_warning,
    double before, double after, double fraction, double s0, double t0,
    double velocity, R inv_gamma2, double time_offset, double reference_time,
    double warning_start, double warning_end,
    double h_ref, double v_ref)
{
    int i = start + blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= end || tag[i] <= 0) return;
    R xi=x[i], yi=y[i], zi=z[i], pxi=px[i], pyi=py[i], dpi=dp[i];
    int ti=tag[i], lt=lost_turn[i];
    float ls=lost_position[i];
    for (int step=0; step<repeats; ++step) {
        if (!bump_drift(xi,yi,zi,pxi,pyi,dpi,ti,ls,lt,inv_gamma2,before,s0,turn)) break;
        s0 += before;
        t0 += before/velocity;
        if (enabled) {
            double time = particle_time ? (t0-(double)zi/velocity)+time_offset : reference_time;
            bool out = time < warning_start || time > warning_end;
            if (out && check_warning) {
                // One atomic per participating warp, not per particle.
                unsigned mask = __activemask();
                if ((threadIdx.x & 31) == __ffs(mask)-1) atomicExch(outside, 1);
            }
            double h=h_ref, v=v_ref;
            if (particle_time) {
                if (isnan(time)) {
                    h=time; v=time;  // match numpy.interp; validity check follows
                } else if (time <= table[0]) {
                    h=table[1]; v=table[2];
                } else if (time >= table[3*(rows-1)]) {
                    h=table[3*(rows-1)+1]; v=table[3*(rows-1)+2];
                } else {
                    int lo=0, hi=rows-1;
                    while (hi-lo > 1) {
                        int mid=(lo+hi)/2;
                        if (table[3*mid] <= time) lo=mid; else hi=mid;
                    }
                    double dt=time-table[3*lo], width=table[3*hi]-table[3*lo];
                    h=table[3*lo+1]+(table[3*hi+1]-table[3*lo+1])/width*dt;
                    v=table[3*lo+2]+(table[3*hi+2]-table[3*lo+2])/width*dt;
                }
            }
            pxi += (R)(fraction*h);
            pyi += (R)(fraction*v);
        }
        if (!bump_drift(xi,yi,zi,pxi,pyi,dpi,ti,ls,lt,inv_gamma2,after,s0,turn)) break;
        s0 += after;
        t0 += after/velocity;
    }
    x[i]=xi; y[i]=yi; z[i]=zi; px[i]=pxi; py[i]=pyi;
    tag[i]=ti; lost_position[i]=ls; lost_turn[i]=lt;
}
'''
