"""Physical passage times through acceleration without changing PASS z.

Physical timing is owned by the beam's core ArrivalClock. This adapter owns
only wake projection geometry and delegates reference changes to that clock.
"""
import numpy as np

from PASS.utils.constants import const
from PASS.core.arrival import ensure_arrival_clock
from .wake_conventions import arrival_times, require_cpu


class WakeClock:
    def __init__(self, beam, slice_names=None):
        self.beam = beam
        self.arrival = ensure_arrival_clock(beam)
        self.xp = self.arrival.xp
        self.slice_names = None if slice_names is None else frozenset(slice_names)
        self.frozen = {}

    @property
    def times(self):
        return self.arrival.times

    @property
    def _arrival_offsets(self):
        return self.arrival._arrival_offsets

    def _relative_times(self, bunch):
        return self.arrival._relative_times(bunch)

    def state_dict(self):
        return self.arrival.state_dict()

    def load_state_dict(self, data):
        self.arrival.load_state_dict(data)
        self.frozen.clear()

    def before_command(self, command):
        self._gpu_immediate_geometry = (command.cmd_type.lower() == "wakefield"
            and getattr(self, "_last_slice_name", None) == command.slice_set_name)
        p = self.beam.particles
        if command.cmd_type.lower() in {"sortbunch", "reorganizebunch"} and not hasattr(p, "wake_macro_charge"):
            # Bunch-average macro weights cannot preserve unequal source charges
            # after particles are exchanged between bunches. Freeze individual
            # signed charges before the first regrouping and permute them too.
            p.wake_macro_charge = self.xp.zeros(len(p.z), dtype=self.xp.float64)
            for b in self.beam.bunches:
                p.wake_macro_charge[b.start_idx:b.end_idx] = b.ratio*b.num_charge*const.e
        self.arrival.before_command(command)

    def after_command(self, command):
        regrouped = self.arrival._regrouping
        injection_tags = self.arrival._injection_tags
        self.arrival.after_command(command)
        p = self.beam.particles
        if injection_tags is not None and hasattr(p, "wake_macro_charge"):
            born = (injection_tags <= 0) & (p.tag > 0)
            for b in self.beam.bunches:
                sl = slice(b.start_idx, b.end_idx)
                p.wake_macro_charge[sl][born[sl]] = b.ratio*b.num_charge*const.e
        if regrouped:
            self.frozen.clear()
        if (command.cmd_type.lower() == "slicer"
                and (self.slice_names is None or command.slice_set_name in self.slice_names)):
            self.freeze_slices(command.slice_set_name)
            self._last_slice_name = command.slice_set_name
        else:
            self._last_slice_name = None
            # Core marks samples dirty; the next consumer samples on demand.
        self._gpu_immediate_geometry = False

    def freeze_slices(self, name):
        if any(getattr(b.slice_sets[name], "coordinate", None) == "ring_position"
               for b in self.beam.bunches):
            raise ValueError("Wake timing requires z_rel or arrival_phase, not ring_position")
        if self.xp is not np:
            return freeze_gpu(self, name)
        self.sample()
        for bunch in self.beam.bunches:
            slices = bunch.slice_sets[name]
            if slices.slice_id is None or slices.slice_table is None:
                continue
            sl = slice(bunch.start_idx, bunch.end_idx)
            z = np.asarray(self.beam.particles.z[sl], float)
            ids = np.asarray(slices.slice_id)
            alive = (self.beam.particles.tag[sl] > 0) & (ids >= 0)
            n = len(slices.slice_table["z_center"])
            count = np.bincount(ids[alive], minlength=n)
            mean = lambda values: np.divide(np.bincount(ids[alive], weights=values[alive], minlength=n),
                                            count, out=np.zeros(n), where=count > 0)
            relative_times = self._relative_times(bunch)
            if slices.periodic:
                # A frozen whole-ring profile occupies [t0,t0+T). Its bins
                # are arrival phases, not unwrapped geometric z_rel bins.
                centers = bunch.t0-np.asarray(slices.slice_table["z_center"], float)/(bunch.beta*const.c)
                self.frozen[(bunch.bunch_id, name)] = (slices.valid_turn,
                    np.array(self.times[sl], copy=True), centers,
                    np.asarray(slices.slice_table["delta_z"], float)/(bunch.beta*const.c))
                continue
            mean_z, mean_t = mean(z), mean(relative_times)
            dz, dt = z-mean_z[np.maximum(ids, 0)], relative_times-mean_t[np.maximum(ids, 0)]
            variance = mean(dz*dz)
            slope = np.divide(mean(dz*dt), variance, out=np.full(n, -1/(bunch.beta*const.c)), where=variance > 0)
            centers = arrival_times(slices.slice_table["z_center"], bunch)
            occupied = count > 0
            centers[occupied] = bunch.t0 + (mean_t+slope*(slices.slice_table["z_center"]-mean_z))[occupied]
            # An empty bin carries no source/witness; its geometric location
            # must nevertheless share the same clock origin after acceleration.
            if np.any(alive):
                correction = np.mean(self._arrival_offsets[sl][alive])
                centers[~occupied] += correction
            self.frozen[(bunch.bunch_id, name)] = (
                slices.valid_turn, np.array(self.times[sl], copy=True),
                centers, np.asarray(slices.slice_table["delta_z"], float)*np.abs(slope),
            )

    def sample(self):
        return self.arrival.sample()

    def slice_geometry(self, bunch, name, ids, alive):
        if self.xp is not np:
            return geometry_gpu(self, bunch, name, ids, alive)
        slices = bunch.slice_sets[name]
        key = (bunch.bunch_id, name)
        if key not in self.frozen or self.frozen[key][0] != slices.valid_turn:
            # Standalone command use, with Slicer at the current location.
            self.freeze_slices(name)
        _, old_times, centers, widths = self.frozen[key]
        if slices.periodic:
            return centers, widths
        current = self.times[bunch.start_idx:bunch.end_idx]
        count = np.bincount(ids[alive], minlength=len(centers))
        shifts = np.bincount(ids[alive], weights=(current-old_times)[alive], minlength=len(centers))
        shifts = np.divide(shifts, count, out=np.zeros_like(shifts), where=count > 0)
        # Fixed membership/width for this turn; transport each bin by its
        # current mean arrival-time shift, and recompute transverse moments.
        return centers+shifts, widths

    def sample_gpu(self):
        return self.sample()


def prepare_wake_tracking(sim, sequences, *, resume=False):
    commands = [cmd for seq in sequences for cmd in seq.cmds]
    wake_ids = {cmd.beam_id for cmd in commands if cmd.cmd_type == "WakeField" and cmd.is_enabled}
    for cmd in commands:
        if not resume and cmd.cmd_type == "WakeField" and cmd.is_enabled and any(s.last_turn is not None for s in cmd.group_states):
            raise ValueError("WakeField retains state from an earlier run; call reset_state() before restarting Executor")
    for beam_id in wake_ids:
        if resume and hasattr(sim.beams[beam_id], "wake_clock"):
            continue
        names = {cmd.slice_set_name for cmd in commands
                 if cmd.beam_id == beam_id and cmd.cmd_type == "WakeField" and cmd.is_enabled}
        sim.beams[beam_id].wake_clock = WakeClock(sim.beams[beam_id], names)
    return wake_ids


# ----------------------------------------------------------------------------
# GPU: device timing
# ----------------------------------------------------------------------------


_GPU_KERNELS = {}
_GPU_CODE = r'''
#if FLOAT_PARTICLES
using R=float;
#else
using R=double;
#endif
extern "C" __global__ void freeze_moments(const R* z,const double* correction,const int* tag,
    const int* ids,int start,int end,int ns,const double* geometric,double velocity,
    int pass,const double* first,double* out,int shared){
    extern __shared__ double hist[];
    int rows=pass==0?3:2;
    if(shared)for(int k=threadIdx.x;k<rows*ns;k+=blockDim.x)hist[k]=0.;
    __syncthreads();
    for(int i=start+blockIdx.x*blockDim.x+threadIdx.x;i<end;i+=blockDim.x*gridDim.x){
        int id=ids[i-start];if(tag[i]<=0||id<0||id>=ns)continue;
        double dz=(double)z[i]-geometric[id],r=correction[i]-dz/velocity;
        double* h=shared?hist:out;
        if(pass==0){atomicAdd(h+id,1.);atomicAdd(h+ns+id,dz);atomicAdd(h+2*ns+id,r);}
        else{
            double n=fmax(first[id],1.);
            dz-=first[ns+id]/n;r-=first[2*ns+id]/n;
            atomicAdd(h+id,dz*dz);atomicAdd(h+ns+id,dz*r);
        }
    }
    __syncthreads();
    if(shared)for(int k=threadIdx.x;k<rows*ns;k+=blockDim.x)atomicAdd(out+k,hist[k]);
}
extern "C" __global__ void freeze_geometry(const double* first,const double* second,
    const double* geometric,const double* widths,int ns,double center,double velocity,
    double* times,double* durations){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=ns)return;
    double slope=second[j]>0?second[ns+j]/second[j]:-1./velocity;
    if(first[j]>0)times[j]=(first[2*ns+j]-slope*first[ns+j])/first[j]-(geometric[j]+center)/velocity;
    else{
        double n=0.,offset=0.;
        for(int k=0;k<ns;k++){n+=first[k];offset+=first[2*ns+k]+first[ns+k]/velocity;}
        times[j]=offset/fmax(n,1.)-(geometric[j]+center)/velocity;
    }
    durations[j]=widths[j]*fabs(slope);
}
'''


def _gpu_kernels(p):
    import cupy as cp
    key = (cp.cuda.runtime.getDevice(), np.dtype(p.dtype))
    if key not in _GPU_KERNELS:
        names = ('freeze_moments', 'freeze_geometry')
        module = cp.RawModule(code=_GPU_CODE, options=("--std=c++17",
            f"-DFLOAT_PARTICLES={int(p.dtype == np.float32)}"), name_expressions=names)
        _GPU_KERNELS[key] = {name: module.get_function(name) for name in names}
    return _GPU_KERNELS[key]


def freeze_gpu(clock, name):
    import cupy as cp
    clock.sample()
    all_zero = bool(cp.all(clock.beam.particles.arrival_offset == 0))
    uniform = clock.__dict__.setdefault("_gpu_uniform_grids", {})
    for b in clock.beam.bunches:
        slices = b.slice_sets[name]
        uniform.pop((b.bunch_id, name), None)
        if slices.slice_id is None or slices.slice_table is None:
            continue
        sl = slice(b.start_idx, b.end_idx)
        p = clock.beam.particles
        ids = cp.asarray(slices.slice_id, dtype=cp.int32)
        ns = len(slices.slice_table["z_center"])
        if slices.model == "equal_length" and hasattr(slices, "_gpu_geometry"):
            # Keep the declared physical mesh in double precision even when
            # particle storage (and Slicer membership arithmetic) is float32.
            lo, hi = ((slices.explicit.z_min, slices.explicit.z_max) if slices.z_range_mode == "explicit"
                      else slices._gpu_geometry)
            dz = (hi-lo)/ns
            geometric = hi-(cp.arange(ns, dtype=cp.float64)+.5)*dz
            widths = cp.full(ns, dz, dtype=cp.float64)
        else:
            geometric = cp.asarray(slices.slice_table["z_center"], dtype=cp.float64)
            widths = cp.asarray(slices.slice_table["delta_z"], dtype=cp.float64)
        relative = clock._relative_times(b)
        if slices.periodic:
            clock.frozen[(b.bunch_id, name)] = (slices.valid_turn, relative,
                -geometric/(b.beta*const.c), widths/(b.beta*const.c), b.t0)
            uniform[(b.bunch_id, name)] = (hi-lo)/(ns*b.beta*const.c)
            continue
        if all_zero:
            # This exact affine geometry is common in fixed-energy tracking.
            # No regression or atomically accumulated particle statistics are
            # necessary. A device reduction verifies the condition each time.
            clock.frozen[(b.bunch_id, name)] = (slices.valid_turn, relative,
                -(geometric+b.z_center)/(b.beta*const.c), widths/(b.beta*const.c), b.t0)
            if slices.model == "equal_length" and hasattr(slices, "_gpu_geometry"):
                uniform[(b.bunch_id, name)] = (hi-lo)/(ns*b.beta*const.c)
            continue
        # Two centered histogram passes avoid a separate bincount (and its
        # device-to-host validation) for every statistical moment. Centering
        # both coordinates preserves weak clock corrections at large z_rel.
        first, second = cp.zeros((3, ns)), cp.zeros((2, ns))
        centers, duration = cp.empty(ns), cp.empty(ns)
        launch = (min(256, max(1, (b.end_idx-b.start_idx+255)//256)),)
        shared = 3*ns*8 <= 24576
        for phase, out in ((0, first), (1, second)):
            _gpu_kernels(p)["freeze_moments"](launch, (256,),
                (p.z, p.arrival_offset, p.tag, ids, np.int32(b.start_idx), np.int32(b.end_idx),
                 np.int32(ns), geometric, np.float64(b.beta*const.c), np.int32(phase), first, out, np.int32(shared)),
                shared_mem=(3 if phase == 0 else 2)*ns*8 if shared else 0)
        _gpu_kernels(p)["freeze_geometry"](((ns+255)//256,), (256,),
            (first, second, geometric, widths, np.int32(ns), np.float64(b.z_center),
             np.float64(b.beta*const.c), centers, duration))
        clock.frozen[(b.bunch_id, name)] = (slices.valid_turn, relative.copy(), centers,
                                            duration, b.t0)


def geometry_gpu(clock, bunch, name, ids, alive):
    import cupy as cp
    slices = bunch.slice_sets[name]
    key = (bunch.bunch_id, name)
    if key not in clock.frozen or clock.frozen[key][0] != slices.valid_turn:
        freeze_gpu(clock, name)
    _, old_relative, centers, widths, old_t0 = clock.frozen[key]
    if slices.periodic or getattr(clock, "_gpu_immediate_geometry", False):
        return bunch.t0+centers, widths
    ns = len(centers)
    valid = alive & (ids >= 0) & (ids < ns)
    safe = cp.clip(ids, 0, max(ns-1, 0))
    counts = cp.bincount(safe, weights=valid.astype(cp.float64), minlength=ns)
    shifts = cp.bincount(safe, weights=cp.where(valid, clock._relative_times(bunch)-old_relative, 0.), minlength=ns)
    shifts /= cp.maximum(counts, 1.)
    return bunch.t0+(centers+shifts), widths
