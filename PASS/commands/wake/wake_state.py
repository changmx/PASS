"""Private state of one physical wake location and one beam."""
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class WakeSources:
    times: np.ndarray
    widths: np.ndarray
    moments: dict
    betas: np.ndarray | None = None

    def __post_init__(self):
        times = np.array(self.times, dtype=float, copy=True)
        widths = np.array(self.widths, dtype=float, copy=True)
        if (times.ndim != 1 or widths.shape != times.shape or np.any(widths < 0)
                or not np.all(np.isfinite(times)) or not np.all(np.isfinite(widths))):
            raise ValueError("Wake source times/widths must be finite matching 1D arrays, widths >= 0")
        moments = {key: np.array(value, dtype=float, copy=True) for key, value in self.moments.items()}
        betas = np.ones_like(times) if self.betas is None else np.array(self.betas, dtype=float, copy=True)
        if betas.shape != times.shape or np.any(~np.isfinite(betas)) or np.any((betas <= 0) | (betas > 1)):
            raise ValueError("Wake source betas must match the bins and lie in (0, 1]")
        if any(v.shape != times.shape or not np.all(np.isfinite(v)) for v in moments.values()):
            raise ValueError("Wake source moments must be finite and match the time bins")
        for array in (times, widths, betas, *moments.values()):
            array.flags.writeable = False
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "widths", widths)
        object.__setattr__(self, "moments", moments)
        object.__setattr__(self, "betas", betas)


@dataclass
class WakeState:
    history: deque = field(default_factory=deque)
    mode_amplitudes: dict = field(default_factory=dict)
    last_time: float | None = None
    last_turn: int | None = None
    last_source_end: float | None = None
    convolution: object = None

    def fork(self, *, copy_modes=True):
        """Stage history; mode readers may borrow arrays if they replace on write."""
        modes = {k: v.copy() if copy_modes else v for k, v in self.mode_amplitudes.items()}
        return WakeState(deque(self.history), modes,
                         self.last_time, self.last_turn, self.last_source_end, self.convolution)

    def reset(self):
        self.history.clear()
        self.mode_amplitudes.clear()
        self.last_time = self.last_turn = self.last_source_end = None
        self.convolution = None

    def state_dict(self):
        def host(v):
            return v.get() if hasattr(v, "get") else np.asarray(v)
        return {"history": [{"turn": turn, "times": host(s.times).tolist(), "widths": host(s.widths).tolist(),
                             "betas": host(s.betas).tolist(),
                             "moments": [{"powers": list(k), "values": host(v).tolist()} for k, v in s.moments.items()]}
                            for turn, s in self.history],
                "modes": [{"component": k, "real": host(v).real.tolist(), "imag": host(v).imag.tolist(),
                           "complex": v.dtype.kind == "c"} for k, v in self.mode_amplitudes.items()],
                "last_time": self.last_time, "last_turn": self.last_turn, "last_source_end": self.last_source_end,
                "convolution": (self.convolution.state_dict() if hasattr(self.convolution, "state_dict")
                                else self.convolution)}

    @classmethod
    def from_state_dict(cls, data):
        history = deque()
        for row in data["history"]:
            history.append((row["turn"], WakeSources(row["times"], row["widths"],
                {tuple(m["powers"]): m["values"] for m in row["moments"]}, row["betas"])))
        modes = {}
        for row in data["modes"]:
            value = np.asarray(row["real"], float)
            if row["complex"]:
                value = value+1j*np.asarray(row["imag"], float)
            if value.ndim != 1 or not np.all(np.isfinite(value)):
                raise ValueError("Wake checkpoint contains invalid mode state")
            modes[row["component"]] = value
        for name in ("last_time", "last_source_end"):
            if data[name] is not None and not np.isfinite(data[name]):
                raise ValueError("Wake checkpoint contains invalid physical time")
        last_turn = data["last_turn"]
        if last_turn is not None and (isinstance(last_turn, bool) or not isinstance(last_turn, int) or last_turn < 0):
            raise ValueError("Wake checkpoint contains invalid turn")
        if any(isinstance(t, bool) or not isinstance(t, int) or t < 0 or last_turn is None or t > last_turn for t, _ in history):
            raise ValueError("Wake checkpoint history turn is inconsistent")
        import copy
        return cls(history, modes, data["last_time"], last_turn, data["last_source_end"],
                   copy.deepcopy(data.get("convolution")))


# ----------------------------------------------------------------------------
# GPU: device source arrays
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceSources:
    times: object
    widths: object
    moments: dict
    betas: object
    point: bool = False
    grid: tuple | None = None
    increasing: bool = False

    @classmethod
    def upload(cls, source):
        import cupy as cp
        if isinstance(source, cls):
            arrays=(source.times,source.widths,source.betas,*source.moments.values())
            device = cp.cuda.runtime.getDevice()
            if any(isinstance(v, cp.ndarray) and v.device.id != device for v in arrays):
                raise ValueError("Device wake sources belong to another CUDA device")
            if all(isinstance(v,cp.ndarray) and v.dtype==cp.float64 and v.flags.c_contiguous for v in arrays):
                if source.times.ndim != 1 or any(v.shape != source.times.shape for v in arrays):
                    raise ValueError("Device wake sources must have matching one-dimensional arrays")
                return source
            pack=lambda v:cp.ascontiguousarray(cp.asarray(v,dtype=cp.float64))
            normalized = cls(pack(source.times),pack(source.widths),{k:pack(v) for k,v in source.moments.items()},
                             pack(source.betas),source.point,source.grid,source.increasing)
            return cls.upload(normalized)
        # One transfer owns the complete emitted snapshot. Row views remain
        # independent of the host source and of all later passages.
        keys = tuple(source.moments)
        rows = cp.asarray(np.stack((source.times, source.widths, source.betas,
                                    *(source.moments[k] for k in keys))), dtype=cp.float64)
        return cls(rows[0], rows[1], {k: rows[i+3] for i, k in enumerate(keys)},
                   rows[2], bool(np.all(source.widths == 0)))

    def ordered(self, order):
        import cupy as cp
        keys=list(self.moments)
        rows=[self.times,self.widths,self.betas,*self.moments.values()]
        out=cp.empty((len(rows),len(order)),dtype=cp.float64)
        pointers=cp.asarray([v.data.ptr for v in rows],dtype=cp.uint64)
        order=cp.ascontiguousarray(order,dtype=cp.int64)
        if len(order):
            array_kernel(self,'gather_rows')(((out.size+255)//256,), (256,),
                (pointers,order,np.int64(len(order)),np.int32(len(rows)),out))
        return DeviceSources(out[0],out[1],{k:out[i+3] for i,k in enumerate(keys)},out[2],self.point)


def array_kernel(owner, name):
    """Instance-owned CUDA helpers for array transport and validation."""
    import cupy as cp
    cache=owner.__dict__.setdefault('_raw_array_kernels',{})
    key=(cp.cuda.runtime.getDevice(),name)
    if key not in cache:cache[key]=cp.RawKernel(_ARRAY_CODE,name,options=('--std=c++17',))
    return cache[key]


def finite_gpu(owner, values):
    import cupy as cp
    if not values.size:return True
    values=cp.ascontiguousarray(values,dtype=cp.float64)
    if values.size <= 65536:
        invalid = cp.empty(1, dtype=cp.int32)
        array_kernel(owner, 'finite_values_block')((1,), (256,),
            (values, np.int64(values.size), invalid))
        return not int(invalid[0])
    invalid=cp.zeros(1,dtype=cp.int32)
    if values.size:
        array_kernel(owner,'finite_values')((min(256,(values.size+255)//256),),(256,),
            (values,np.int64(values.size),invalid))
    return not int(invalid[0])


def source_metadata_gpu(owner, source):
    """One fused reduction/transfer: validity, beta range and physical window."""
    import cupy as cp
    result=cp.empty(5,dtype=cp.float64)
    moments=cp.asarray([v.data.ptr for v in source.moments.values()],dtype=cp.uint64)
    array_kernel(owner,'source_metadata')((1,),(256,),
        (source.times,source.widths,source.betas,moments,np.int32(len(source.moments)),np.int64(len(source.times)),result))
    return result.get()


def time_bounds_gpu(owner, source):
    import cupy as cp
    out=cp.empty(4,dtype=cp.float64)
    array_kernel(owner,'time_bounds')((1,),(256,),
        (source.times,source.widths,np.int64(len(source.times)),out))
    return out.get()


def shifted_times_gpu(owner, times, shift):
    import cupy as cp
    out=cp.empty_like(times)
    if out.size:array_kernel(owner,'shift_times')(((out.size+255)//256,),(256,),
        (times,np.float64(shift),np.int64(out.size),out))
    return out


def add_gpu(owner, target, value):
    if target.size:
        array_kernel(owner,'add_values')(((target.size+255)//256,),(256,),
            (target,value,np.int64(target.size)))
    return target


_ARRAY_CODE=r'''
#define INFINITY __longlong_as_double(0x7ff0000000000000LL)
extern "C" __global__ void shift_times(const double* t,double shift,long long n,double* out){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=t[i]+shift;
}
extern "C" __global__ void time_bounds(const double* t,const double* w,long long n,double* out){
    __shared__ double s[4][256];int k=threadIdx.x;double lo=INFINITY,hi=-INFINITY,width=0.,end=-INFINITY;
    for(long long i=k;i<n;i+=256){lo=fmin(lo,t[i]);hi=fmax(hi,t[i]);width=fmax(width,w[i]);end=fmax(end,t[i]+w[i]*.5);}
    s[0][k]=lo;s[1][k]=hi;s[2][k]=width;s[3][k]=end;__syncthreads();
    for(int step=128;step;step/=2){if(k<step){s[0][k]=fmin(s[0][k],s[0][k+step]);
        for(int row=1;row<4;row++)s[row][k]=fmax(s[row][k],s[row][k+step]);}__syncthreads();}
    if(k<4)out[k]=s[k][0];
}
extern "C" __global__ void finite_values(const double* a,long long n,int* invalid){
    for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)blockDim.x*gridDim.x)
        if(!isfinite(a[i]))atomicExch(invalid,1);
}
extern "C" __global__ void finite_values_block(const double* a,long long n,int* invalid){
    int bad=0;
    for(long long i=threadIdx.x;i<n;i+=blockDim.x)bad|=!isfinite(a[i]);
    int any_bad=__syncthreads_or(bad);
    if(threadIdx.x==0)*invalid=any_bad;
}
extern "C" __global__ void add_values(double* a,const double* b,long long n){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)a[i]+=b[i];
}
extern "C" __global__ void gather_rows(const unsigned long long* ptr,const long long* order,long long n,int nr,double* out){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n*nr)out[i]=((const double*)ptr[i/n])[order[i%n]];
}
extern "C" __global__ void source_metadata(const double* t,const double* w,const double* beta,
    const unsigned long long* moments,int nm,long long n,double* result){
    __shared__ double shared[5][256];int k=threadIdx.x;
    double valid=1.,lo=INFINITY,hi=-INFINITY,start=INFINITY,end=-INFINITY;
    for(long long i=k;i<n;i+=256){double b=beta[i],a=t[i]-w[i]*.5,z=t[i]+w[i]*.5;
        if(!isfinite(t[i])||!isfinite(w[i])||w[i]<0.||!isfinite(b)||b<=0.||b>1.)valid=0.;
        lo=fmin(lo,b);hi=fmax(hi,b);start=fmin(start,a);end=fmax(end,z);
        for(int j=0;j<nm;j++)if(!isfinite(((const double*)moments[j])[i]))valid=0.;}
    shared[0][k]=valid;shared[1][k]=lo;shared[2][k]=hi;shared[3][k]=start;shared[4][k]=end;
    __syncthreads();for(int offset=128;offset;offset/=2){if(k<offset){
        shared[0][k]=fmin(shared[0][k],shared[0][k+offset]);shared[1][k]=fmin(shared[1][k],shared[1][k+offset]);
        shared[2][k]=fmax(shared[2][k],shared[2][k+offset]);shared[3][k]=fmin(shared[3][k],shared[3][k+offset]);
        shared[4][k]=fmax(shared[4][k],shared[4][k+offset]);}__syncthreads();}
    if(k<5)result[k]=shared[k][0];
}
'''
