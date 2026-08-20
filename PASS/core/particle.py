import numpy as np


class ParticlePool:

    real_fields = frozenset((
        "x", "px", "y", "py", "z", "dp",
        "last_x", "last_px", "last_y", "last_py",
        "last_phasex", "last_phasey",
    ))

    def __init__(self, n_particles: int, xp, dtype=np.float64,
                 is_cal_phase: bool = True):

        """
        px = Px/P0
        py = Py/P0
        z  = s-β0*c*t
        dp = (P-P0)/P0 = δ
        """

        self.xp = xp
        self.dtype = np.dtype(dtype)
        if self.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
            raise ValueError(
                "ParticlePool dtype must be float32 or float64, but got "
                f"{self.dtype}"
            )
        self.real = self.dtype.type

        self.x = self.xp.zeros(n_particles, dtype=self.dtype)
        self.px = self.xp.zeros(n_particles, dtype=self.dtype)
        self.y = self.xp.zeros(n_particles, dtype=self.dtype)
        self.py = self.xp.zeros(n_particles, dtype=self.dtype)
        self.z = self.xp.zeros(n_particles, dtype=self.dtype)
        self.dp = self.xp.zeros(n_particles, dtype=self.dtype)
        self.tag = self.xp.arange(1, 1 + n_particles, dtype=self.xp.int32)
        self.lost_turn = self.xp.full(n_particles, -1, dtype=self.xp.int32)
        self.lost_position = self.xp.full(n_particles, -1, dtype=self.xp.float32)
        self.slice_id = self.xp.full(n_particles, -1, dtype=self.xp.int32)

        self.last_x = self.xp.zeros(n_particles, dtype=self.dtype) if is_cal_phase else None
        self.last_px = self.xp.zeros(n_particles, dtype=self.dtype) if is_cal_phase else None
        self.last_y = self.xp.zeros(n_particles, dtype=self.dtype) if is_cal_phase else None
        self.last_py = self.xp.zeros(n_particles, dtype=self.dtype) if is_cal_phase else None
        self.last_phasex = self.xp.zeros(n_particles, dtype=self.dtype) if is_cal_phase else None
        self.last_phasey = self.xp.zeros(n_particles, dtype=self.dtype) if is_cal_phase else None

    def copy(self, xp_target, fields=None, dtype=None):
        """Copy to another backend and, optionally, another particle dtype.

        ``dtype`` changes six-dimensional state and phase-history arrays only.
        Integer metadata and float32 ``lost_position`` remain unchanged.
        """
        target_dtype = self.dtype if dtype is None else np.dtype(dtype)
        if target_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
            raise ValueError(
                "ParticlePool copy dtype must be float32 or float64, but got "
                f"{target_dtype}"
            )

        def convert(name, value):
            value_dtype = target_dtype if name in self.real_fields else None
            return convert_array(value, xp_target, dtype=value_dtype)

        new = ParticlePool.__new__(ParticlePool)
        new.xp = xp_target
        new.dtype = target_dtype
        new.real = target_dtype.type

        # Copy all data.
        # E.g. p_gpu = p.copy(cp)/p_cpu = p.copy(np)
        if fields is None:
            for k, v in self.__dict__.items():
                if k not in {"xp", "dtype", "real"}:
                    setattr(new, k, convert(k, v))
            return new

        # Copy a portion of the data.
        # E.g. p2 = p.copy(cp,fields=["x","px"])
        fields = set(fields)

        for k, v in self.__dict__.items():
            if k in {"xp", "dtype", "real"}:
                continue
            if k in fields:
                setattr(new, k, convert(k, v))
            else:
                setattr(new, k, v)

        return new


def convert_array(x, xp_target, dtype=None):
    if x is None:
        return None

    # GPU->CPU.  Avoid importing CuPy just to inspect a CPU array; CuPy
    # arrays expose ``get`` and the target backend already owns conversion.
    if xp_target is np and hasattr(x, "get") and x.__class__.__module__.startswith("cupy"):
        x = x.get()

    # CPU->GPU
    if isinstance(x, np.ndarray) and xp_target is not np:
        return xp_target.asarray(x, dtype=dtype)

    # CPU->CPU and GPU->GPU
    if dtype is not None and hasattr(x, "astype"):
        return x.astype(dtype, copy=False)
    return x
