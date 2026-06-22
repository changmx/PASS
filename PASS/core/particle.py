import numpy as np
import cupy as cp


class ParticlePool:

    def __init__(self, n_particles: int, xp, is_cal_phase: bool = True):

        self.xp = xp

        self.x = self.xp.zeros(n_particles, dtype=self.xp.float64)
        self.px = self.xp.zeros(n_particles, dtype=self.xp.float64)
        self.y = self.xp.zeros(n_particles, dtype=self.xp.float64)
        self.py = self.xp.zeros(n_particles, dtype=self.xp.float64)
        self.z = self.xp.zeros(n_particles, dtype=self.xp.float64)
        self.pz = self.xp.zeros(n_particles, dtype=self.xp.float64)
        self.tag = self.xp.arange(1, 1 + n_particles, dtype=self.xp.int32)
        self.lost_turn = self.xp.full(n_particles, -1, dtype=self.xp.int32)
        self.lost_position = self.xp.full(n_particles, -1, dtype=self.xp.float32)
        self.slice_id = self.xp.full(n_particles, -1, dtype=self.xp.int32)

        self.last_x = self.xp.zeros(n_particles, dtype=self.xp.float64) if is_cal_phase else None
        self.last_px = self.xp.zeros(n_particles, dtype=self.xp.float64) if is_cal_phase else None
        self.last_y = self.xp.zeros(n_particles, dtype=self.xp.float64) if is_cal_phase else None
        self.last_py = self.xp.zeros(n_particles, dtype=self.xp.float64) if is_cal_phase else None
        self.last_phasex = self.xp.zeros(n_particles, dtype=self.xp.float64) if is_cal_phase else None
        self.last_phasey = self.xp.zeros(n_particles, dtype=self.xp.float64) if is_cal_phase else None

    def copy(self, xp_target, fields=None):

        def convert(v):
            return convert_array(v, xp_target)

        new = ParticlePool.__new__(ParticlePool)
        new.xp = xp_target

        # Copy all data.
        # E.g. p_gpu = p.copy(cp)/p_cpu = p.copy(np)
        if fields is None:
            for k, v in self.__dict__.items():
                setattr(new, k, convert(v))
            return new

        # Copy a portion of the data.
        # E.g. p2 = p.copy(cp,fields=["x","px"])
        fields = set(fields)

        for k, v in self.__dict__.items():
            if k in fields:
                setattr(new, k, convert(v))
            else:
                setattr(new, k, v)

        return new


def convert_array(x, xp_target):
    if x is None:
        return None

    # GPU->CPU
    if isinstance(x, cp.ndarray) and xp_target is np:
        return x.get()

    # CPU->GPU
    if isinstance(x, np.ndarray) and xp_target is cp:
        return cp.asarray(x)

    # CPU->CPU and GPU->GPU
    return x
