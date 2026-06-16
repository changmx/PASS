class ParticlePool:

    def __init__(self, n_particles: int, xp, is_cal_phase: bool = True):

        self.x = xp.zeros(n_particles, dtype=xp.float64)
        self.px = xp.zeros(n_particles, dtype=xp.float64)
        self.y = xp.zeros(n_particles, dtype=xp.float64)
        self.py = xp.zeros(n_particles, dtype=xp.float64)
        self.z = xp.zeros(n_particles, dtype=xp.float64)
        self.pz = xp.zeros(n_particles, dtype=xp.float64)
        self.tag = xp.range(1, 1 + n_particles, dtype=xp.int32)
        self.lost_turn = xp.full(n_particles, -1, dtype=xp.int32)
        self.slice_id = xp.full(n_particles, -1, dtype=xp.int32)

        self.last_x = xp.zeros(n_particles, dtype=xp.float64) if is_cal_phase else None
        self.last_px = xp.zeros(n_particles, dtype=xp.float64) if is_cal_phase else None
        self.last_y = xp.zeros(n_particles, dtype=xp.float64) if is_cal_phase else None
        self.last_py = xp.zeros(n_particles, dtype=xp.float64) if is_cal_phase else None
        self.last_phasex = xp.zeros(n_particles, dtype=xp.float64) if is_cal_phase else None
        self.last_phasey = xp.zeros(n_particles, dtype=xp.float64) if is_cal_phase else None
