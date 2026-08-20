from PASS.commands.command import Command
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.config import Config
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const
from PASS.utils.aperture import check_aperture_cpu

import numpy as np
import cupy as cp
import logging

logger = logging.getLogger(__name__)


@Command.register("rfcavity")
class RFCavity(Command):
    """
    RF cavity (高频加速腔).

    Applies a longitudinal RF kick to particles.  The energy gain depends
    on the particle's laboratory longitudinal position
    z_lab = z_rel + z_center:

        dE = (q/A) * V * sin(phase + phi_offset - h*z_lab/R)

    where R = C/(2*pi) is the machine radius.  The synchronous particle
    of each bunch (z_rel=0, z_lab=z_center) receives a bunch-dependent
    reference kick:

        dE_ref = (q/A) * V * sin(phase + phi_offset - h*z_center/R)

    phi_offset:
        A constant phase offset applied to all particles, shifting the
        RF waveform in time.  This is useful when multiple RF cavities
        share the same frequency but need different phase references
        (e.g. cavity spacing not an integer multiple of the RF
        wavelength, or multi-harmonic systems where each cavity needs
        an independent phase trim).  phi_offset rotates the entire
        sin curve; phase plus phi_offset defines the zero-azimuth
        reference phase.

    Moving reference frame:
        After the kick each bunch reference energy is updated to include
        dE_ref for that bunch center, so the bunch-center particle always
        sits at delta ~ 0 even if the RF harmonic is not an integer
        multiple of the beam grouping harmonic.
        Transverse momenta px, py are rescaled by beta0*gamma0 /
        (beta1*gamma1) to preserve the absolute transverse momentum
        (adiabatic damping of geometric emittance).

    The energy -> momentum -> delta conversion is exact (no linearisation):

        E_particle' = E_total0 + dE0 + dE_kick
        p'c         = sqrt(E_particle'^2 - (m0*c^2)^2)
        delta'      = p'/p0_new - 1

    Coordinate convention (PASS):
        x, px, y, py, z, dp(=delta)
        px = Px/P0,  py = Py/P0,  dp = (P-P0)/P0
        z  = s - beta0*c*t  (zeta coordinate)
    """

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.length = 0.0
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]

        # --- RF parameters (two input modes) ---
        # Mode 1 (fixed): voltage, harmonic, phase, phi_offset as scalars
        # Mode 2 (file):  tfs file with columns HARMONIC, VOLTAGE, PHASE, PHI_OFFSET
        #                 one row per turn (turn index starts from 0)
        self.voltage = kwargs.get("voltage (v)", 0.0)
        self.harmonic = int(kwargs.get("harmonic", 1))
        if self.harmonic < 1:
            raise ValueError(
                f"RF harmonic of {self.cmd_name} must be a positive integer, "
                f"got {self.harmonic}"
            )
        self.phase = kwargs.get("phase (rad)", 0.0)
        self.phi_offset = kwargs.get("phi offset (rad)", 0.0)

        # Beam harmonic_number is a bunch-grouping convention, not a
        # restriction on the RF harmonic.  The RF kick below always uses the
        # particle's laboratory azimuth z_lab = z_rel + z_center, so RF
        # harmonics that are not integer multiples of the grouping harmonic
        # are allowed and tracked bunch-by-bunch.

        # RF data file (ramping): tfs file with columns VOLTAGE, HARMONIC, PHASE, PHI_OFFSET
        rf_file = kwargs.get("rf data file", None)
        self._rf_table = None
        if rf_file is not None:
            self._rf_table = self._load_rf_table(rf_file)

        # --- switch ---
        self.is_enabled: bool = kwargs.get("is enabled", True)
        if not isinstance(self.is_enabled, bool):
            raise ValueError(
                f"is_enabled must be a boolean in {self.cmd_name}, "
                f"got {type(self.is_enabled)}"
            )

        # --- dp acceptance (longitudinal aperture) ---
        dp_aper = kwargs.get("dp aperture", None)
        if dp_aper is not None:
            self.dp_aperture_lower = float(dp_aper[0])
            self.dp_aperture_upper = float(dp_aper[1])
        else:
            self.dp_aperture_lower = -1.0
            self.dp_aperture_upper = 1.0

        # --- transverse aperture ---
        self.aperture_type: str = kwargs.get("aperture type", "off").lower()
        self.aperture_value: list = kwargs.get("aperture value", [])
        if not isinstance(self.aperture_value, list):
            raise ValueError(
                f"Aperture value of {self.cmd_name} must be a list, "
                f"but got {type(self.aperture_value)}"
            )

        super().__init__()

    # ------------------------------------------------------------------
    # Load RF ramping table from TFS file
    # ------------------------------------------------------------------

    @staticmethod
    def _load_rf_table(filepath):
        """Load RF ramping table from TFS file.

        Required columns (by name, order-independent, case-insensitive):
            HARMONIC, VOLTAGE, PHASE, PHI_OFFSET

        One row per turn (turn 0 = first data row).
        Column names are converted to lowercase internally, so any
        case (e.g. ``Harmonic``, ``voltage``, ``PHASE``) is accepted.
        The TFS file may also contain header metadata (title, etc.)
        which is automatically handled by tfs-pandas.
        """
        import tfs

        df = tfs.read(filepath)
        df.columns = df.columns.str.lower()

        required = ["harmonic", "voltage", "phase", "phi_offset"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"RF TFS file '{filepath}' missing required column(s): {missing}. "
                f"Required: {required}. Found: {list(df.columns)}"
            )
        return {
            "harmonic": df["harmonic"].to_numpy().astype(np.int64),
            "voltage": df["voltage"].to_numpy().astype(np.float64),
            "phase": df["phase"].to_numpy().astype(np.float64),
            "phi_offset": df["phi_offset"].to_numpy().astype(np.float64),
            "n_turns": len(df),
        }

    def _get_rf_params(self, turn):
        """Get RF parameters for a given turn (0-indexed)."""
        if self._rf_table is not None:
            idx = min(turn, self._rf_table["n_turns"] - 1)
            idx = max(idx, 0)
            harmonic = int(self._rf_table["harmonic"][idx])
            if harmonic < 1:
                raise ValueError(
                    f"RF data file harmonic at turn {idx} must be a positive "
                    f"integer, got {harmonic}"
                )
            return (
                self._rf_table["voltage"][idx],
                harmonic,
                self._rf_table["phase"][idx],
                self._rf_table["phi_offset"][idx],
            )
        return self.voltage, self.harmonic, self.phase, self.phi_offset

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------

    def print(self):
        set_simple_logging()
        if self._rf_table is not None:
            logger.info(
                f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, "
                f"RFDataFile(turns={self._rf_table['n_turns']}), "
                f"IsEnabled={self.is_enabled}, "
                f"DpAperture=[{self.dp_aperture_lower}, {self.dp_aperture_upper}], "
                f"ApertureType={self.aperture_type:s}, "
                f"ApertureValue={self.aperture_value}"
            )
        else:
            logger.info(
                f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, "
                f"Voltage={self.voltage:.6e} V, Harmonic={self.harmonic}, "
                f"Phase={self.phase:.6f} rad, PhiOffset={self.phi_offset:.6f} rad, "
                f"IsEnabled={self.is_enabled}, "
                f"DpAperture=[{self.dp_aperture_lower}, {self.dp_aperture_upper}], "
                f"ApertureType={self.aperture_type:s}, "
                f"ApertureValue={self.aperture_value}"
            )
        set_normal_logging()

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def execute_cpu(self, sim):
        if not self.is_enabled:
            return

        beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches
        turn = sim.state.turn

        for i, bunch in enumerate(bunches):
            self._track_rf_cpu(beam, bunch, turn)
            check_aperture_cpu(
                beam,
                bunch,
                self.aperture_type,
                self.aperture_value,
                self.s,
                turn,
            )

    def execute_gpu(self, sim):
        raise NotImplementedError(
            "GPU implementation of RFCavity is not yet available"
        )

    # ------------------------------------------------------------------
    # Core RF tracking (CPU)
    # ------------------------------------------------------------------

    def _track_rf_cpu(self, beam: Beam, bunch: BunchInfo, turn: int):
        """Track particles through the RF cavity (thin-lens kick).

        Physics:
          1. Each particle's RF phase depends on its longitudinal position z.
          2. Energy kick is applied exactly (no linearisation).
          3. The bunch reference is updated (moving frame).
          4. Transverse momenta are rescaled (adiabatic damping).
        """
        voltage, harmonic, phase, phi_offset = self._get_rf_params(turn)

        if abs(voltage) < const.eps:
            return  # no RF, nothing to do

        beta0 = bunch.beta
        gamma0 = bunch.gamma
        circum = bunch.circum
        m0 = bunch.m0          # rest mass per nucleon (eV/c^2)
        qm_ratio = bunch.qm_ratio   # |q|/A
        Ek0 = bunch.Ek          # kinetic energy per nucleon (eV/u)
        p0_old = bunch.p0       # old reference momentum per nucleon (eV/c)

        E_total0 = Ek0 + m0     # old total energy per nucleon (eV)
        radius = circum / (2.0 * const.pi)

        start = bunch.start_idx
        end = bunch.end_idx

        p = beam.particles
        x = p.x[start:end]
        px = p.px[start:end]
        y = p.y[start:end]
        py = p.py[start:end]
        z = p.z[start:end]
        dp = p.dp[start:end]
        tag = p.tag[start:end]

        alive_before = tag > 0

        # --- 1. RF phase for each particle ---
        # phi = phase + phi_offset - h * z_lab / R
        # This is equivalent to phase - omega*tau, since
        #   omega*tau = 2*pi*f * z/(beta0*c) = 2*pi*h*z/C = h*z/R
        #
        # z is the bunch-relative coordinate; the laboratory position is
        # z_lab = z + z_center (explicitly, like the Exciter).  The beam's
        # harmonic_number is only a grouping convention; it does not need to
        # divide this cavity's RF harmonic.  Therefore the bunch reference
        # particle must use its own center phase below.
        z_lab = z + bunch.z_center
        # Reduce mod C/h before evaluating sin (robust with unwrapped z).
        rf_period = circum / harmonic
        z_phase = z_lab - rf_period * np.rint(z_lab / rf_period)
        theta = z_phase / radius
        phi_particle = phase + phi_offset - harmonic * theta

        # --- 2. Energy kick ---
        # dE_kick = (q/A) * V * sin(phi_particle)   [eV/u]
        dE_kick = qm_ratio * voltage * np.sin(phi_particle)

        # --- 3. Bunch-reference energy gain ---
        # The moving reference frame follows this bunch's ideal particle at
        # z_rel=0, i.e. z_lab=z_center.  For non-integer relationships between
        # the RF harmonic and the grouping harmonic, different bunch centers
        # can see different RF phases, so dE_ref is bunch-dependent.
        z_center_phase = bunch.z_center - rf_period * np.rint(
            bunch.z_center / rf_period
        )
        phi_center = (
            phase + phi_offset - harmonic * z_center_phase / radius
        )
        dE_ref = qm_ratio * voltage * np.sin(phi_center)  # scalar [eV/u]

        # --- 4. New reference energy ---
        E_total1 = E_total0 + dE_ref  # new total energy per nucleon (eV)
        gamma1 = E_total1 / m0
        beta1 = np.sqrt(1.0 - 1.0 / (gamma1 * gamma1))
        p0_new = gamma1 * m0 * beta1  # new reference momentum per nucleon (eV/c)
        Ek1 = E_total1 - m0           # new kinetic energy per nucleon (eV/u)

        # --- 5. Exact energy -> momentum -> delta conversion ---
        # All quantities in natural units (c=1): m0 [eV], p0 [eV], E [eV].
        # E^2 = p^2 + m0^2  (exact relativistic energy-momentum relation)
        #
        # Before kick:  p_old = p0_old * (1 + delta)
        #               E_old = sqrt(p_old^2 + m0^2)
        # After kick:   E_new = E_old + dE_kick
        #               p_new = sqrt(E_new^2 - m0^2)
        #               delta_new = p_new / p0_new - 1

        p_particle_old = p0_old * (1.0 + dp)  # eV (natural units)
        E_particle_old = np.sqrt(p_particle_old**2 + m0**2)  # eV

        # After kick
        E_particle_new = E_particle_old + dE_kick * alive_before  # eV

        # Guard against negative or unphysical energies
        E_particle_new_safe = np.maximum(E_particle_new, const.eps)

        # New momentum (exact relativistic)
        p_particle_new = np.sqrt(
            E_particle_new_safe**2 - m0**2
        )  # eV/c (natural units)

        # New delta relative to new reference
        dp_new = p_particle_new / p0_new - 1.0

        # Apply mask (dead particles unchanged)
        dp[:] = np.where(alive_before, dp_new, dp)

        # --- 6. Transverse momentum rescaling (adiabatic damping) ---
        # px_new = px * (p0_old / p0_new) = px * (beta0*gamma0) / (beta1*gamma1)
        trans_scale = p0_old / p0_new  # = beta0*gamma0 / (beta1*gamma1)
        px[:] = np.where(alive_before, px * trans_scale, px)
        py[:] = np.where(alive_before, py * trans_scale, py)

        # z is unchanged (thin-lens kick, no drift)

        # --- 7. Update bunch reference (moving frame) ---
        bunch.Ek = Ek1
        bunch.gamma = gamma1
        bunch.beta = beta1
        bunch.p0 = p0_new
        p0_kg_new = gamma1 * (m0 * const.e / (const.c * const.c)) * beta1 * const.c
        bunch.p0_kg = p0_kg_new
        bunch.brho = p0_kg_new / (qm_ratio * const.e)

        # --- 8. dp aperture check (longitudinal acceptance) ---
        dp_outside = (dp < self.dp_aperture_lower) | (
            dp > self.dp_aperture_upper
        )
        newly_lost_dp = alive_before & dp_outside
        if np.any(newly_lost_dp):
            tag[newly_lost_dp] = -np.abs(tag[newly_lost_dp])

        # --- 9. Update lost particle info ---
        newly_lost = alive_before & (tag < 0)
        if np.any(newly_lost):
            lost_position = p.lost_position[start:end]
            lost_turn = p.lost_turn[start:end]
            lost_position[newly_lost] = self.s
            lost_turn[newly_lost] = turn
