"""Phase-constrained interpolation of uncoupled, paraxial ring optics.

On each source interval, a quintic Hermite polynomial matches mu, mu' and
mu'' at both ends. In cycles, mu'=1/(2*pi*beta), mu''=alpha/(pi*beta**2).
Deriving beta and alpha from that same polynomial preserves their differential
relations as well as every source phase. No extrapolation or phase wrapping is
performed. Repeated S rows retain separate incoming/outgoing optical states.
"""

import numpy as np
from scipy.interpolate import BPoly, CubicHermiteSpline, PPoly


OPTICS_COLUMNS = ("BETX", "ALFX", "MUX", "BETY", "ALFY", "MUY", "DX", "DPX")


class RingTwissInterpolator:
    """Interpolate a complete ring TFS table, without changing its source data."""

    def __init__(self, table):
        self.circumference = float(table.headers["LENGTH"])
        if not np.isfinite(self.circumference) or self.circumference <= 0:
            raise ValueError("TFS LENGTH must be finite and positive.")
        s = table["S"].to_numpy(dtype=float, copy=True)
        optics = table[list(OPTICS_COLUMNS)].to_numpy(dtype=float, copy=True)
        if len(s) < 2 or not np.all(np.isfinite(s)) or not np.all(np.isfinite(optics)):
            raise ValueError("TFS needs at least two rows with finite S and optical functions.")
        if np.any(optics[:, (0, 3)] <= 0):
            raise ValueError("TFS BETX and BETY must be positive.")
        self.position_tolerance = 32 * np.finfo(float).eps * max(1., self.circumference)
        tol = self.position_tolerance
        if np.any(np.diff(s) < -tol):
            raise ValueError("TFS S must follow the ring in nondecreasing order.")
        if abs(s[0]) > tol or abs(s[-1] - self.circumference) > tol:
            raise ValueError("等间距插值需要覆盖 s=0 到 LENGTH 的完整 TFS；请导出起终点，不能外推补齐。")
        s[0], s[-1] = 0., self.circumference
        starts = [0]
        for i in range(1, len(s)):
            if abs(s[i] - s[starts[-1]]) > tol:
                starts.append(i)
        ends = np.r_[np.asarray(starts[1:]) - 1, len(s)-1]
        self.s = s[starts]
        self.left = optics[starts]
        self.right = optics[ends]
        if len(self.s) < 2 or np.any(np.diff(self.s) <= 0):
            raise ValueError("TFS must contain at least two distinct S positions.")

        # All states at the same position must have the same cumulative phase.
        # Their alpha/dispersion can differ across a thin lens.
        for first, last in zip(starts, ends):
            if not np.allclose(optics[first:last+1, (2, 5)], optics[first, (2, 5)], rtol=0, atol=1e-12):
                raise ValueError(f"Repeated S={s[first]:.12g} has different phases; cannot resolve a zero-length phase advance.")

        self.tunes = self.right[-1, (2, 5)] - self.left[0, (2, 5)]
        for plane, tune, header in zip(("x", "y"), self.tunes, ("Q1", "Q2")):
            supplied = float(table.headers[header])
            if not np.isfinite(supplied) or supplied <= 0 or tune <= 0:
                raise ValueError(f"TFS {header} and cumulative {plane} phase advance must be positive and finite.")
            if not np.isclose(tune, supplied, rtol=2e-8, atol=2e-9):
                raise ValueError(f"TFS {header}={supplied:.12g} disagrees with cumulative phase span {tune:.12g}; use unwrapped full-ring phases.")

        self._phase = [[], []]
        self._dispersion = []
        for i, h in enumerate(np.diff(self.s)):
            a, b = self.right[i], self.left[i+1]
            for plane, offset in enumerate((0, 3)):
                beta0, alpha0, mu0 = a[offset:offset+3]
                beta1, alpha1, mu1 = b[offset:offset+3]
                if mu1 <= mu0:
                    raise ValueError(f"TFS phase must increase between S={self.s[i]:.12g} and {self.s[i+1]:.12g}.")
                # Local phase offsets and a unit interval avoid large-S/large-Q
                # cancellation when differentiating the polynomial.
                data = [[0., h/(2*np.pi*beta0), h*h*alpha0/(np.pi*beta0**2)],
                        [mu1-mu0, h/(2*np.pi*beta1), h*h*alpha1/(np.pi*beta1**2)]]
                if not np.all(np.isfinite(data)):
                    raise ValueError("TFS optical scales exceed finite interpolation precision.")
                p = BPoly.from_derivatives([0., 1.], data, extrapolate=False)
                # Check the entire quartic phase derivative, including every
                # internal extremum; checking output samples alone is unsafe.
                extrema = PPoly.from_bernstein_basis(p.derivative(2)).roots(extrapolate=False)
                probes = np.r_[0., extrema[(extrema > 0) & (extrema < 1)], 1.]
                rate = p(probes, 1)
                if not np.all(np.isfinite(rate)) or np.min(rate) <= 0:
                    axis = "x" if plane == 0 else "y"
                    raise ValueError(
                        f"{axis} 平面 S=[{self.s[i]:.12g}, {self.s[i+1]:.12g}] 的高阶插值相位不单调。"
                        "请增加此区间的原始 MAD-X 采样密度；增加输出点数不能修复源数据。")
                self._phase[plane].append(p)
            # In the supported on-reference, paraxial convention DPX is the
            # s derivative of DX. Retain the TFS dispersion normalization.
            self._dispersion.append(CubicHermiteSpline(
                [0., 1.], [a[6], b[6]], [h*a[7], h*b[7]], extrapolate=False))

    @property
    def discontinuities(self):
        """Positions whose incoming and outgoing optical states differ."""
        changed = ~np.all(np.isclose(self.left, self.right, rtol=1e-12, atol=1e-14), axis=1)
        return self.s[changed]

    def at(self, position: float, side: str = "right") -> np.ndarray:
        """Return BETX, ALFX, MUX, BETY, ALFY, MUY, DX, DPX at one position."""
        tol = self.position_tolerance
        if not np.isfinite(position) or position < -tol or position > self.circumference+tol:
            raise ValueError("Requested S lies outside the TFS ring.")
        position = min(max(position, 0.), self.circumference)
        index = int(np.searchsorted(self.s, position))
        for knot in (index, index-1):
            if 0 <= knot < len(self.s) and abs(self.s[knot]-position) <= self.position_tolerance:
                return (self.left if side == "left" else self.right)[knot].copy()
        i = index-1
        h = self.s[i+1]-self.s[i]
        t = (position-self.s[i])/h
        result = np.empty(8)
        for plane, offset in enumerate((0, 3)):
            p = self._phase[plane][i]
            rate, acceleration = p(t, 1), p(t, 2)
            result[offset:offset+3] = (h/(2*np.pi*rate),
                                      acceleration/(4*np.pi*rate**2),
                                      self.right[i, offset+2]+p(t))
        d = self._dispersion[i]
        result[6:] = d(t), d(t, 1)/h
        if not np.all(np.isfinite(result)) or np.any(result[[0, 3]] <= 0):
            raise ValueError(f"Nonfinite or nonpositive interpolated optics at S={position:.12g}.")
        return result


def resample_madx_twiss(twiss_file, num_interp_slice, error_file, muz, dqx, dqy,
                       is_field_error, insert_patterns, longitudinal_transfer, interp_kind):
    """Build a uniform Twiss sequence with splits at kicks and optical jumps."""
    import tfs
    from PASS.commands import command_priority
    from PASS.para.madx import _insert_elements, _make_match_key, read_madx_errors
    from PASS.para.schema.elements import MultipoleElement
    from PASS.para.schema.twiss import TwissPoint

    if isinstance(num_interp_slice, (bool, np.bool_)) or not isinstance(num_interp_slice, (int, np.integer)) or num_interp_slice < 2:
        raise ValueError("num_interp_slice must be an integer >= 2 (including 0 and C).")
    if interp_kind != "phase_hermite":
        raise ValueError("Use interp_kind='phase_hermite'; independent per-column interpolation is no longer supported.")
    if longitudinal_transfer not in ("off", "drift", "matrix"):
        raise ValueError("Invalid longitudinal_transfer; use off, drift or matrix.")
    table = tfs.read(twiss_file)
    optics = RingTwissInterpolator(table)
    circumference = optics.circumference
    dqx = float(table.headers["DQ1"] if dqx == "from_file" else dqx)
    dqy = float(table.headers["DQ2"] if dqy == "from_file" else dqy)
    muz = float(muz)
    if not np.all(np.isfinite([dqx, dqy, muz])):
        raise ValueError("DQx, DQy and Mu z must be finite.")
    if longitudinal_transfer != "matrix":
        muz = 0.

    extras, extra_names = [], []
    if insert_patterns:
        extras, extra_names = _insert_elements(table, insert_patterns)
    if is_field_error:
        if not error_file:
            raise ValueError("附加场误差需要选择误差 TFS 文件。")
        positions, occurrences = {}, {}
        for _, row in table.iterrows():
            name = str(row["NAME"])
            occurrences[name] = occurrences.get(name, 0) + 1
            positions[_make_match_key(name, occurrences[name])] = float(row["S"])
        for key, errors in read_madx_errors(error_file).items():
            if key not in positions:
                raise ValueError(f"Field-error element {key!r} is missing from the source TFS.")
            extras.append(MultipoleElement(s=positions[key], length=0., knl=errors["knl"], ksl=errors["ksl"]))
            extra_names.append(f"{key}_error")

    base = np.linspace(0., circumference, num_interp_slice)
    required = np.unique(np.r_[optics.discontinuities, [item.s for item in extras]])
    # Snap coincident grid positions to the exact kick S, without moving kicks.
    for s in required:
        nearest = int(np.argmin(np.abs(base-s)))
        if abs(base[nearest]-s) <= optics.position_tolerance:
            base[nearest] = s
    positions = np.unique(np.r_[base, required])
    items, names = [], []
    fields = ("beta_x", "alpha_x", "mu_x", "beta_y", "alpha_y", "mu_y", "dx", "dpx")

    def append_transport(s, previous_s, current, previous):
        values = {key: float(value) for key, value in zip(fields, current)}
        values.update({key+"_previous": float(value) for key, value in zip(fields, previous)})
        item = TwissPoint(
            s=float(s), s_previous=float(previous_s), **values,
            mu_z=s/circumference*muz, mu_z_previous=previous_s/circumference*muz,
            dqx=dqx*(current[2]-previous[2])/optics.tunes[0],
            dqy=dqy*(current[5]-previous[5])/optics.tunes[1],
            longitudinal_transfer=longitudinal_transfer)
        names.append(f"twiss_interp_{len(items):06d}")
        items.append(item)

    previous_s, previous = 0., optics.at(0., "left")
    for s in positions:
        incoming, outgoing = optics.at(s, "left"), optics.at(s, "right")
        append_transport(s, previous_s, incoming, previous)
        if not np.allclose(incoming, outgoing, rtol=1e-12, atol=1e-14):
            # The source optical jump is part of the base map; explicit
            # kicks/errors are additional and execute after the Twiss maps.
            append_transport(s, s, outgoing, incoming)
        previous_s, previous = s, outgoing
    items.extend(extras)
    names.extend(extra_names)
    used = set()
    for i, name in enumerate(names):
        candidate, suffix = name, 2
        while candidate in used:
            candidate, suffix = f"{name}_{suffix}", suffix+1
        names[i] = candidate
        used.add(candidate)
    ordered = sorted(zip(items, names), key=lambda pair: (pair[0].s, command_priority(pair[0].command)))
    items, names = map(list, zip(*ordered))
    print(f"[Read MADX Twiss] {num_interp_slice} base points, {len(positions)-num_interp_slice} extra positions, "
          f"{len(items)} commands; phase-constrained quintic Hermite; C={circumference}")
    return items, names, circumference
