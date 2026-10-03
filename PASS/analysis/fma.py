"""Portable two-window frequency drift diagnostics."""


def compute_fma(x,
                y,
                *,
                windows=None,
                sample_spacing=1.0,
                method="refined_fft",
                window="hann",
                padding_factor=8,
                interpolation="parabolic",
                frequency_range_x=None,
                frequency_range_y=None,
                remove_mean=True,
                frequency_estimator=None):
    """Compare transverse frequencies in two equal, disjoint turn intervals.

    This function is independently copyable and requires only NumPy. Input
    arrays have the same shape ``(..., turns)``; no rows are mixed or cleaned
    by ensemble tune. Default windows are ``(0, N//2), (N//2, 2*(N//2))``;
    an odd trailing sample is unused. Explicit intervals are half-open,
    equally long, ordered, disjoint and in bounds. FFT requires at least two
    samples per interval; refined FFT requires four. The embedded refined
    estimator matches ``compute_refined_fft`` without importing PASS.

    Frequencies have units cycles per sample-spacing unit, so they are tunes
    when samples are consecutive turns and sample_spacing=1. Real position
    traces cover the nonnegative Nyquist range: they cannot distinguish Q
    from 1-Q. Complex normalized coordinates retain signed frequencies.
    Differences are second minus first, without wrapping. ``drift`` is their
    Euclidean norm and ``diffusion_log10`` is log10(drift), with exactly zero
    drift represented by -inf. This finite-window indicator is not a physical
    diffusion coefficient; acceleration and collective evolution can cause
    drift without chaos. Frequencies must represent the same mode in both
    intervals for a meaningful comparison.

    Nonfinite samples (including NaNs used for lost particles) invalidate only
    the affected row/window. Missing samples are never filled. Invalid drift
    outputs are NaN and ``quality`` explains the window(s); valid rows may
    retain boundary-peak warnings. Frequency arrays keep all leading axes.

    An optional ``frequency_estimator`` overrides the embedded estimator. It
    is called separately for each finite 1D interval as
    ``estimator(samples, sample_spacing=..., frequency_range=...,
    remove_mean=...)``. Configure its other parameters through a wrapper. It
    must return a real scalar frequency or a dictionary with one-element
    ``peak_frequency`` and optional ``valid``/``quality``. Returned invalid
    estimates are flagged; estimator exceptions propagate. ``method`` still
    sets the interval's minimum length. The caller owns custom frequency and
    phase conventions.
    """
    import numpy as np

    # Embedded estimator is kept identical to the portable refined FFT function.
    def _estimate_refined_fft(signal,
                              sample_spacing=1.0,
                              *,
                              axis=-1,
                              window="hann",
                              padding_factor=8,
                              interpolation="parabolic",
                              frequency_range=None,
                              n_peaks=1,
                              remove_mean=False,
                              return_spectrum=True):
        """Estimate independent spectral peaks using only NumPy.

        Copying this function alone is supported. Samples are uniformly spaced;
        ``sample_spacing`` sets frequency units. Windows are symmetric rectangle,
        Hann, Hamming or Blackman. Coefficients are normalized by the sum of the
        window, never by the padded FFT length. Real spectra use peak-amplitude
        one-sided coefficients; complex spectra are signed and two-sided.

        Peaks are local amplitude maxima ranked by amplitude in the inclusive
        ``frequency_range=(lower, upper)``. Log-amplitude parabolic interpolation
        uses three neighboring FFT samples; ``none`` returns sampled frequencies.
        A direct windowed Fourier sum recomputes amplitude and phase at each
        estimated frequency. This is not NAFF, and padding does not add signal
        information or guarantee an accuracy. Nearby lines, leakage, DC/Nyquist
        overlap and changing frequencies can bias the estimates.

        Peak arrays have shape ``signal.shape`` without ``axis`` plus
        ``(n_peaks,)``. Invalid peaks contain NaN with ``valid=False``. Quality is
        ``ok``, ``boundary_peak`` (no interpolation at a search/spectrum edge),
        ``flat_peak``, ``zero_signal`` or ``insufficient_peaks``. Valid marks a
        detected finite peak, not a physical mode identification. Spectrum arrays
        are returned only when requested, with the frequency axis last. Phase is
        in radians at the first input sample. Nonfinite samples raise ValueError.
        """
        import numpy as np

        if isinstance(sample_spacing, (bool, np.bool_)) or not np.isscalar(sample_spacing):
            raise ValueError("sample_spacing must be a finite positive number")
        try:
            sample_spacing = float(sample_spacing)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("sample_spacing must be a finite positive number") from exc
        if not np.isfinite(sample_spacing) or sample_spacing <= 0:
            raise ValueError("sample_spacing must be a finite positive number")
        if not isinstance(remove_mean, (bool, np.bool_)) or not isinstance(return_spectrum, (bool, np.bool_)):
            raise ValueError("remove_mean and return_spectrum must be booleans")
        for name, value in (("padding_factor", padding_factor), ("n_peaks", n_peaks)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if window not in ("rectangle", "hann", "hamming", "blackman"):
            raise ValueError("window must be rectangle, hann, hamming or blackman")
        if interpolation not in ("none", "parabolic"):
            raise ValueError("interpolation must be none or parabolic")
        values = np.asarray(signal)
        if values.ndim == 0:
            raise ValueError("signal must have a sample axis, not be a scalar")
        if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)) or not -values.ndim <= axis < values.ndim:
            raise ValueError("axis must identify an existing signal axis")
        try:
            values = np.asarray(values, dtype=np.complex128 if np.iscomplexobj(values) else np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("signal must contain numeric samples") from exc
        values = np.moveaxis(values, axis, -1)
        n_samples = values.shape[-1]
        if n_samples < 4:
            raise ValueError("refined FFT requires at least 4 samples")
        if not np.all(np.isfinite(values)):
            raise ValueError("signal contains nonfinite samples; select a complete finite interval")
        if remove_mean:
            values = values - np.mean(values, axis=-1, keepdims=True)
        window_values = {
            "rectangle": np.ones,
            "hann": np.hanning,
            "hamming": np.hamming,
            "blackman": np.blackman,
        }[window](n_samples)
        window_sum = np.sum(window_values)
        weighted = values * window_values
        n_fft = n_samples * int(padding_factor)
        is_real = not np.iscomplexobj(values)
        if is_real:
            frequency = np.fft.rfftfreq(n_fft, d=sample_spacing)
            coefficients = np.fft.rfft(weighted, n=n_fft, axis=-1) / window_sum
            coefficients[..., 1:-1 if n_fft % 2 == 0 else None] *= 2
        else:
            frequency = np.fft.fftshift(np.fft.fftfreq(n_fft, d=sample_spacing))
            coefficients = np.fft.fftshift(np.fft.fft(weighted, n=n_fft, axis=-1), axes=-1) / window_sum
        selection = np.ones(frequency.size, dtype=bool)
        if frequency_range is not None:
            try:
                bounds = np.asarray(frequency_range, dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("frequency_range must contain two finite increasing bounds") from exc
            if bounds.shape != (2, ) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
                raise ValueError("frequency_range must contain two finite increasing bounds")
            selection = (frequency >= bounds[0]) & (frequency <= bounds[1])
            if not np.any(selection):
                raise ValueError("frequency_range contains no sampled FFT frequencies")
        amplitude = np.abs(coefficients)
        leading_shape = values.shape[:-1]
        peak_shape = leading_shape + (int(n_peaks), )
        peak_frequency = np.full(peak_shape, np.nan)
        peak_amplitude = np.full(peak_shape, np.nan)
        peak_phase = np.full(peak_shape, np.nan)
        valid = np.zeros(peak_shape, dtype=bool)
        quality = np.full(peak_shape, "insufficient_peaks", dtype="<U24")
        rows = int(np.prod(leading_shape)) if leading_shape else 1
        row_values = weighted.reshape(rows, n_samples)
        row_amplitude = amplitude.reshape(rows, frequency.size)
        output_frequency = peak_frequency.reshape(rows, n_peaks)
        output_amplitude = peak_amplitude.reshape(rows, n_peaks)
        output_phase = peak_phase.reshape(rows, n_peaks)
        output_valid = valid.reshape(rows, n_peaks)
        output_quality = quality.reshape(rows, n_peaks)
        spacing = 1.0 / (n_fft * sample_spacing)
        sample_times = np.arange(n_samples) * sample_spacing
        selected_indices = np.flatnonzero(selection)
        for row_index in range(rows):
            magnitudes = row_amplitude[row_index].copy()
            if is_real:
                # Detect extrema before one-sided doubling to avoid false edge peaks.
                magnitudes[1:-1 if n_fft % 2 == 0 else None] *= 0.5
            scale = np.max(magnitudes)
            if scale == 0:
                output_quality[row_index] = "zero_signal"
                continue
            threshold = scale * 64 * np.finfo(float).eps
            selected_magnitudes = magnitudes[selected_indices]
            maxima = selected_magnitudes > threshold
            maxima[1:] &= selected_magnitudes[1:] >= selected_magnitudes[:-1]
            maxima[:-1] &= selected_magnitudes[:-1] > selected_magnitudes[1:]
            candidates = selected_indices[maxima]
            candidates = candidates[np.argsort(-row_amplitude[row_index, candidates], kind="stable")]
            for peak_index, index in enumerate(candidates[:n_peaks]):
                estimate = frequency[index]
                peak_quality = "ok"
                edge = index == selected_indices[0] or index == selected_indices[-1]
                if edge:
                    peak_quality = "boundary_peak"
                elif interpolation == "parabolic":
                    log_values = np.log(np.maximum(magnitudes[index - 1:index + 2], np.finfo(float).tiny))
                    denominator = log_values[0] - 2 * log_values[1] + log_values[2]
                    if denominator < 0:
                        shift = np.clip(0.5 * (log_values[0] - log_values[2]) / denominator, -0.5, 0.5)
                        estimate += shift * spacing
                    else:
                        peak_quality = "flat_peak"
                coefficient = np.sum(row_values[row_index] * np.exp(-2j * np.pi * estimate * sample_times)) / window_sum
                if is_real and estimate > 0 and not np.isclose(estimate, 0.5 / sample_spacing, rtol=0, atol=spacing * 1e-10):
                    coefficient *= 2
                output_frequency[row_index, peak_index] = estimate
                output_amplitude[row_index, peak_index] = abs(coefficient)
                output_phase[row_index, peak_index] = np.angle(coefficient)
                output_valid[row_index, peak_index] = True
                output_quality[row_index, peak_index] = peak_quality
        result = {
            "peak_frequency": peak_frequency,
            "peak_amplitude": peak_amplitude,
            "peak_phase": peak_phase,
            "valid": valid,
            "quality": quality,
            "metadata": {
                "method": "refined_fft",
                "n_samples": n_samples,
                "n_fft": n_fft,
                "sample_spacing": sample_spacing,
                "frequency_units": "cycles per sample-spacing unit",
                "phase_reference": "first input sample",
                "spectrum": "one_sided_real" if is_real else "two_sided_complex",
                "coefficient_normalization": "window sum; real interior positive frequencies doubled",
                "window": window,
                "window_sum": float(window_sum),
                "padding_factor": int(padding_factor),
                "interpolation": interpolation,
                "frequency_range": None if frequency_range is None else bounds.tolist(),
                "frequency_axis": -1,
                "remove_mean": bool(remove_mean),
            },
        }
        if return_spectrum:
            result.update(frequency=frequency, coefficients=coefficients, amplitude=amplitude, phase=np.angle(coefficients))
        return result

    if method not in ("fft", "refined_fft"):
        raise ValueError("method must be fft or refined_fft")
    if frequency_estimator is not None and not callable(frequency_estimator):
        raise ValueError("frequency_estimator must be callable or None")
    if isinstance(sample_spacing, (bool, np.bool_)) or not np.isscalar(sample_spacing):
        raise ValueError("sample_spacing must be a finite positive number")
    try:
        sample_spacing = float(sample_spacing)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("sample_spacing must be a finite positive number") from exc
    if not np.isfinite(sample_spacing) or sample_spacing <= 0:
        raise ValueError("sample_spacing must be a finite positive number")
    if not isinstance(remove_mean, (bool, np.bool_)):
        raise ValueError("remove_mean must be a boolean")
    try:
        x_values = np.asarray(x)
        y_values = np.asarray(y)
        x_values = np.asarray(x_values, dtype=np.complex128 if np.iscomplexobj(x_values) else np.float64)
        y_values = np.asarray(y_values, dtype=np.complex128 if np.iscomplexobj(y_values) else np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("x and y must contain numeric samples") from exc
    if x_values.ndim == 0 or x_values.shape != y_values.shape:
        raise ValueError("x and y must have matching shape (..., turns)")
    n_samples = x_values.shape[-1]
    if windows is None:
        half = n_samples // 2
        intervals = ((0, half), (half, 2 * half))
    else:
        try:
            entries = np.asarray(windows)
        except (TypeError, ValueError) as exc:
            raise ValueError("windows must be two integer half-open intervals") from exc
        if entries.shape != (2, 2) or entries.dtype.kind not in "iu":
            raise ValueError("windows must be two integer half-open intervals")
        intervals = tuple(tuple(int(value) for value in pair) for pair in entries)
    (first_start, first_end), (second_start, second_end) = intervals
    minimum = 4 if method == "refined_fft" else 2
    interval_size = first_end - first_start
    if (first_start < 0 or first_end > second_start or second_end > n_samples or interval_size != second_end - second_start
            or interval_size < minimum):
        raise ValueError(f"windows must be ordered, disjoint, equal-length in-bounds intervals with at least {minimum} samples each")

    def _estimate_fft(samples, frequency_range):
        values = samples - np.mean(samples) if remove_mean else samples
        is_real = not np.iscomplexobj(values)
        if is_real:
            frequency = np.fft.rfftfreq(len(values), d=sample_spacing)
            coefficients = np.fft.rfft(values) / len(values)
            coefficients[1:-1 if len(values) % 2 == 0 else None] *= 2
        else:
            frequency = np.fft.fftshift(np.fft.fftfreq(len(values), d=sample_spacing))
            coefficients = np.fft.fftshift(np.fft.fft(values)) / len(values)
        selection = np.ones(len(frequency), dtype=bool)
        if frequency_range is not None:
            try:
                bounds = np.asarray(frequency_range, dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("frequency_range must contain two finite increasing bounds") from exc
            if bounds.shape != (2, ) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
                raise ValueError("frequency_range must contain two finite increasing bounds")
            selection = (frequency >= bounds[0]) & (frequency <= bounds[1])
            if not np.any(selection):
                raise ValueError("frequency_range contains no sampled FFT frequencies")
        indices = np.flatnonzero(selection)
        index = indices[np.argmax(np.abs(coefficients[indices]))]
        if abs(coefficients[index]) == 0:
            return {"peak_frequency": np.nan, "valid": False, "quality": "zero_signal"}
        quality = "boundary_peak" if index in (indices[0], indices[-1]) else "ok"
        return {"peak_frequency": frequency[index], "valid": True, "quality": quality}

    def _estimate(samples, frequency_range):
        if frequency_estimator is not None:
            return frequency_estimator(samples, sample_spacing=sample_spacing, frequency_range=frequency_range, remove_mean=remove_mean)
        if method == "fft":
            return _estimate_fft(samples, frequency_range)
        return _estimate_refined_fft(samples,
                                     sample_spacing,
                                     window=window,
                                     padding_factor=padding_factor,
                                     interpolation=interpolation,
                                     frequency_range=frequency_range,
                                     remove_mean=remove_mean,
                                     return_spectrum=False)

    # Validate the configured default estimator even when every input row is lost.
    if frequency_estimator is None:
        _estimate(np.zeros(interval_size, dtype=x_values.dtype), frequency_range_x)
        _estimate(np.zeros(interval_size, dtype=y_values.dtype), frequency_range_y)
    leading_shape = x_values.shape[:-1]
    rows = int(np.prod(leading_shape)) if leading_shape else 1
    planes = (x_values.reshape(rows, n_samples), y_values.reshape(rows, n_samples))
    frequencies = np.full((4, rows), np.nan)
    accepted = np.zeros((4, rows), dtype=bool)
    messages = [[] for _ in range(rows)]
    labels = ("x_first", "y_first", "x_second", "y_second")
    for interval_index, (start, end) in enumerate(intervals):
        for plane_index, (plane, frequency_range) in enumerate(zip(planes, (frequency_range_x, frequency_range_y))):
            output_index = 2 * interval_index + plane_index
            label = labels[output_index]
            for row_index in range(rows):
                samples = plane[row_index, start:end]
                if not np.all(np.isfinite(samples)):
                    messages[row_index].append("nonfinite_" + label)
                    continue
                estimate = _estimate(samples, frequency_range)
                if isinstance(estimate, dict):
                    if "peak_frequency" not in estimate:
                        raise ValueError("frequency_estimator dictionary must contain peak_frequency")
                    estimate_frequency = np.asarray(estimate["peak_frequency"])
                    estimate_valid = np.asarray(estimate.get("valid", True))
                    estimate_quality = np.asarray(estimate.get("quality", "ok"))
                else:
                    estimate_frequency = np.asarray(estimate)
                    estimate_valid = np.asarray(True)
                    estimate_quality = np.asarray("ok")
                if estimate_frequency.size != 1 or estimate_valid.size != 1 or estimate_quality.size != 1 or np.iscomplexobj(estimate_frequency):
                    raise ValueError("frequency_estimator must return exactly one real frequency and scalar validity/quality")
                try:
                    measured = float(estimate_frequency.reshape(-1)[0])
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("frequency_estimator must return a numeric real frequency") from exc
                good = bool(estimate_valid.reshape(-1)[0]) and np.isfinite(measured)
                quality = str(estimate_quality.reshape(-1)[0])
                if good:
                    frequencies[output_index, row_index] = measured
                    accepted[output_index, row_index] = True
                if not good or quality != "ok":
                    reason = quality if quality != "ok" else "invalid_estimate"
                    messages[row_index].append(label + ":" + reason)
    valid = np.all(accepted, axis=0).reshape(leading_shape)
    qx_first, qy_first, qx_second, qy_second = (values.reshape(leading_shape) for values in frequencies)
    delta_qx = np.where(valid, qx_second - qx_first, np.nan)
    delta_qy = np.where(valid, qy_second - qy_first, np.nan)
    drift = np.hypot(delta_qx, delta_qy)
    with np.errstate(divide="ignore", invalid="ignore"):
        diffusion_log10 = np.log10(drift)
    quality = np.asarray([";".join(value) if value else "ok" for value in messages], dtype=str).reshape(leading_shape)
    return {
        "qx_first": qx_first,
        "qy_first": qy_first,
        "qx_second": qx_second,
        "qy_second": qy_second,
        "delta_qx": delta_qx,
        "delta_qy": delta_qy,
        "drift": drift,
        "diffusion_log10": diffusion_log10,
        "valid": valid,
        "quality": quality,
        "metadata": {
            "method": method if frequency_estimator is None else "custom",
            "n_samples": n_samples,
            "windows": [list(pair) for pair in intervals],
            "window_samples": interval_size,
            "sample_spacing": sample_spacing,
            "frequency_units": "cycles per sample-spacing unit",
            "difference_convention": "second minus first; no wrapping",
            "diffusion_definition": "log10(hypot(delta_qx, delta_qy)); indicator, not a physical diffusion coefficient",
            "window": window if method == "refined_fft" and frequency_estimator is None else None,
            "padding_factor": int(padding_factor) if method == "refined_fft" and frequency_estimator is None else None,
            "interpolation": interpolation if method == "refined_fft" and frequency_estimator is None else None,
            "frequency_range_x": None if frequency_range_x is None else list(frequency_range_x),
            "frequency_range_y": None if frequency_range_y is None else list(frequency_range_y),
            "remove_mean": bool(remove_mean),
        },
    }
