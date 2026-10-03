"""Portable windowed Fourier spectra and local peak estimates."""


def compute_refined_fft(signal,
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
