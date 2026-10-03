"""Portable amplitude-normalized Fourier spectra."""


def compute_fft(signal, sample_spacing=1.0, *, axis=-1, remove_mean=False):
    """Compute independent spectra along ``axis`` using only NumPy.

    This function can be copied alone into another script. The returned
    frequency axis is always last; other axes retain their original order.
    Frequencies are cycles per unit of ``sample_spacing`` and phases are in
    radians at sample zero. Real input produces a one-sided spectrum: positive
    frequency coefficients are doubled except DC and an even-length Nyquist
    bin, so a resolved cosine has its peak amplitude. Complex input produces
    a signed, ascending, two-sided spectrum without doubling. ``coefficients``
    use these same amplitude normalizations, not NumPy's raw FFT convention.
    Missing/nonfinite samples are rejected rather than filled or interpolated.
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
    if not isinstance(remove_mean, (bool, np.bool_)):
        raise ValueError("remove_mean must be a boolean")
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
    if n_samples < 2:
        raise ValueError("FFT requires at least 2 samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal contains nonfinite samples; select a complete finite interval")
    if remove_mean:
        values = values - np.mean(values, axis=-1, keepdims=True)
    is_real = not np.iscomplexobj(values)
    if is_real:
        frequency = np.fft.rfftfreq(n_samples, d=sample_spacing)
        coefficients = np.fft.rfft(values, axis=-1) / n_samples
        coefficients[..., 1:-1 if n_samples % 2 == 0 else None] *= 2
    else:
        frequency = np.fft.fftshift(np.fft.fftfreq(n_samples, d=sample_spacing))
        coefficients = np.fft.fftshift(np.fft.fft(values, axis=-1), axes=-1) / n_samples
    return {
        "frequency": frequency,
        "coefficients": coefficients,
        "amplitude": np.abs(coefficients),
        "phase": np.angle(coefficients),
        "metadata": {
            "method": "fft",
            "n_samples": n_samples,
            "n_fft": n_samples,
            "sample_spacing": sample_spacing,
            "frequency_units": "cycles per sample-spacing unit",
            "phase_reference": "first input sample",
            "spectrum": "one_sided_real" if is_real else "two_sided_complex",
            "coefficient_normalization": "peak amplitude; real interior positive frequencies doubled",
            "frequency_axis": -1,
            "remove_mean": bool(remove_mean),
        },
    }
