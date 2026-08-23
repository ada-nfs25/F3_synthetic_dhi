# Implementation developed with AI (Claude Code) assistance - see AI_USAGE.md.
"""
Seismic attribute computation for dataset patches.

Same Hilbert-transform approach as notebook 02's facies-pipeline attributes
(envelope, instantaneous phase), generalised to operate along the last axis
of an array of any shape - works for a single trace, a 2D inline slice, or a
3D (inline, crossline, time) patch without changes.

F7: compute_attribute_stack originally exported only 3 channels (amplitude,
envelope, instantaneous phase), so none of the ~20 attributes implemented for
the facies pipeline (irp-nfs25/notebooks/dhi_attribute_computation.ipynb)
reached the exported .npz at all - blocking the XGBoost model work entirely,
since there was nothing beyond those 3 channels to build features from.

Extended below with a small, deliberately curated set - not all ~20 wholesale
(see the runbook, C3) - weighted towards bounded/relative channels, since
those transfer across surveys whereas absolute-scale ones (raw envelope, raw
RMS) carry whatever gain the processing applied and don't. This isn't
theoretical: it's exactly what the regional-bias investigation in this
project found directly - raw_amplitude/apparent_polarity/relative_acoustic_
impedance flipped sign between the train and test inline ranges, and the
XGBoost model trained on them collapsed to ~0 recall out-of-region as a
result. `band_ratio` below is the one channel here that's bounded [-1, 1] by
construction; the rest still carry the survey's absolute gain, but C4's
window/whole-patch ratio feature construction is what's meant to neutralise
that downstream, not the channel definitions themselves.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt


def compute_envelope(data):
    """Reflection strength: |analytic signal|, along the last (time) axis."""
    analytic = hilbert(data, axis=-1)
    return np.abs(analytic)


def compute_instantaneous_phase(data):
    """Unwrapped instantaneous phase, along the last (time) axis."""
    analytic = hilbert(data, axis=-1)
    return np.unwrap(np.angle(analytic), axis=-1)


def compute_instantaneous_frequency(data, dt_s):
    """
    Instantaneous frequency: (1 / 2*pi) * d(phase)/dt, along the last (time)
    axis. dt_s: sample interval in seconds.
    """
    phase = compute_instantaneous_phase(data)
    freq = np.diff(phase, axis=-1) / (2 * np.pi * dt_s)
    # diff shrinks the last axis by 1 - repeat the last value to preserve shape
    return np.concatenate([freq, freq[..., -1:]], axis=-1)


def compute_rms_amplitude(data, window_samples=9):
    """Windowed RMS amplitude along the last (time) axis."""
    mean_sq = uniform_filter1d(data ** 2, size=window_samples, axis=-1, mode='nearest')
    return np.sqrt(mean_sq)


def compute_sweetness(data, dt_s):
    """
    Sweetness: RMS amplitude / sqrt(|instantaneous frequency|) - hydrocarbon
    indicator (Petrel-style). High amplitude + low frequency = "sweet spot".
    """
    rms = compute_rms_amplitude(data)
    inst_freq = compute_instantaneous_frequency(data, dt_s)
    return rms / np.maximum(np.sqrt(np.abs(inst_freq)), 1e-6)


def estimate_dominant_frequency(data, dt_s, min_freq_hz=5.0, window_ms=150.0):
    """
    FFT-based dominant frequency: mean amplitude spectrum across every trace
    in `data` (any leading shape, last axis = time), Hann-tapered and
    demeaned, ignoring near-DC bins (short-window spectral leakage, not a
    real dominant frequency - same convention as the survey-level
    calibration in scripts/regenerate_p0_dataset.py and
    notebooks/synthetic_dhi_generation.ipynb).

    Estimated over a `window_ms`-wide central time window, not the full
    input - same reason calibration() above only ever used a ~150ms
    reference window rather than a whole patch: a long real-data window
    mixes in low-frequency structural/geological trend content that
    dominates the spectrum over the wavelet's own frequency (measured
    directly - a 500ms dim_spot patch gave an ~8.8Hz "dominant" frequency
    this way, nowhere near this survey's actual ~60Hz, and drove
    compute_band_ratio's derived low_freq/bandwidth negative). A short
    central window is much closer to what calibration()'s own narrow
    reference window measures.
    """
    n_time = data.shape[-1]
    window_samples = min(n_time, max(1, int(round(window_ms / 1000.0 / dt_s))))
    lo = (n_time - window_samples) // 2
    windowed = data[..., lo:lo + window_samples]

    flat = windowed.reshape(-1, windowed.shape[-1])
    tapered = (flat - flat.mean(axis=-1, keepdims=True)) * np.hanning(flat.shape[-1])
    n_fft = max(512, flat.shape[-1])
    freqs = np.fft.rfftfreq(n_fft, d=dt_s)
    spectrum = np.abs(np.fft.rfft(tapered, n=n_fft, axis=-1)).mean(axis=0)
    return float(freqs[np.argmax(spectrum * (freqs >= min_freq_hz))])


def compute_band_ratio(data, dt_s, dominant_freq_hz=None, low_frac=0.25, high_frac=0.75, bandwidth=6.0):
    """
    Bounded [-1, 1] spectral contrast: (low-band envelope - high-band
    envelope) / (low-band envelope + high-band envelope + eps), along the
    last (time) axis. Positive where low-frequency energy dominates (a tuning/
    thin-bed or attenuation cue), negative where high-frequency energy
    dominates. Unlike raw envelope or RMS, this is a genuine ratio and so
    doesn't carry the survey's arbitrary amplitude gain - see the module
    docstring.

    dominant_freq_hz: survey-adaptive band anchor (Round 2, P1 item 2,
    Aziz - "fixed 15/45Hz bands sit below the dominant band in shallower
    data"): low_freq = low_frac*dominant_freq_hz, high_freq =
    high_frac*dominant_freq_hz, instead of fixed absolute Hz values that only
    mean what they're supposed to at whatever frequency they were originally
    tuned against. If None (default), estimated directly from `data` itself
    (see estimate_dominant_frequency) - "the measured dominant frequency of
    whatever volume is being processed", not a pre-supplied constant, so this
    adapts per depth band/survey with no caller-side plumbing required.
    low_frac=0.25/high_frac=0.75 are chosen so this is numerically IDENTICAL
    to the previous fixed 15/45Hz bands whenever dominant_freq_hz is ~60Hz
    (this survey's own measured value) - a generalisation of the existing
    behaviour, not a change to it, for every already-generated H1 example.
    """
    if dominant_freq_hz is None:
        dominant_freq_hz = estimate_dominant_frequency(data, dt_s)
    nyquist = 0.5 / dt_s
    low_freq = np.clip(low_frac * dominant_freq_hz, bandwidth / 2 + 0.5, nyquist - bandwidth / 2 - 0.5)
    high_freq = np.clip(high_frac * dominant_freq_hz, low_freq + bandwidth, nyquist - bandwidth / 2 - 0.5)

    def band_envelope(center_freq):
        f_lo, f_hi = center_freq - bandwidth / 2, center_freq + bandwidth / 2
        sos = butter(4, [f_lo, f_hi], btype='bandpass', fs=1.0 / dt_s, output='sos')
        filtered = sosfiltfilt(sos, data, axis=-1)
        return np.abs(hilbert(filtered, axis=-1))

    low_env = band_envelope(low_freq)
    high_env = band_envelope(high_freq)
    return (low_env - high_env) / (low_env + high_env + 1e-10)


def compute_local_var(data, window_samples=9):
    """
    Local variance along the last (time) axis - a simple texture/
    heterogeneity measure: high where the waveform is locally erratic, low
    where it's smooth.
    """
    mean = uniform_filter1d(data, size=window_samples, axis=-1, mode='nearest')
    mean_sq = uniform_filter1d(data ** 2, size=window_samples, axis=-1, mode='nearest')
    return mean_sq - mean ** 2


def compute_attribute_stack(amplitude, dt_s):
    """
    Stack a curated set of seismic attributes into a single multi-channel
    array (F7 - see module docstring for why this is a deliberately small,
    mostly-bounded selection rather than all ~20 facies-pipeline attributes).

    amplitude: (..., n_samples) array, NaN-free (Hilbert transform doesn't
        handle NaN gaps - fill missing traces before calling this).
    dt_s: sample interval in seconds (needed for the frequency-based channels).

    Returns (channels, ...) array plus the channel-name list, in matching
    order, so features stay traceable downstream (e.g. for feature
    importance plots in the XGBoost step, C4).
    """
    channels = {
        'amplitude': amplitude,
        'envelope': compute_envelope(amplitude),
        'inst_phase': compute_instantaneous_phase(amplitude),
        'inst_freq': compute_instantaneous_frequency(amplitude, dt_s),
        'sweetness': compute_sweetness(amplitude, dt_s),
        'band_ratio': compute_band_ratio(amplitude, dt_s),
        'rms': compute_rms_amplitude(amplitude),
        'local_var': compute_local_var(amplitude),
    }
    return np.stack(list(channels.values()), axis=0), list(channels)
