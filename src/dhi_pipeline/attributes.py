"""
Seismic attribute computation for dataset patches.

Same Hilbert-transform approach as notebook 02's facies-pipeline attributes
(envelope, instantaneous phase), generalised to operate along the last axis
of an array of any shape - works for a single trace, a 2D inline slice, or a
3D (inline, crossline, time) patch without changes.
"""

import numpy as np
from scipy.signal import hilbert


def compute_envelope(data):
    """Reflection strength: |analytic signal|, along the last (time) axis."""
    analytic = hilbert(data, axis=-1)
    return np.abs(analytic)


def compute_instantaneous_phase(data):
    """Unwrapped instantaneous phase, along the last (time) axis."""
    analytic = hilbert(data, axis=-1)
    return np.unwrap(np.angle(analytic), axis=-1)


def compute_attribute_stack(amplitude):
    """
    Stack raw amplitude, envelope, and instantaneous phase into a single
    multi-channel array.

    amplitude: (..., n_samples) array, NaN-free (Hilbert transform doesn't
        handle NaN gaps - fill missing traces before calling this).

    Returns array of shape (3, ...) - channel 0 = amplitude, 1 = envelope,
    2 = instantaneous phase.
    """
    return np.stack([amplitude, compute_envelope(amplitude), compute_instantaneous_phase(amplitude)], axis=0)
