"""Unit III: frequency-domain processing."""
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from .utils import to_uint8


def spectrum(img):
    """Centered log-magnitude spectrum, scaled to 0-255 for display."""
    s = np.log1p(np.abs(fftshift(fft2(img.astype(float)))))
    return to_uint8(255 * (s - s.min()) / (s.max() - s.min() + 1e-9))

def _dist(shape):
    M, N = shape
    u = np.arange(M) - M // 2
    v = np.arange(N) - N // 2
    return np.sqrt(u[:, None] ** 2 + v[None, :] ** 2)

def lowpass_mask(shape, d0, kind="gaussian", order=2):
    D = _dist(shape)
    if kind == "ideal":
        return (D <= d0).astype(float)
    if kind == "butterworth":
        return 1.0 / (1.0 + (D / d0) ** (2 * order))
    return np.exp(-(D ** 2) / (2 * d0 ** 2))

def highpass_mask(shape, d0, kind="gaussian", order=2):
    return 1.0 - lowpass_mask(shape, d0, kind, order)

def apply_filter(img, H):
    F = fftshift(fft2(img.astype(float)))
    return np.real(ifft2(ifftshift(F * H)))

def lowpass(img, d0, kind="gaussian"):
    return to_uint8(apply_filter(img, lowpass_mask(img.shape, d0, kind)))

def highpass(img, d0, kind="gaussian"):
    """High-pass output has negative values; shift/scale for display."""
    out = apply_filter(img, highpass_mask(img.shape, d0, kind))
    out = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return to_uint8(255 * out)

def high_frequency_emphasis(img, d0=30, a=0.5, b=1.5):
    H = a + b * highpass_mask(img.shape, d0, "gaussian")
    return to_uint8(apply_filter(img, H))

def homomorphic(img, d0=30, gl=0.5, gh=2.0, c=1.0):
    """Compress illumination (low freq) and boost reflectance (high freq)."""
    f = np.log1p(img.astype(float))
    H = (gh - gl) * (1 - np.exp(-c * _dist(img.shape) ** 2 / d0 ** 2)) + gl
    g = np.expm1(apply_filter(f, H))
    lo, hi = np.percentile(g, (1, 99))          # percentile stretch: robust to outliers
    return to_uint8(255 * (g - lo) / (hi - lo + 1e-9))
