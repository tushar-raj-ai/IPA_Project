"""Unit IV: degradation models and restoration filters (float images, 0-255 range)."""
import numpy as np
from numpy.fft import fft2, ifft2


def gaussian_psf(size=15, sigma=3.0):
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2))
    return g / g.sum()

def motion_psf(length=15, angle=0):
    size = length + 2 if length % 2 else length + 3
    psf = np.zeros((size, size)); c = size // 2
    t = np.linspace(-length / 2, length / 2, 4 * length)
    xs = np.round(c + t * np.cos(np.deg2rad(angle))).astype(int)
    ys = np.round(c - t * np.sin(np.deg2rad(angle))).astype(int)
    psf[ys, xs] = 1
    return psf / psf.sum()

def psf2otf(psf, shape):
    pad = np.zeros(shape)
    kh, kw = psf.shape
    pad[:kh, :kw] = psf
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return fft2(pad)

def degrade(img, psf, noise_sigma=0.0, seed=0):
    """g = h * f + n  (circular convolution + additive Gaussian noise)."""
    H = psf2otf(psf, img.shape)
    blurred = np.real(ifft2(fft2(img.astype(float)) * H))
    noise = np.random.default_rng(seed).normal(0, noise_sigma, img.shape)
    return blurred + noise

def add_salt_pepper(img, amount=0.05, seed=0):
    rng = np.random.default_rng(seed)
    out = img.copy()
    r = rng.random(img.shape)
    out[r < amount / 2] = 0
    out[r > 1 - amount / 2] = 255
    return out

def inverse_filter(g, psf, thr=1e-3):
    """Pseudo-inverse: frequencies where |H| < thr are zeroed (avoids noise blow-up)."""
    H = psf2otf(psf, g.shape)
    Hinv = np.where(np.abs(H) > thr, 1.0 / np.where(H == 0, 1, H), 0)
    return np.real(ifft2(fft2(g) * Hinv))

def wiener_filter(g, psf, K=0.01):
    """K ~ noise power / signal power."""
    H = psf2otf(psf, g.shape)
    W = np.conj(H) / (np.abs(H) ** 2 + K)
    return np.real(ifft2(fft2(g) * W))

def cls_filter(g, psf, gamma=0.01):
    """Constrained least squares with Laplacian smoothness constraint."""
    H = psf2otf(psf, g.shape)
    P = psf2otf(np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], float), g.shape)
    W = np.conj(H) / (np.abs(H) ** 2 + gamma * np.abs(P) ** 2)
    return np.real(ifft2(fft2(g) * W))
