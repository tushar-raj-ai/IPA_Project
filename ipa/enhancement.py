"""Unit II: spatial-domain enhancement."""
import numpy as np
from skimage import exposure
from .utils import to_uint8


# ---------- gray-level transformations ----------
def negative(img):
    return 255 - img

def log_transform(img):
    c = 255 / np.log1p(img.max())
    return to_uint8(c * np.log1p(img.astype(float)))

def gamma_transform(img, gamma):
    return to_uint8(255 * (img / 255.0) ** gamma)

def contrast_stretch(img):
    lo, hi = float(img.min()), float(img.max())
    return to_uint8((img - lo) / max(hi - lo, 1) * 255)

def threshold(img, t=128):
    return ((img >= t) * 255).astype(np.uint8)


# ---------- histogram processing ----------
def histogram(img):
    return np.bincount(img.ravel(), minlength=256)

def histogram_equalization(img):
    cdf = histogram(img).cumsum()
    cdf_min = cdf[cdf > 0][0]
    denom = max(img.size - cdf_min, 1)
    lut = to_uint8((cdf - cdf_min) / denom * 255)
    return lut[img]

def histogram_specification(img, ref):
    """Map the histogram of `img` onto the histogram of `ref`."""
    cdf_s = histogram(img).cumsum() / img.size
    cdf_r = histogram(ref).cumsum() / ref.size
    lut = np.searchsorted(cdf_r, cdf_s).clip(0, 255).astype(np.uint8)
    return lut[img]

def clahe(img, clip_limit=0.02, tiles=8):
    """Adaptive HE (library call: skimage)."""
    k = max(img.shape) // tiles
    return to_uint8(255 * exposure.equalize_adapthist(img, kernel_size=k, clip_limit=clip_limit))


# ---------- spatial filtering ----------
def convolve2d(img, kernel):
    """Own 2-D convolution with reflect padding (odd-sized kernels)."""
    k = np.flipud(np.fliplr(kernel))
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    p = np.pad(img.astype(np.float64), ((ph, ph), (pw, pw)), mode="reflect")
    H, W = img.shape
    out = np.zeros((H, W))
    for i in range(kh):
        for j in range(kw):
            out += k[i, j] * p[i:i + H, j:j + W]
    return out

def box_kernel(n=3):
    return np.ones((n, n)) / (n * n)

def gaussian_kernel(n=5, sigma=1.0):
    ax = np.arange(n) - n // 2
    g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2))
    return g / g.sum()

LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)

def smooth_box(img, n=5):
    return to_uint8(convolve2d(img, box_kernel(n)))

def smooth_gaussian(img, n=5, sigma=1.5):
    return to_uint8(convolve2d(img, gaussian_kernel(n, sigma)))

def median_filter(img, n=3):
    p = np.pad(img, n // 2, mode="reflect")
    H, W = img.shape
    stack = [p[i:i + H, j:j + W] for i in range(n) for j in range(n)]
    return np.median(np.stack(stack), axis=0).astype(np.uint8)

def sharpen_laplacian(img):
    return to_uint8(img - convolve2d(img, LAPLACIAN))

def unsharp_mask(img, k=1.0, n=5, sigma=1.5):
    """k=1: unsharp masking; k>1: high-boost filtering."""
    blur = convolve2d(img, gaussian_kernel(n, sigma))
    return to_uint8(img + k * (img - blur))
