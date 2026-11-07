"""Quality and compression metrics."""
import numpy as np
from skimage.metrics import structural_similarity


def mse(a, b):
    return float(np.mean((a.astype(float) - b.astype(float)) ** 2))


def psnr(a, b):
    m = mse(np.clip(a, 0, 255), np.clip(b, 0, 255))
    return float("inf") if m == 0 else float(10 * np.log10(255.0 ** 2 / m))


def ssim(a, b):
    a = np.clip(a, 0, 255); b = np.clip(b, 0, 255)
    return float(structural_similarity(a, b, data_range=255, channel_axis=-1 if a.ndim == 3 else None))


def entropy(img):
    """Shannon entropy in bits/pixel of an 8-bit image."""
    p = np.bincount(img.astype(np.uint8).ravel(), minlength=256) / img.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())
