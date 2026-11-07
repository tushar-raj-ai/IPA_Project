"""Unit I: sampling and quantization."""
import numpy as np


def downsample(img, factor):
    """Spatial sampling: keep every `factor`-th pixel, then show at original size (nearest)."""
    small = img[::factor, ::factor]
    big = np.repeat(np.repeat(small, factor, 0), factor, 1)
    return big[: img.shape[0], : img.shape[1]]


def requantize(img, bits):
    """Reduce gray levels to 2**bits."""
    levels = 2 ** bits
    step = 256 / levels
    q = np.floor(img / step)
    return np.round(q * (255 / (levels - 1))).astype(np.uint8)
