"""Unit V (part 1): color models and pseudo-color."""
import numpy as np
import matplotlib
from skimage import color as skcolor
from .utils import to_uint8


def rgb_to_ycbcr(rgb):
    """JFIF / JPEG full-range conversion. Returns float array (H, W, 3)."""
    r, g, b = [rgb[..., i].astype(float) for i in range(3)]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    return np.stack([y, cb, cr], axis=-1)

def ycbcr_to_rgb(ycc):
    y, cb, cr = ycc[..., 0], ycc[..., 1] - 128, ycc[..., 2] - 128
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return to_uint8(np.stack([r, g, b], axis=-1))

def rgb_to_cmy(rgb):
    return 255 - rgb

def rgb_to_hsv(rgb):
    """Library call; H, S, V each in [0, 1]."""
    return skcolor.rgb2hsv(rgb)

def pseudo_color(gray, cmap="jet"):
    """Intensity slicing / pseudo-coloring with a colormap lookup table."""
    lut = matplotlib.colormaps[cmap](np.linspace(0, 1, 256))[:, :3]
    return to_uint8(255 * lut[gray])
