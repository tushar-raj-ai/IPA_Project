"""Loading, conversion and plotting helpers."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import data, io, transform

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def to_uint8(x):
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def to_gray(rgb):
    """Luminance conversion (ITU-R BT.601)."""
    if rgb.ndim == 2:
        return rgb
    return to_uint8(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])


def _resize(img, size):
    out = transform.resize(img, (size, size) + img.shape[2:], preserve_range=True, anti_aliasing=True)
    return to_uint8(out)


def load_images(path=None, size=256):
    """Return (gray, color) uint8 test images. Uses built-in skimage images if no path is given."""
    if path:
        img = io.imread(path)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        color = _resize(img, size) if img.ndim == 3 else _resize(np.stack([img] * 3, -1), size)
        return to_gray(color), color

    # Default inputs are stored in the project's images/ folder.
    # camera.png is used as the grayscale source; astronaut.png is used as RGB.
    image_dir = Path(__file__).resolve().parent.parent / "images"
    camera_path = image_dir / "camera.png"
    astronaut_path = image_dir / "astronaut.png"
    if camera_path.exists() and astronaut_path.exists():
        gray_img = io.imread(camera_path)
        color_img = io.imread(astronaut_path)
        if color_img.ndim == 3 and color_img.shape[2] == 4:
            color_img = color_img[..., :3]
        return _resize(to_gray(gray_img), size), _resize(color_img, size)

    # Fallback for older installations without the bundled images.
    return _resize(data.camera(), size), _resize(data.astronaut(), size)


def grid(images, titles, name, ncols=4, cmap="gray", figw=3.2):
    """Save a grid of images to results/<name>.png."""
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figw * ncols, figw * nrows * 1.2))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, im, t in zip(axes, images, titles):
        if im.ndim == 2:
            ax.imshow(im, cmap=cmap, vmin=0, vmax=255)
        else:
            ax.imshow(im)
        ax.set_title(t, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}.png", dpi=130)
    plt.close(fig)


def line_plot(xs, series, xlabel, ylabel, title, name, logx=False):
    """series: dict label -> list of y values."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for label, ys in series.items():
        ax.plot(xs, ys, marker="o", label=label)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}.png", dpi=130)
    plt.close(fig)
