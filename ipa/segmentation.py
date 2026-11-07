"""Unit VI (part 2): edge detection and segmentation."""
from collections import deque
import numpy as np
from scipy.ndimage import binary_dilation
from skimage import feature
from .enhancement import convolve2d, gaussian_kernel
from .utils import to_uint8

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], float)


def gradient_magnitude(img, kx):
    gx = convolve2d(img, kx)
    gy = convolve2d(img, kx.T)
    return np.hypot(gx, gy)

def sobel(img):
    return gradient_magnitude(img, SOBEL_X)

def prewitt(img):
    return gradient_magnitude(img, PREWITT_X)

def canny(img, sigma=1.5, low=0.1, high=0.3):
    """Library call (skimage). Own Gaussian smoothing + gradient is shown in sobel()."""
    return feature.canny(img, sigma=sigma, low_threshold=low * 255, high_threshold=high * 255)

def edge_map(mag, frac=0.25):
    return mag > frac * mag.max()

def edge_f1(ref, test, tol=1):
    """F1 between two binary edge maps allowing `tol` pixels of misalignment."""
    se = np.ones((2 * tol + 1, 2 * tol + 1), bool)
    prec = (test & binary_dilation(ref, se)).sum() / max(test.sum(), 1)
    rec = (ref & binary_dilation(test, se)).sum() / max(ref.sum(), 1)
    return float(2 * prec * rec / max(prec + rec, 1e-9))

def otsu_threshold(img):
    hist = np.bincount(img.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    sigma_b = (mu[-1] * omega - mu) ** 2 / np.maximum(omega * (1 - omega), 1e-12)
    return int(np.argmax(sigma_b))

def region_growing(img, seed, thr=15):
    """4-connected region growing: add neighbours within `thr` of the seed intensity."""
    H, W = img.shape
    mask = np.zeros((H, W), bool)
    ref = float(img[seed])
    q = deque([seed]); mask[seed] = True
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and not mask[rr, cc] and abs(float(img[rr, cc]) - ref) <= thr:
                mask[rr, cc] = True; q.append((rr, cc))
    return mask

def kmeans_segment(img, k=3, iters=25, seed=0):
    """K-means clustering on pixel intensities / colors. Returns label image and centers."""
    X = img.reshape(-1, 1 if img.ndim == 2 else img.shape[2]).astype(float)
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        labels = np.argmin(((X[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        new = np.array([X[labels == j].mean(0) if np.any(labels == j) else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return labels.reshape(img.shape[:2]), centers

def label_to_image(labels, centers):
    out = centers[labels]
    return to_uint8(out if out.shape[-1] == 3 else out[..., 0])
