"""Unit VII: shape and texture features (library-assisted)."""
import numpy as np
from skimage import measure, feature


def shape_features(mask, top=5):
    """Region descriptors for the `top` largest connected components."""
    props = sorted(measure.regionprops(measure.label(mask)), key=lambda p: -p.area)[:top]
    return [{"label": p.label, "area": int(p.area), "perimeter": round(float(p.perimeter), 1),
             "centroid": tuple(round(float(x), 1) for x in p.centroid),
             "eccentricity": round(float(p.eccentricity), 3),
             "solidity": round(float(p.solidity), 3)} for p in props]

def texture_features(img):
    """GLCM statistics + LBP histogram entropy for a gray patch."""
    q = (img // 4).astype(np.uint8)              # 64 gray levels
    glcm = feature.graycomatrix(q, [1], [0, np.pi / 2], levels=64, symmetric=True, normed=True)
    out = {p: round(float(feature.graycoprops(glcm, p).mean()), 4)
           for p in ("contrast", "homogeneity", "energy", "correlation")}
    lbp = feature.local_binary_pattern(img, 8, 1, method="uniform")
    h = np.bincount(lbp.astype(int).ravel(), minlength=10) / lbp.size
    h = h[h > 0]
    out["lbp_entropy"] = round(float(-(h * np.log2(h)).sum()), 4)
    return out
