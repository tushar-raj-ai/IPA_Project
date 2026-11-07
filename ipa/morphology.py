"""Unit VI (part 1): binary morphology with a flat structuring element (own implementation)."""
import numpy as np


def _shifts(mask, se, pad_value=False):
    kh, kw = se.shape
    ph, pw = kh // 2, kw // 2
    p = np.pad(mask, ((ph, ph), (pw, pw)), mode="constant", constant_values=pad_value)
    H, W = mask.shape
    return [p[i:i + H, j:j + W] for i in range(kh) for j in range(kw) if se[i, j]]

def dilate(mask, se=np.ones((3, 3))):
    return np.any(_shifts(mask.astype(bool), se), axis=0)

def erode(mask, se=np.ones((3, 3))):
    return np.all(_shifts(mask.astype(bool), se, pad_value=True), axis=0)  # border is not eroded

def opening(mask, se=np.ones((3, 3))):
    return dilate(erode(mask, se), se)

def closing(mask, se=np.ones((3, 3))):
    return erode(dilate(mask, se), se)
