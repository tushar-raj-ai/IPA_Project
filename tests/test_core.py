"""Sanity tests: run with `python -m pytest -q`."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from scipy.fft import dctn
from skimage import data
from ipa import compression as cp, enhancement as en, metrics as mt, restoration as rs


def test_dct_matches_scipy():
    b = np.random.default_rng(0).random((8, 8)) * 255
    assert np.allclose(cp.dct2(b), dctn(b, norm="ortho"))
    assert np.allclose(cp.idct2(cp.dct2(b)), b)

def test_zigzag_roundtrip_and_order():
    b = np.arange(64).reshape(8, 8)
    assert np.array_equal(cp.inverse_zigzag(cp.zigzag(b)), b)
    assert list(cp.zigzag(b)[:6]) == [0, 1, 8, 16, 9, 2]

def test_huffman_prefix_free():
    codes = cp.build_huffman(list("aaaabbbccd"))
    vals = list(codes.values())
    assert not any(a != b and b.startswith(a) for a in vals for b in vals)

def test_codec_gray_high_quality():
    img = data.camera()[:128, :128]
    out = cp.decompress_gray(cp.compress_gray(img, 95))
    assert out.shape == img.shape and mt.psnr(img, out) > 35

def test_codec_odd_size_color():
    img = data.astronaut()[:101, :75]
    out = cp.decompress_color(cp.compress_color(img, 80))
    assert out.shape == img.shape and mt.psnr(img, out) > 28

def test_quality_monotonic():
    img = data.camera()[:128, :128]
    bits = [cp.compress_gray(img, q)["bits"] for q in (10, 50, 90)]
    assert bits[0] < bits[1] < bits[2]

def test_hist_eq_matches_skimage_range():
    img = data.camera()
    assert en.histogram_equalization(img).max() == 255

def test_wiener_beats_inverse_with_noise():
    img = data.camera()[:128, :128].astype(float)
    psf = rs.gaussian_psf(9, 2)
    g = rs.degrade(img, psf, 5)
    assert mt.psnr(img, rs.wiener_filter(g, psf, 0.01)) > mt.psnr(img, rs.inverse_filter(g, psf, 1e-6))
