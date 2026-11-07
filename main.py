"""Run all experiments and save figures/tables to results/.

Usage:
    python main.py                      # all units, built-in test images
    python main.py --image my.jpg       # use your own image
    python main.py --units 4 5          # only selected units
"""
import argparse, csv, io
import numpy as np
from PIL import Image
from ipa import (utils, metrics as mt, fundamentals as fu, enhancement as en, frequency as fq,
                 restoration as rs, color as co, compression as cp, segmentation as sg,
                 morphology as mo, features as ft)
from ipa.utils import grid, line_plot, RESULTS, to_uint8
import matplotlib.pyplot as plt


def save_csv(name, rows):
    with open(RESULTS / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n[{name}]")
    for r in rows:
        print("  " + " | ".join(f"{k}={v}" for k, v in r.items()))


# ------------------------------------------------------------------ Unit I
def unit1(gray, color):
    ims = [gray] + [fu.downsample(gray, f) for f in (2, 4, 8)]
    ts = ["Original", "Sampled /2", "Sampled /4", "Sampled /8"]
    ims += [fu.requantize(gray, b) for b in (6, 4, 2, 1)]
    ts += ["6-bit (64 levels)", "4-bit (16)", "2-bit (4)", "1-bit (2)"]
    grid(ims, ts, "unit1_sampling_quantization")


# ----------------------------------------------------------------- Unit II
def unit2(gray, color):
    low = to_uint8(gray * 0.3 + 90)                        # synthetic low-contrast image
    outs = [low, en.contrast_stretch(low), en.histogram_equalization(low), en.clahe(low),
            en.gamma_transform(low, 0.6), en.histogram_specification(low, gray)]
    ts = ["Low contrast", "Contrast stretch", "Hist. equalization", "CLAHE", "Gamma 0.6", "Hist. specification"]
    grid(outs, ts, "unit2_enhancement", ncols=3)

    fig, ax = plt.subplots(2, 3, figsize=(10, 5))
    for a, im, t in zip(ax.ravel(), outs, ts):
        a.bar(range(256), en.histogram(im), width=1); a.set_title(t, fontsize=9)
    fig.tight_layout(); fig.savefig(RESULTS / "unit2_histograms.png", dpi=120); plt.close(fig)

    noisy = rs.add_salt_pepper(gray, 0.05)
    sp = [noisy, en.smooth_box(noisy, 3), en.smooth_gaussian(noisy), en.median_filter(noisy, 3),
          gray, en.sharpen_laplacian(gray), en.unsharp_mask(gray, 1), en.unsharp_mask(gray, 2.5)]
    st = ["Salt & pepper 5%", "Box 3x3", "Gaussian 5x5", "Median 3x3",
          "Original", "Laplacian sharpen", "Unsharp mask", "High-boost k=2.5"]
    grid(sp, st, "unit2_spatial_filtering")
    save_csv("unit2_metrics", [
        {"method": "Entropy (bits/pixel): low-contrast input", "value": round(mt.entropy(low), 3)},
        {"method": "Entropy: hist. equalization", "value": round(mt.entropy(outs[2]), 3)},
        {"method": "Entropy: CLAHE", "value": round(mt.entropy(outs[3]), 3)},
        {"method": "PSNR (dB): salt & pepper noisy", "value": round(mt.psnr(gray, noisy), 2)},
        {"method": "PSNR: box 3x3", "value": round(mt.psnr(gray, sp[1]), 2)},
        {"method": "PSNR: Gaussian 5x5", "value": round(mt.psnr(gray, sp[2]), 2)},
        {"method": "PSNR: median 3x3", "value": round(mt.psnr(gray, sp[3]), 2)}])


# ---------------------------------------------------------------- Unit III
def unit3(gray, color):
    ims = [gray, fq.spectrum(gray)]
    ts = ["Original", "Log spectrum"]
    for d0 in (10, 30, 60):
        ims.append(fq.lowpass(gray, d0, "gaussian")); ts.append(f"Gaussian LPF D0={d0}")
    ims += [fq.lowpass(gray, 30, "ideal"), fq.lowpass(gray, 30, "butterworth")]
    ts += ["Ideal LPF D0=30 (ringing)", "Butterworth LPF D0=30"]
    ims += [fq.highpass(gray, 30, "gaussian"), fq.highpass(gray, 30, "butterworth"),
            fq.high_frequency_emphasis(gray), fq.homomorphic(gray)]
    ts += ["Gaussian HPF D0=30", "Butterworth HPF D0=30", "High-freq emphasis", "Homomorphic"]
    grid(ims, ts, "unit3_frequency", ncols=4)
    d0s = [5, 10, 20, 40, 80]
    line_plot(d0s, {k: [mt.psnr(gray, fq.lowpass(gray, d, k)) for d in d0s] for k in ("ideal", "butterworth", "gaussian")},
              "Cutoff D0", "PSNR vs original (dB)", "Low-pass cutoff vs fidelity", "unit3_cutoff_psnr")


# ----------------------------------------------------------------- Unit IV
def unit4(gray, color):
    img = gray.astype(float)
    psf = rs.gaussian_psf(15, 3.0)
    rows, sigmas = [], [0, 2, 5, 10, 20]
    series = {"Degraded": [], "Inverse (thr=1e-3)": [], "Wiener (best K)": [], "CLS (best gamma)": []}
    shown = None
    for s in sigmas:
        g = rs.degrade(img, psf, s)
        inv = rs.inverse_filter(g, psf, 1e-3)
        best_w = max(((rs.wiener_filter(g, psf, K), K) for K in (1e-4, 1e-3, 1e-2, 5e-2, 1e-1)), key=lambda t: mt.psnr(img, t[0]))
        best_c = max(((rs.cls_filter(g, psf, G), G) for G in (1e-4, 1e-3, 1e-2, 5e-2, 1e-1)), key=lambda t: mt.psnr(img, t[0]))
        vals = [mt.psnr(img, g), mt.psnr(img, inv), mt.psnr(img, best_w[0]), mt.psnr(img, best_c[0])]
        for k, v in zip(series, vals):
            series[k].append(v)
        rows.append({"noise_sigma": s, "degraded": round(vals[0], 2), "inverse": round(vals[1], 2),
                     "wiener": round(vals[2], 2), "K": best_w[1], "cls": round(vals[3], 2), "gamma": best_c[1]})
        if s == 5:
            shown = [img, g, inv, best_w[0], best_c[0]]
    grid([to_uint8(x) for x in shown],
         ["Original", "Blur + noise (σ=5)", "Inverse filter", "Wiener", "CLS"], "unit4_restoration", ncols=5, figw=2.8)
    line_plot(sigmas, series, "Noise σ", "PSNR (dB)", "Restoration vs noise level", "unit4_psnr_vs_noise")
    save_csv("unit4_metrics", rows)

    mpsf = rs.motion_psf(21, 30)
    g = rs.degrade(img, mpsf, 1.0)
    grid([to_uint8(x) for x in (img, g, rs.inverse_filter(g, mpsf, 1e-2), rs.wiener_filter(g, mpsf, 0.01))],
         ["Original", "Motion blur (21px, 30°)", "Inverse", "Wiener K=0.01"], "unit4_motion_blur", figw=3)


# ------------------------------------------------------------------ Unit V
def unit5(gray, color):
    ycc = co.rgb_to_ycbcr(color)
    hsv = co.rgb_to_hsv(color)
    ims = [color, to_uint8(ycc[..., 0]), to_uint8(ycc[..., 1]), to_uint8(ycc[..., 2]),
           co.rgb_to_cmy(color), to_uint8(hsv[..., 0] * 255), to_uint8(hsv[..., 1] * 255), co.pseudo_color(gray)]
    ts = ["RGB", "Y", "Cb", "Cr", "CMY", "H", "S", "Pseudo-color (jet)"]
    grid(ims, ts, "unit5_color_models")

    qs = [5, 10, 20, 30, 50, 70, 90]
    rows, recs = [], {}
    for q in qs:
        c = cp.compress_color(color, q)
        rec = cp.decompress_color(c)
        buf = io.BytesIO(); Image.fromarray(color).save(buf, "JPEG", quality=q)
        recs[q] = rec
        rows.append({"quality": q, "bits_per_pixel": round(c["bits"] / (color.shape[0] * color.shape[1]), 3),
                     "compression_ratio": round(cp.compression_ratio(color.shape[:2], 3, c["bits"]), 2),
                     "psnr_db": round(mt.psnr(color, rec), 2), "ssim": round(mt.ssim(color, rec), 4),
                     "pillow_jpeg_ratio": round(color.size / buf.tell(), 2)})
    save_csv("unit5_codec_metrics", rows)
    grid([color] + [recs[q] for q in (10, 30, 50, 90)],
         ["Original"] + [f"Q={q}  CR={r['compression_ratio']}  {r['psnr_db']}dB" for q, r in
                         zip((10, 30, 50, 90), [x for x in rows if x["quality"] in (10, 30, 50, 90)])],
         "unit5_codec_quality", ncols=5, figw=2.8)
    line_plot([r["compression_ratio"] for r in rows], {"Own codec": [r["psnr_db"] for r in rows]},
              "Compression ratio", "PSNR (dB)", "Rate-distortion (color)", "unit5_rate_distortion")
    line_plot(qs, {"Own codec": [r["compression_ratio"] for r in rows],
                   "Pillow JPEG": [r["pillow_jpeg_ratio"] for r in rows]},
              "Quality factor", "Compression ratio", "Own codec vs Pillow JPEG", "unit5_vs_pillow")

    # entropy-stage ablation on the luminance channel at Q=50
    rows2 = []
    ql = cp.scale_qtable(cp.Q_LUM, 50)
    enc = cp.encode_channel(gray, ql)
    n_sym = sum(1 for _ in enc["codes"])
    rows2.append({"stage": "Raw 8-bit", "bits_per_pixel": 8.0})
    rows2.append({"stage": "Image entropy (lower bound, no transform)", "bits_per_pixel": round(mt.entropy(gray), 3)})
    rows2.append({"stage": "DCT+quant+RLE+Huffman (payload only)", "bits_per_pixel": round(len(enc["bits"]) / gray.size, 3)})
    rows2.append({"stage": f"... incl. Huffman table ({n_sym} entries)", "bits_per_pixel": round(cp.channel_bits(enc) / gray.size, 3)})
    save_csv("unit5_entropy_stages", rows2)

    # chroma subsampling effect
    rows3 = []
    for sub in (False, True):
        c = cp.compress_color(color, 50, subsample=sub); rec = cp.decompress_color(c)
        rows3.append({"chroma_subsampling_420": sub, "compression_ratio": round(cp.compression_ratio(color.shape[:2], 3, c["bits"]), 2),
                      "psnr_db": round(mt.psnr(color, rec), 2)})
    save_csv("unit5_subsampling", rows3)


# ----------------------------------------------------------------- Unit VI
def unit6(gray, color):
    comp = {q: cp.decompress_gray(cp.compress_gray(gray, q)) for q in (10, 50, 90)}
    refcanny = sg.canny(gray)
    ims = [gray, to_uint8(255 * sg.edge_map(sg.sobel(gray))), to_uint8(255 * sg.edge_map(sg.prewitt(gray))), to_uint8(255 * refcanny)]
    ts = ["Original", "Sobel", "Prewitt", "Canny"]
    rows = []
    for q, rec in comp.items():
        ce = sg.canny(rec)
        ims += [rec, to_uint8(255 * ce)]
        ts += [f"Decoded Q={q}", f"Canny on Q={q}"]
        rows.append({"quality": q, "canny_edge_F1_vs_original": round(sg.edge_f1(refcanny, ce), 3),
                     "sobel_edge_F1": round(sg.edge_f1(sg.edge_map(sg.sobel(gray)), sg.edge_map(sg.sobel(rec))), 3)})
    grid(ims, ts, "unit6_edges_vs_compression", ncols=4)
    save_csv("unit6_edge_preservation", rows)

    t = sg.otsu_threshold(gray)
    mask = gray >= t
    se = np.ones((3, 3))
    clean = mo.closing(mo.opening(mask, se), se)
    seed = (10, gray.shape[1] // 2)
    rg = sg.region_growing(gray, seed, thr=15)
    lab, cen = sg.kmeans_segment(gray, 3)
    labc, cenc = sg.kmeans_segment(color, 4)
    grid([to_uint8(255 * mask), to_uint8(255 * clean), to_uint8(255 * mo.dilate(mask)), to_uint8(255 * mo.erode(mask)),
          to_uint8(255 * rg), sg.label_to_image(lab, cen), sg.label_to_image(labc, cenc)],
         [f"Otsu T={t}", "Opening+closing", "Dilation", "Erosion", "Region growing", "K-means (k=3, gray)", "K-means (k=4, color)"],
         "unit6_segmentation", ncols=4)
    return clean


# ---------------------------------------------------------------- Unit VII
def unit7(gray, color, mask=None):
    if mask is None:
        mask = gray >= sg.otsu_threshold(gray)
    shape_rows = ft.shape_features(mask)
    save_csv("unit7_shape_features", shape_rows)
    h, w = gray.shape
    patches = {"top-left": gray[: h // 2, : w // 2], "bottom-right": gray[h // 2:, w // 2:]}
    save_csv("unit7_texture_features", [{"patch": k, **ft.texture_features(np.ascontiguousarray(v))} for k, v in patches.items()])


UNITS = {1: unit1, 2: unit2, 3: unit3, 4: unit4, 5: unit5, 6: unit6, 7: unit7}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--units", type=int, nargs="*", default=list(UNITS))
    a = ap.parse_args()
    gray, color = utils.load_images(a.image, a.size)
    clean = None
    for u in a.units:
        print(f"\n=== Unit {u} ===")
        r = UNITS[u](gray, color) if u != 7 else unit7(gray, color, clean)
        if u == 6:
            clean = r
    print(f"\nDone. Outputs in {RESULTS}")
