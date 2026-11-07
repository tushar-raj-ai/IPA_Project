# Image Enhancement, Restoration and Compression: A Complete Digital Image Processing Pipeline

**Course:** Image Processing and Applications (CS-30015), KIIT

A Python project that implements the core algorithms of Units I–VII from scratch (NumPy), with
libraries used only for cross-checking, display, or well-known primitives (marked below).

## Setup & run

```bash
pip install -r requirements.txt
python main.py                  # runs everything (~10 s), outputs go to results/
python main.py --units 4 5      # only some units
python main.py --image my.jpg   # use your own photo
python -m pytest -q tests       # 8 sanity tests (DCT vs SciPy, lossless entropy stage, ...)
```

## File structure

```
ipa_project/
├── main.py                 # experiment runner (one function per unit)
├── requirements.txt
├── README.md
├── ipa/
│   ├── utils.py            # image loading, grid/line plot helpers
│   ├── metrics.py          # MSE, PSNR, SSIM, entropy
│   ├── fundamentals.py     # Unit I   sampling, quantization
│   ├── enhancement.py      # Unit II  gray-level transforms, HE, specification, CLAHE, spatial filters
│   ├── frequency.py        # Unit III FFT, ideal/Butterworth/Gaussian LPF+HPF, homomorphic
│   ├── restoration.py      # Unit IV  PSFs, noise, inverse, Wiener, CLS
│   ├── color.py            # Unit V   RGB<->YCbCr, CMY, HSV, pseudo-color
│   ├── compression.py      # Unit V   DCT, quantization, zigzag, RLE, Huffman, full codec
│   ├── morphology.py       # Unit VI  dilation, erosion, opening, closing
│   ├── segmentation.py     # Unit VI  Sobel, Prewitt, Canny, Otsu, region growing, K-means
│   └── features.py         # Unit VII shape (regionprops) and texture (GLCM, LBP) features
├── tests/test_core.py
├── images/                 # bundled defaults: camera.png + astronaut.png; add your own test images here
├── results/                # generated figures (.png) and tables (.csv)
└── report/                 # put your final report here
```

## Syllabus mapping

| Unit | Module                   | Written from scratch                                                                                  | Library used                     |
| ---- | ------------------------ | ----------------------------------------------------------------------------------------------------- | -------------------------------- |
| I    | fundamentals             | sampling, requantization                                                                              | none                             |
| II   | enhancement              | HE, specification, gamma/log/stretch, convolution, box/Gaussian/median, Laplacian, unsharp/high-boost | `skimage` CLAHE                  |
| III  | frequency                | spectrum, all LPF/HPF masks, high-freq emphasis, homomorphic                                          | `numpy.fft`                      |
| IV   | restoration              | Gaussian/motion PSF, degradation, inverse, Wiener, CLS                                                | `numpy.fft`                      |
| V    | color, compression       | YCbCr, CMY, DCT/IDCT, quant tables, zigzag, DC diff, RLE, Huffman, 4:2:0, full codec                  | `skimage` HSV                    |
| VI   | morphology, segmentation | Sobel, Prewitt, Otsu, region growing, K-means, morphology                                             | `skimage` Canny                  |
| VII  | features                 | none                                                                                                  | `skimage` regionprops, GLCM, LBP |

## Experiments and what to say about each

| Output (results/)                                                             | Observation to discuss                                                                            |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `unit1_sampling_quantization.png`                                             | Checkerboarding when sampling drops; false contouring at low bit depth                            |
| `unit2_enhancement.png`, `unit2_histograms.png`                               | HE spreads the histogram globally; CLAHE keeps local contrast but limits noise amplification      |
| `unit2_spatial_filtering.png`, `unit2_metrics.csv`                            | Median beats box/Gaussian on salt-and-pepper noise (PSNR); high-boost k trades sharpness vs noise |
| `unit3_frequency.png`, `unit3_cutoff_psnr.png`                                | Ideal LPF causes ringing; Butterworth/Gaussian do not. Lower D0 = blurrier                        |
| `unit4_metrics.csv`, `unit4_psnr_vs_noise.png`                                | Inverse filter is best with zero noise and collapses with any noise; Wiener/CLS stay stable       |
| `unit4_motion_blur.png`                                                       | Restoration with a known motion PSF                                                               |
| `unit5_color_models.png`                                                      | Y holds structure, Cb/Cr are smooth, which is why chroma can be compressed harder                 |
| `unit5_codec_metrics.csv`, `unit5_rate_distortion.png`, `unit5_vs_pillow.png` | Rate–distortion tradeoff; your codec vs the real JPEG (Pillow)                                    |
| `unit5_entropy_stages.csv`                                                    | How many bits each stage saves; includes Huffman table overhead                                   |
| `unit5_subsampling.csv`                                                       | 4:2:0 gives a higher ratio for a small PSNR loss                                                  |
| `unit6_edges_vs_compression.png`, `unit6_edge_preservation.csv`               | How strongly compression affects Sobel/Canny edges (F1 with 1-pixel tolerance)                    |
| `unit6_segmentation.png`                                                      | Otsu, morphological cleanup, region growing, K-means                                              |
| `unit7_*.csv`                                                                 | Area, perimeter, eccentricity, solidity; GLCM contrast/homogeneity/energy; LBP entropy            |

## Things to be aware of

- **Huffman table overhead** is estimated at 24 bits per entry (16-bit symbol + 8-bit length); real JPEG headers differ slightly.
- **Wiener K and CLS gamma** in `unit4` are picked by best PSNR against the original (an oracle). In practice K is estimated from the noise variance.
- The codec is **JPEG-style, not a standards-compliant JPEG file**: it uses the same DCT, quantization tables, zigzag and RLE, but a single custom Huffman table per channel and a simple bitstring, so it can't be opened in an image viewer.
- Edge F1 uses a 1-pixel tolerance; try `tol=0` in `edge_f1` for a stricter test.
- Restoration assumes circular convolution (FFT) and a known PSF.

## Suggested report outline (4–6 pages)

1. Introduction and objectives (map to CO1–CO6)
2. Methodology: one short subsection per unit with the key equation
   (e.g. `s = T(r)` for HE, `W = H*/(|H|²+K)` for Wiener, `F(u,v) = C f Cᵀ` for DCT)
3. Experiments and results: figures and tables from `results/`, 2–3 sentences each
4. Codec analysis: rate–distortion curve, own codec vs Pillow, entropy stages
5. Conclusion and limitations (previous section)
6. References: Gonzalez & Woods, _Digital Image Processing_, 4th ed.; Wallace, "The JPEG still picture compression standard", 1992

## Ideas to extend if you have time

- Add 4:2:0 vs 4:4:4 and block-size (4×4, 16×16) experiments
- Salt-and-pepper restoration comparison for Unit IV (median vs Wiener)
- Arithmetic coding or a JPEG2000 (wavelet) comparison

## Bundled default input images

The `images/` folder includes `camera.png` (grayscale source for Units I–IV and VII)
and `astronaut.png` (RGB source for color/compression experiments). Running `python main.py`
with no `--image` argument loads these files by default, so the project no longer depends
on downloading the sample images at runtime. The generated figures and CSV files are written
to `results/` as before.
