"""Unit V (part 2): JPEG-style codec written from scratch.

Pipeline (per channel):
  level shift -> 8x8 blocks -> DCT -> quantize -> zigzag -> DC differencing
  -> run-length coding of AC -> Huffman coding -> bitstring
Decoder is the exact inverse. The entropy stage is lossless; all loss comes from quantization.
"""
import heapq
from collections import Counter
import numpy as np
from .color import rgb_to_ycbcr, ycbcr_to_rgb
from .utils import to_uint8

# ----------------------------------------------------------------- DCT
def dct_matrix(n=8):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    C = np.cos((2 * i + 1) * k * np.pi / (2 * n)) * np.sqrt(2.0 / n)
    C[0, :] = np.sqrt(1.0 / n)
    return C

C8 = dct_matrix(8)

def dct2(blocks):
    """2-D DCT-II of one 8x8 block or a stack (N, 8, 8)."""
    return C8 @ blocks @ C8.T

def idct2(blocks):
    return C8.T @ blocks @ C8

# ------------------------------------------------------------- blocking
def pad_to_multiple(ch, m=8):
    h, w = ch.shape
    return np.pad(ch, ((0, (-h) % m), (0, (-w) % m)), mode="edge")

def split_blocks(ch):
    h, w = ch.shape
    return ch.reshape(h // 8, 8, w // 8, 8).swapaxes(1, 2).reshape(-1, 8, 8)

def merge_blocks(blocks, shape):
    h, w = shape
    return blocks.reshape(h // 8, w // 8, 8, 8).swapaxes(1, 2).reshape(h, w)

# --------------------------------------------------------- quantization
Q_LUM = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype=float)

Q_CHR = np.full((8, 8), 99.0)
Q_CHR[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]])

def scale_qtable(table, quality):
    """IJG quality scaling: quality in [1, 100]."""
    quality = int(np.clip(quality, 1, 100))
    s = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.clip(np.floor((table * s + 50) / 100), 1, 255)

# --------------------------------------------------------------- zigzag
ZZ = np.array(sorted(((i, j) for i in range(8) for j in range(8)),
                     key=lambda p: (p[0] + p[1], p[0] if (p[0] + p[1]) % 2 else -p[0])))

def zigzag(block):
    return block[ZZ[:, 0], ZZ[:, 1]]

def inverse_zigzag(vec):
    out = np.zeros((8, 8), dtype=vec.dtype)
    out[ZZ[:, 0], ZZ[:, 1]] = vec
    return out

# ------------------------------------------------------- run-length (AC)
def rle_ac(ac):
    """AC coefficients -> list of ('A', zero_run, value), ('A', 15, 0)=ZRL (16 zeros), ('E',)=EOB."""
    out, run = [], 0
    for v in ac:
        if v == 0:
            run += 1
        else:
            while run > 15:
                out.append(("A", 15, 0)); run -= 16
            out.append(("A", run, int(v))); run = 0
    if run > 0:
        out.append(("E",))
    return out

# -------------------------------------------------------------- Huffman
def build_huffman(symbols):
    """Return dict symbol -> bitstring (prefix-free, optimal for the given frequencies)."""
    freq = Counter(symbols)
    if len(freq) == 1:
        return {next(iter(freq)): "0"}
    heap = [(w, i, s) for i, (s, w) in enumerate(freq.items())]   # leaf = tuple, internal = list
    heapq.heapify(heap)
    cnt = len(heap)
    while len(heap) > 1:
        w1, _, a = heapq.heappop(heap)
        w2, _, b = heapq.heappop(heap)
        heapq.heappush(heap, (w1 + w2, cnt, [a, b])); cnt += 1
    codes, stack = {}, [(heap[0][2], "")]
    while stack:
        node, prefix = stack.pop()
        if isinstance(node, list):
            stack.append((node[0], prefix + "0")); stack.append((node[1], prefix + "1"))
        else:
            codes[node] = prefix
    return codes

TABLE_BITS_PER_ENTRY = 24   # estimate: 16-bit symbol + 8-bit code length, stored in the header

# -------------------------------------------------- single-channel codec
def encode_channel(ch, qtable):
    h, w = ch.shape
    padded = pad_to_multiple(ch.astype(float) - 128.0)
    q = np.round(dct2(split_blocks(padded)) / qtable).astype(int)
    symbols, prev_dc = [], 0
    for blk in q:
        z = zigzag(blk)
        symbols.append(("D", int(z[0]) - prev_dc)); prev_dc = int(z[0])
        symbols += rle_ac(z[1:])
    codes = build_huffman(symbols)
    bits = "".join(codes[s] for s in symbols)
    return {"bits": bits, "codes": codes, "shape": (h, w), "padded": padded.shape, "q": qtable}

def decode_channel(enc):
    inv = {c: s for s, c in enc["codes"].items()}
    symbols, cur = [], ""
    for b in enc["bits"]:
        cur += b
        if cur in inv:
            symbols.append(inv[cur]); cur = ""
    ph, pw = enc["padded"]
    nblocks = (ph // 8) * (pw // 8)
    blocks, pos, prev_dc = [], 0, 0
    for _ in range(nblocks):
        prev_dc += symbols[pos][1]; pos += 1
        ac = []
        while len(ac) < 63:
            s = symbols[pos]; pos += 1
            if s[0] == "E":
                ac += [0] * (63 - len(ac))
            elif s[2] == 0:                       # ZRL
                ac += [0] * 16
            else:
                ac += [0] * s[1] + [s[2]]
        blocks.append(inverse_zigzag(np.array([prev_dc] + ac[:63], dtype=float)))
    rec = idct2(np.array(blocks) * enc["q"]) + 128.0
    h, w = enc["shape"]
    return merge_blocks(rec, (ph, pw))[:h, :w]

def channel_bits(enc):
    return len(enc["bits"]) + TABLE_BITS_PER_ENTRY * len(enc["codes"])

# -------------------------------------------------------- full images
def compress_gray(img, quality):
    enc = encode_channel(img, scale_qtable(Q_LUM, quality))
    return {"channels": [enc], "bits": channel_bits(enc), "shape": img.shape, "subsample": False}

def decompress_gray(c):
    return to_uint8(decode_channel(c["channels"][0]))

def _down2(ch):
    ch = np.pad(ch, ((0, ch.shape[0] % 2), (0, ch.shape[1] % 2)), mode="edge")
    return ch.reshape(ch.shape[0] // 2, 2, ch.shape[1] // 2, 2).mean((1, 3))

def _up2(ch, shape):
    return np.repeat(np.repeat(ch, 2, 0), 2, 1)[: shape[0], : shape[1]]

def compress_color(rgb, quality, subsample=True):
    ycc = rgb_to_ycbcr(rgb)
    y, cb, cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    if subsample:                                  # 4:2:0 chroma subsampling
        cb, cr = _down2(cb), _down2(cr)
    ql, qc = scale_qtable(Q_LUM, quality), scale_qtable(Q_CHR, quality)
    chans = [encode_channel(y, ql), encode_channel(cb, qc), encode_channel(cr, qc)]
    return {"channels": chans, "bits": sum(channel_bits(c) for c in chans),
            "shape": rgb.shape[:2], "subsample": subsample}

def decompress_color(c):
    y, cb, cr = [decode_channel(e) for e in c["channels"]]
    if c["subsample"]:
        cb, cr = _up2(cb, c["shape"]), _up2(cr, c["shape"])
    return ycbcr_to_rgb(np.stack([y, cb, cr], axis=-1))

def compression_ratio(shape, channels, bits):
    return shape[0] * shape[1] * channels * 8 / bits
