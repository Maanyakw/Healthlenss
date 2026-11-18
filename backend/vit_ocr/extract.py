"""Lab-report reader: PDF / PNG / JPEG -> text lines recognised by the Vision Transformer.

Pipeline: load pages -> scale + deskew -> invert dark header bars -> remove table rules ->
split into text rows -> split rows into phrase/cell chunks -> ViT-CTC reads every chunk ->
lines are re-assembled left-to-right, top-to-bottom. `parse.py` then turns lines into fields.
"""
from __future__ import annotations

import base64
import io
import threading

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from .. import config
from . import preprocess as P
from .model import ViTCTC, greedy_decode

Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS

TARGET_W = 1600
MAX_CHUNKS_PER_PAGE = 700


class UnsupportedFile(ValueError):
    pass


# ------------------------------------------------------------------ loading
def load_pages(data: bytes) -> list[np.ndarray]:
    """Return a list of grayscale uint8 page images."""
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise UnsupportedFile(f"File is larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if data[:4] == b"%PDF":
        import pypdfium2 as pdfium
        try:
            pdf = pdfium.PdfDocument(data)
        except Exception as e:  # encrypted / corrupt
            raise UnsupportedFile("The PDF could not be opened (it may be password-protected or damaged).") from e
        pages = []
        for i in range(min(len(pdf), config.MAX_PDF_PAGES)):
            img = pdf[i].render(scale=2.5).to_pil().convert("RGB")
            pages.append(np.array(ImageOps.grayscale(img)))
        if not pages:
            raise UnsupportedFile("The PDF has no pages.")
        return pages
    if data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff":
        try:
            img = Image.open(io.BytesIO(data))
            if img.width * img.height > config.MAX_IMAGE_PIXELS:
                raise UnsupportedFile("The image has too many pixels.")
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img)
            return [np.array(ImageOps.grayscale(img.convert("RGB")))]
        except UnsupportedFile:
            raise
        except Exception as e:
            raise UnsupportedFile("The image could not be read.") from e
    raise UnsupportedFile("Unsupported file type. Please upload a PDF, PNG or JPEG.")


# ------------------------------------------------------------------ page conditioning
def rescale(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    if w < 1200 or w > 2000:
        f = TARGET_W / w
        g = cv2.resize(g, (TARGET_W, int(round(h * f))), interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
    return g


def deskew(g: np.ndarray) -> np.ndarray:
    small_w = 700
    f = small_w / g.shape[1]
    s = cv2.resize(g, (small_w, int(g.shape[0] * f)), interpolation=cv2.INTER_AREA)
    _, b = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = b.shape

    def score(angle: float) -> float:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        r = cv2.warpAffine(b, M, (w, h), flags=cv2.INTER_NEAREST)
        p = r.sum(axis=1).astype(np.float64)
        return float(np.var(np.diff(p)))

    coarse = max(np.arange(-6, 6.1, 1.0), key=score)
    fine = max(np.arange(coarse - 0.8, coarse + 0.81, 0.2), key=score)
    if abs(fine) < 0.3:
        return g
    H, W = g.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), fine, 1.0)
    bg = int(np.median(g))
    return cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=bg)


def invert_dark_blocks(g: np.ndarray) -> np.ndarray:
    """White-on-dark header bars -> dark-on-light so the same recogniser can read them."""
    dark = (g < 100).astype(np.uint8)
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    out = g.copy()
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w > 60 and 12 <= h <= 160 and area > 0:
            fill = dark[y:y + h, x:x + w].mean()
            if fill > 0.38:
                out[y:y + h, x:x + w] = 255 - g[y:y + h, x:x + w]
    return out


def ink_and_rules(g: np.ndarray):
    """Binary ink with table rules removed; returns (ink bool, rules mask uint8)."""
    h, w = g.shape
    block = max(25, (w // 60) | 1)
    ink = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 12)
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(45, w // 22), 1)))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 42)))
    rules = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    ink[rules > 0] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, bool)
    a, ch = stats[:, cv2.CC_STAT_AREA], stats[:, cv2.CC_STAT_HEIGHT]
    keep[1:] = (a[1:] >= 5) & (ch[1:] <= 120)
    return keep[lab], rules


def runs(mask1d: np.ndarray) -> list[tuple[int, int]]:
    """Consecutive True runs as (start, end_exclusive)."""
    if not mask1d.any():
        return []
    d = np.diff(np.concatenate([[0], mask1d.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def merge_runs(rs: list[tuple[int, int]], max_gap: float) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in rs:
        if out and s - out[-1][1] <= max_gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def find_chunks(ink: np.ndarray):
    """Yield (y0, y1, [ (x0, x1) ... ]) for each text row; chunks are phrases / table cells."""
    rows = runs(ink.sum(axis=1) >= 2)
    heights = [e - s for s, e in rows if e - s >= 8]
    if not heights:
        return []
    med = float(np.median(heights))
    bands = [(s, e) for s, e in merge_runs(rows, max(2.0, 0.18 * med)) if 7 <= e - s <= 100]
    result = []
    for y0, y1 in bands:
        band = ink[y0:y1]
        col_runs = runs(band.any(axis=0))
        if not col_runs:
            continue
        bh = y1 - y0
        chunks = merge_runs(col_runs, max(6.0, 0.7 * bh))
        result.append((y0, y1, chunks, col_runs))
    return result


def split_wide(chunk, col_runs, ink, y0, y1):
    """Split a chunk whose normalised width would exceed the model canvas, at its widest inner gap."""
    x0, x1 = chunk
    inner = [r for r in col_runs if r[0] >= x0 and r[1] <= x1]
    rows_used = np.where(ink[y0:y1, x0:x1].any(axis=1))[0]
    if len(rows_used) == 0:
        return []
    h = int(rows_used[-1] - rows_used[0] + 1)
    if P.scaled_width(x1 - x0, h) <= P.W - 4 or len(inner) < 2:
        return [chunk] if P.scaled_width(x1 - x0, h) <= P.W - 4 else []
    gaps = [(inner[i + 1][0] - inner[i][1], i) for i in range(len(inner) - 1)]
    _, k = max(gaps)
    left = (inner[0][0], inner[k][1])
    right = (inner[k + 1][0], inner[-1][1])
    return split_wide(left, col_runs, ink, y0, y1) + split_wide(right, col_runs, ink, y0, y1)


# ------------------------------------------------------------------ recogniser
class LabReportReader:
    def __init__(self) -> None:
        ckpt = torch.load(config.VIT_WEIGHTS_PATH, map_location="cpu")
        self.model = ViTCTC(**ckpt["cfg"])
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()
        self._lock = threading.Lock()  # one inference at a time on small servers

    @torch.no_grad()
    def recognise(self, canvases: list[np.ndarray]) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        with self._lock:
            for i in range(0, len(canvases), 64):
                x = torch.from_numpy(np.stack(canvases[i:i + 64])).float().div_(255.0).unsqueeze(1)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    logits = self.model(x)
                out.extend(greedy_decode(logits.float()))
        return out

    def read_page(self, g: np.ndarray, page_no: int = 0):
        g = deskew(rescale(g))
        g = invert_dark_blocks(g)
        ink, rules = ink_and_rules(g)
        work = cv2.inpaint(g, rules, 3, cv2.INPAINT_TELEA) if rules.any() else g
        H, W = g.shape
        placed = []  # (y0, y1, x0, x1)
        canvases = []
        for y0, y1, chunks, col_runs in find_chunks(ink):
            for ch in chunks:
                for (x0, x1) in split_wide(ch, col_runs, ink, y0, y1):
                    crop = work[max(0, y0 - 3):min(H, y1 + 3), max(0, x0 - 3):min(W, x1 + 3)]
                    canvas = P.normalize_chunk(crop)
                    if canvas is not None:
                        canvases.append(canvas)
                        placed.append((y0, y1, x0, x1))
        truncated = len(canvases) > MAX_CHUNKS_PER_PAGE
        canvases, placed = canvases[:MAX_CHUNKS_PER_PAGE], placed[:MAX_CHUNKS_PER_PAGE]
        texts = self.recognise(canvases) if canvases else []

        # group chunks into lines by their band (y0, y1)
        lines: list[dict] = []
        for (y0, y1, x0, x1), (txt, conf) in zip(placed, texts):
            txt = txt.strip()
            if not txt:
                continue
            if lines and lines[-1]["_y"] == (y0, y1):
                L = lines[-1]
            else:
                L = {"page": page_no, "_y": (y0, y1), "chunks": []}
                lines.append(L)
            L["chunks"].append({"x0": x0 / W, "x1": x1 / W, "text": txt, "conf": round(conf, 3)})
        out = []
        for L in lines:
            ch = sorted(L["chunks"], key=lambda c: c["x0"])
            weights = [max(len(c["text"]), 1) for c in ch]
            conf = float(np.average([c["conf"] for c in ch], weights=weights))
            out.append({
                "page": L["page"],
                "text": "   ".join(c["text"] for c in ch),
                "conf": round(conf, 3),
                "y0": L["_y"][0] / H, "y1": L["_y"][1] / H,
                "x0": ch[0]["x0"], "x1": ch[-1]["x1"],
                "chunks": ch,
            })
        return g, out, truncated

    def read(self, data: bytes) -> dict:
        pages_raw = load_pages(data)
        all_lines, previews, truncated = [], [], False
        for i, g0 in enumerate(pages_raw):
            g, lines, trunc = self.read_page(g0, i)
            truncated |= trunc
            all_lines.extend(lines)
            f = min(1.0, 1000 / g.shape[1])
            small = cv2.resize(g, (int(g.shape[1] * f), int(g.shape[0] * f)), interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
            previews.append({"image": "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode(),
                             "aspect": g.shape[0] / g.shape[1]})
        return {"pages": previews, "lines": all_lines, "truncated": truncated}
