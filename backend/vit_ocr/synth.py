"""Synthetic training data for the ViT recogniser: lab-report-like text chunks rendered with many
fonts, sizes, colours, blur, noise, rotation and JPEG artefacts.

Fonts are split into TRAIN_FONTS and HELD_OUT_FONTS (never seen in training) so that the
generalisation to unseen typefaces can be measured honestly.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import preprocess as P
from .charset import CHAR2IDX, clean

FD = "/usr/share/fonts/truetype"
TRAIN_FONTS = [
    f"{FD}/dejavu/DejaVuSans.ttf", f"{FD}/dejavu/DejaVuSans-Bold.ttf", f"{FD}/dejavu/DejaVuSansCondensed.ttf",
    f"{FD}/dejavu/DejaVuSansMono.ttf", f"{FD}/dejavu/DejaVuSerif.ttf",
    f"{FD}/freefont/FreeSans.ttf", f"{FD}/freefont/FreeSansBold.ttf", f"{FD}/freefont/FreeSerif.ttf",
    f"{FD}/freefont/FreeMono.ttf",
    f"{FD}/liberation/LiberationSans-Regular.ttf", f"{FD}/liberation/LiberationSans-Bold.ttf",
    f"{FD}/liberation/LiberationSansNarrow-Regular.ttf", f"{FD}/liberation/LiberationSerif-Regular.ttf",
    f"{FD}/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/opentype/inter/Inter-Regular.otf", "/usr/share/fonts/opentype/inter/Inter-Medium.otf",
    f"{FD}/google-fonts/Poppins-Regular.ttf", f"{FD}/google-fonts/ofl/cardo/Cardo-Regular.ttf",
    f"{FD}/google-fonts/ofl/sourcesans3/SourceSans3[wght].ttf", f"{FD}/google-fonts/ofl/opensans/OpenSans[wdth,wght].ttf",
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf",
    "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf",
]
HELD_OUT_FONTS = [
    f"{FD}/google-fonts/ofl/roboto/Roboto[wdth,wght].ttf", f"{FD}/google-fonts/ofl/gelasio/Gelasio[wght].ttf",
    f"{FD}/google-fonts/ofl/dmsans/DMSans[opsz,wght].ttf",
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-regular.otf",
    f"{FD}/google-fonts/Poppins-Medium.ttf", f"{FD}/freefont/FreeSerifBold.ttf",
]

TESTS = [
    "HbA1c", "HbA1C", "Hb A1c", "HbA1c (Glycated Hemoglobin)", "Glycated Haemoglobin", "Glycated Hemoglobin (HbA1c)",
    "Glycosylated Hemoglobin", "Hemoglobin A1c", "Glycohemoglobin", "Fasting Blood Glucose", "Fasting Plasma Glucose",
    "Blood Glucose (Fasting)", "Glucose, Fasting", "Random Blood Glucose", "Random Blood Sugar", "Blood Sugar (Random)",
    "Post Prandial Glucose", "Plasma Glucose (Random)", "FBS", "RBS", "PPBS", "Blood Glucose", "Glucose", "Blood Sugar Fasting",
    "Body Mass Index", "BMI", "BMI (kg/m2)", "Height", "Weight", "Age", "Sex", "Gender", "Age/Sex", "Patient Name", "Patient ID",
    "Hypertension", "Heart Disease", "Smoking Status", "Total Cholesterol", "HDL Cholesterol", "LDL Cholesterol",
    "Triglycerides", "Serum Creatinine", "Blood Urea", "Haemoglobin", "WBC Count", "Platelet Count", "Uric Acid",
    "SGPT (ALT)", "SGOT (AST)", "TSH", "Vitamin D", "Sodium", "Potassium", "Calcium", "Urine Albumin", "eGFR",
    "Mean Blood Glucose", "Estimated Average Glucose", "Insulin Fasting", "C-Peptide", "Microalbumin",
]
HEADERS = [
    "Test Name", "Result", "Unit", "Units", "Reference Range", "Bio. Ref. Interval", "Method", "Flag", "Investigation",
    "Observed Value", "Normal Range", "Remarks", "Specimen", "Sample Type", "Report Date", "Collected On", "Reported On",
    "Page 1 of 2", "Laboratory Report", "Pathology Laboratory", "Clinical Biochemistry", "Diagnostic Centre",
    "Reg. No.", "Lab No", "Ref. By", "Printed on", "Whole Blood EDTA", "Serum", "Plasma (Fluoride)", "HPLC",
    "Hexokinase", "GOD-POD", "Enzymatic", "Immunoturbidimetry", "End of Report", "Interpretation:", "Note:", "Comments:",
    "SAMPLE - synthetic data", "Department of Biochemistry", "Consultant Pathologist", "Checked by", "Test Report",
    "Please correlate clinically", "Results are for the doctor", "Status: Final", "Sample Collected", "Biochemistry",
]
FIRST = ["Aarav", "Priya", "Rahul", "Anita", "Sunita", "Rajesh", "Neha", "Vikram", "Pooja", "Amit", "Divya", "Karan", "Meera",
         "Sanjay", "Kavita", "Arjun", "Anjali", "Rohit", "Sneha", "Manoj", "John", "Mary", "David", "Sarah", "Michael", "Emma",
         "Daniel", "Olivia", "James", "Sophia", "Ahmed", "Fatima", "Wei", "Siti", "Lakshmi", "Deepak", "Ritu", "Maria", "Omar"]
LAST = ["Sharma", "Verma", "Gupta", "Singh", "Dixit", "Patel", "Kumar", "Reddy", "Nair", "Iyer", "Khan", "Das", "Mehta",
        "Joshi", "Smith", "Johnson", "Brown", "Lee", "Tan", "Wong", "Ali", "Rao", "Mishra", "Chopra", "Garcia", "Silva"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WORDS = ["the", "patient", "report", "sample", "value", "range", "high", "low", "normal", "borderline", "fasting", "random",
         "blood", "serum", "test", "result", "clinical", "doctor", "advised", "repeat", "diabetes", "risk", "control", "good",
         "poor", "monitor", "level", "average", "months", "weeks", "ADA", "WHO", "criteria", "above", "below", "within"]


def _num(r: random.Random, lo: float, hi: float, nd: int) -> str:
    v = r.uniform(lo, hi)
    return f"{v:.{nd}f}" if nd else str(int(round(v)))


def gen_value(r: random.Random) -> str:
    kind = r.random()
    if kind < 0.28:      # HbA1c-like
        n = _num(r, 3.5, 14.5, 1 if r.random() < 0.85 else 2)
        return r.choice([f"{n}", f"{n} %", f"{n}%", f"{n} % ", f"{n} mmol/mol" if float(n) > 20 else f"{n} %"]).strip()
    if kind < 0.58:      # glucose-like
        if r.random() < 0.8:
            n = _num(r, 40, 520, 0)
            return r.choice([n, f"{n} mg/dL", f"{n} mg/dl", f"{n}mg/dL", f"{n} MG/DL"])
        n = _num(r, 2.2, 28, 1)
        return r.choice([n, f"{n} mmol/L", f"{n} mmol/l"])
    if kind < 0.72:      # BMI / height / weight
        c = r.random()
        if c < 0.5:
            n = _num(r, 12, 55, r.choice([1, 1, 2]))
            return r.choice([n, f"{n} kg/m2", f"{n} Kg/m2"])
        if c < 0.75:
            return r.choice([f"{_num(r, 90, 205, 0)} cm", f"{_num(r, 90, 205, 0)} cms", f"{_num(r, 3.0, 6.9, 1)} ft"])
        return r.choice([f"{_num(r, 10, 160, 0)} kg", f"{_num(r, 10, 160, 1)} kg", f"{_num(r, 20, 350, 0)} lbs"])
    if kind < 0.84:      # age / sex
        a = _num(r, 1, 99, 0)
        return r.choice([f"{a} Years", f"{a} Yrs", f"{a} Y", f"{a}Y/F", f"{a}Y/M", f"{a} Years / Female", f"{a} Y / Male",
                         "Male", "Female", "M", "F", f"Age: {a}", f"Age: {a} Years", f"Sex: {r.choice(['Male', 'Female', 'M', 'F'])}",
                         f"Gender: {r.choice(['Male', 'Female'])}"])
    if kind < 0.92:
        return r.choice(["H", "L", "High", "Low", "Normal", "(H)", "(L)", "Yes", "No", "Never", "Current", "Former", "Borderline", "Nil"])
    return _num(r, 0, 9999, r.choice([0, 1, 2]))


def gen_range(r: random.Random) -> str:
    c = r.random()
    if c < 0.3:
        a, b = _num(r, 3.5, 5.0, 1), _num(r, 5.4, 6.1, 1)
        return r.choice([f"{a} - {b}", f"{a}-{b}", f"{a} - {b} %", f"Ref: {a} - {b} %", f"Normal: {a} - {b}"])
    if c < 0.55:
        a, b = _num(r, 60, 80, 0), _num(r, 99, 140, 0)
        return r.choice([f"{a} - {b}", f"{a}-{b}", f"{a} - {b} mg/dL", f"Ref: {a} - {b} mg/dL", f"({a} - {b} mg/dL)"])
    if c < 0.7:
        return r.choice(["< 5.7 %", "< 5.7", "Normal: < 5.7 %", "Prediabetes: 5.7 - 6.4 %", "Diabetes: >= 6.5 %", "> 6.5",
                         "< 100 mg/dL", "Normal: < 100 mg/dL", "Prediabetes: 100 - 125", "Diabetes: >= 126 mg/dL", "18.5 - 24.9",
                         "Normal: 18.5 - 24.9", "< 200 mg/dL", "70 - 140 mg/dL", "3.9 - 5.5 mmol/L", "4.0 - 5.6 %"])
    a = _num(r, 0.1, 400, r.choice([0, 1, 2]))
    b = _num(r, 0.1, 400, r.choice([0, 1, 2]))
    return r.choice([f"{a} - {b}", f"< {a}", f"> {a}", f"{a}-{b} g/dL", f"Ref: {a} - {b}"])


def gen_misc(r: random.Random) -> str:
    c = r.random()
    if c < 0.3:
        return f"{r.randint(1, 28):02d}/{r.randint(1, 12):02d}/{r.randint(2019, 2026)}"
    if c < 0.4:
        return f"{r.randint(1, 28):02d}-{r.choice(MONTHS)}-{r.randint(2019, 2026)}"
    if c < 0.45:
        return f"{r.randint(2019, 2026)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"
    if c < 0.6:
        return r.choice(["PID-", "Lab No: ", "Reg No: ", "Sample ID: ", "UHID ", "Ref. No. ", "ID: "]) + r.choice(
            [str(r.randint(1000, 99999999)), f"{r.randint(10, 99)}-{r.randint(1000, 99999)}",
             f"{r.choice('ABCDEFGHKLMNPRSTUVWXYZ')}{r.randint(100, 999999)}"])
    if c < 0.75:
        t = r.choice(["Mr. ", "Mrs. ", "Ms. ", "Dr. ", "", "", "Name: ", "Patient: "])
        return f"{t}{r.choice(FIRST)} {r.choice(LAST)}"
    if c < 0.85:
        return " ".join(r.choice(WORDS) for _ in range(r.randint(1, 4))).capitalize()
    n = r.randint(3, 16)
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,:;-/()%<>=+_'#"
    return "".join(r.choice(alphabet) for _ in range(n))


def gen_line(r: random.Random, max_chars: int = 30) -> str:
    c = r.random()
    if c < 0.22:
        s = f"{r.choice(TESTS)}{r.choice([':', '', ' :', ' -'])} {gen_value(r)}"
    elif c < 0.45:
        s = gen_value(r)
    elif c < 0.57:
        s = r.choice(TESTS)
    elif c < 0.67:
        s = gen_range(r)
    elif c < 0.77:
        s = r.choice(HEADERS)
    elif c < 0.87:
        s = gen_misc(r)
    else:  # several cell-like pieces separated by 1-3 spaces (columns that were not split)
        parts = [r.choice([gen_value, gen_range, gen_misc, lambda rr: rr.choice(TESTS)])(r) for _ in range(r.randint(2, 3))]
        s = (" " * r.randint(1, 3)).join(parts)
    s = clean(s)
    if len(s) > max_chars:
        cut = s[:max_chars].rsplit(" ", 1)[0]
        s = cut if len(cut) >= 3 else s[:max_chars]
    return s.strip()


# ------------------------------------------------------------------ rendering
_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _font(path: str, size: int):
    k = (path, size)
    if k not in _font_cache:
        _font_cache[k] = ImageFont.truetype(path, size)
    return _font_cache[k]


def render(text: str, font_path: str, rng: np.random.Generator, r: random.Random) -> np.ndarray:
    size = int(rng.integers(16, 64))
    draw_text = text
    if " - " in text and r.random() < 0.3:
        draw_text = text.replace(" - ", " – ")
    font = _font(font_path, size)
    inverted = r.random() < 0.06
    if inverted:
        bg, fg = int(rng.integers(5, 90)), int(rng.integers(190, 256))
    else:
        bg = int(rng.integers(195, 256))
        fg = int(rng.integers(0, 80)) if r.random() < 0.9 else int(rng.integers(80, 140))
    probe = ImageDraw.Draw(Image.new("L", (4, 4)))
    l, t, rgt, b = probe.textbbox((0, 0), draw_text, font=font)
    px = int(rng.integers(int(size * 0.2), int(size * 0.7)))
    py = int(rng.integers(int(size * 0.15), int(size * 0.5)))
    img = Image.new("L", (rgt - l + 2 * px, b - t + 2 * py), bg)
    d = ImageDraw.Draw(img)
    d.text((px - l, py - t), draw_text, font=font, fill=fg)
    a = np.array(img)
    # stray table borders (the app removes most of them, but not always)
    if r.random() < 0.05:
        thick = int(rng.integers(1, 3))
        y = int(rng.choice([max(0, py // 2), a.shape[0] - 1 - py // 2]))
        a[y:y + thick, :] = fg
    return a


def augment(a: np.ndarray, rng: np.random.Generator, r: random.Random) -> np.ndarray:
    h, w = a.shape
    bg = int(np.median(a))
    if r.random() < 0.4:  # rotation / shear
        ang = float(rng.uniform(-1.5, 1.5))
        shear = float(rng.uniform(-0.12, 0.12)) if r.random() < 0.25 else 0.0
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        M[0, 1] += shear
        a = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    if r.random() < 0.6:  # resolution loss
        f = float(rng.uniform(0.35, 0.9))
        small = cv2.resize(a, (max(8, int(w * f)), max(4, int(h * f))), interpolation=cv2.INTER_AREA)
        a = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR if r.random() < 0.5 else cv2.INTER_CUBIC)
    if r.random() < 0.35:
        k = int(rng.choice([3, 5]))
        a = cv2.GaussianBlur(a, (k, k), float(rng.uniform(0.3, 1.3)))
    a = a.astype(np.float32)
    if r.random() < 0.15:  # shading gradient
        grad = np.linspace(rng.uniform(-35, 35), rng.uniform(-35, 35), w, dtype=np.float32)[None, :]
        a += grad
    if r.random() < 0.5:
        a += rng.normal(0, rng.uniform(2, 14), a.shape).astype(np.float32)
    if r.random() < 0.1:
        m = rng.random(a.shape)
        a[m < 0.004] = 0
        a[m > 0.996] = 255
    c = float(rng.uniform(0.75, 1.1))
    a = (a - 128) * c + 128 + float(rng.uniform(-15, 15))
    a = np.clip(a, 0, 255).astype(np.uint8)
    if r.random() < 0.5:
        q = int(rng.integers(25, 95))
        ok, enc = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            a = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return a


def make_sample(rng: np.random.Generator, r: random.Random, fonts: list[str]):
    """Returns (canvas uint8 [H, W], label str) or None when the sample is unusable."""
    for _ in range(5):
        text = gen_line(r)
        if len(text) < 1:
            continue
        a = augment(render(text, r.choice(fonts), rng, r), rng, r)
        canvas = P.normalize_chunk(a)
        if canvas is not None:
            return canvas, text
    return None


def generate_dataset(n: int, seed: int, fonts: list[str]) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    r = random.Random(seed)
    fonts = [f for f in fonts if Path(f).exists()]
    X = np.zeros((n, P.H, P.W), np.uint8)
    labels: list[str] = []
    i = 0
    while i < n:
        s = make_sample(rng, r, fonts)
        if s is None:
            continue
        X[i], lab = s
        labels.append(lab)
        i += 1
    return X, labels


def _worker(args):
    n, seed, fonts = args
    return generate_dataset(n, seed, fonts)


def generate_parallel(n: int, seed: int, fonts: list[str], procs: int = 2) -> tuple[np.ndarray, list[str]]:
    import multiprocessing as mp
    per = n // procs
    jobs = [(per + (n - per * procs if k == procs - 1 else 0), seed * 1000 + k, fonts) for k in range(procs)]
    with mp.get_context("fork").Pool(procs) as pool:
        parts = pool.map(_worker, jobs)
    return np.concatenate([p[0] for p in parts]), sum((p[1] for p in parts), [])


assert all(c in CHAR2IDX for c in " 0123456789")
