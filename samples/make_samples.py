"""Generate SYNTHETIC lab reports (PDF / PNG / JPEG) with known ground truth.

Used (a) to ship a few demo files in ./samples and (b) by tests/eval_extraction.py to measure
how well the Vision-Transformer reader recovers values. All patients are fictional and every
page carries a 'SAMPLE - synthetic data' banner.

    python samples/make_samples.py --out samples --n 6 --seed 1
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FD = "/usr/share/fonts/truetype"
EXTRA_FONTS = {  # fonts NOT used to train the ViT (held-out) -- registered when available
    "PoppinsMedium": f"{FD}/google-fonts/Poppins-Medium.ttf",
    "FreeSerifBold": f"{FD}/freefont/FreeSerifBold.ttf",
}
for _n, _p in EXTRA_FONTS.items():
    try:
        pdfmetrics.registerFont(TTFont(_n, _p))
    except Exception:
        pass
_AVAILABLE = set(pdfmetrics.getRegisteredFontNames())

FIRST = ["Aarav", "Priya", "Rahul", "Anita", "Sunita", "Neha", "Vikram", "Pooja", "Amit", "Meera", "John", "Mary", "David", "Sarah"]
LAST = ["Sharma", "Verma", "Gupta", "Singh", "Patel", "Kumar", "Reddy", "Nair", "Khan", "Smith", "Brown", "Tan"]
LABS = ["Sunrise Diagnostics", "Metro Pathology Lab", "CityCare Laboratories", "Greenfield Clinical Lab", "Apex Health Labs"]


def make_spec(rng: random.Random) -> dict:
    sex = rng.choice(["Female", "Male"])
    age = rng.randint(19, 79)
    hb = round(rng.choice([rng.uniform(4.6, 5.6), rng.uniform(5.7, 6.4), rng.uniform(6.5, 10.5)]), 1)
    glu = int(rng.choice([rng.uniform(78, 99), rng.uniform(100, 125), rng.uniform(126, 280)]))
    bmi_printed = rng.random() < 0.6
    height = rng.randint(150, 188)
    bmi_true = round(rng.uniform(18.5, 38.0), 1)
    weight = int(round(bmi_true * (height / 100) ** 2))
    bmi = bmi_true if bmi_printed else round(weight / (height / 100) ** 2, 1)
    return {
        "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}", "sex": sex, "age": age, "hba1c": hb, "glucose": glu,
        "glucose_kind": rng.choice(["Fasting Blood Glucose", "Random Blood Glucose", "Plasma Glucose (Fasting)"]),
        "bmi": bmi, "bmi_printed": bmi_printed, "height": height, "weight": weight,
        "lab": rng.choice(LABS), "date": f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2026",
        "labno": str(rng.randint(10000, 99999)),
        "age_fmt": rng.choice(["{a} Years / {s}", "{a}Y / {s0}", "{a} Yrs / {s}"]),
        "distractors": rng.sample([("Total Cholesterol", "186", "mg/dL", "< 200"), ("HDL Cholesterol", "48", "mg/dL", "> 40"),
                                   ("Triglycerides", "142", "mg/dL", "< 150"), ("Serum Creatinine", "0.9", "mg/dL", "0.6 - 1.2"),
                                   ("Haemoglobin", "13.4", "g/dL", "12.0 - 16.0"), ("Uric Acid", "5.2", "mg/dL", "3.5 - 7.2")], 3),
    }


def draw_report(path: Path, spec: dict, style: dict) -> None:
    W, H = 595, 842
    c = canvas.Canvas(str(path), pagesize=(W, H))
    reg, bold, fs = style["font"], style["bold"], style["size"]
    # header
    if style["dark_header"]:
        c.setFillColor(colors.HexColor("#14405f")); c.rect(0, H - 72, W, 72, fill=1, stroke=0)
        c.setFillColor(colors.white); c.setFont(bold, 18); c.drawString(40, H - 40, spec["lab"])
        c.setFont(reg, 9); c.drawString(40, H - 58, "Clinical Biochemistry - SAMPLE - synthetic data")
    else:
        c.setFillColor(colors.black); c.setFont(bold, 18); c.drawString(40, H - 45, spec["lab"])
        c.setFont(reg, 9); c.drawString(40, H - 60, "Clinical Biochemistry - SAMPLE - synthetic data")
        c.setLineWidth(1); c.line(40, H - 68, W - 40, H - 68)
    c.setFillColor(colors.black)
    # patient block
    y = H - 100
    sex_s = spec["sex"]
    age_s = spec["age_fmt"].format(a=spec["age"], s=sex_s, s0=sex_s[0])
    left = [("Patient Name", spec["name"]), ("Age / Sex", age_s)]
    right = [("Report Date", spec["date"]), ("Lab No", spec["labno"])]
    c.setFont(reg, fs + 1)
    for i, ((k1, v1), (k2, v2)) in enumerate(zip(left, right)):
        c.drawString(40, y - i * 18, f"{k1}: {v1}")
        c.drawString(330, y - i * 18, f"{k2}: {v2}")
    y -= 56
    # table
    cols = [40, 235, 320, 385, 500]
    heads = ["Test Name", "Result", "Unit", "Reference Range", "Flag"]
    rows = [(spec["glucose_kind"], str(spec["glucose"]), "mg/dL", "70 - 99" if "Fasting" in spec["glucose_kind"] else "< 140",
             "H" if spec["glucose"] >= 126 else ""),
            ("HbA1c (Glycated Hemoglobin)", f"{spec['hba1c']:.1f}", "%", "4.0 - 5.6", "H" if spec["hba1c"] >= 5.7 else "")]
    rows += [(n, v, u, r, "") for n, v, u, r in spec["distractors"]]
    if spec["bmi_printed"]:
        rows.append(("Body Mass Index (BMI)", f"{spec['bmi']:.1f}", "kg/m2", "18.5 - 24.9", ""))
    else:
        rows.append(("Height", str(spec["height"]), "cm", "", ""))
        rows.append(("Weight", str(spec["weight"]), "kg", "", ""))
    rh = 24
    if style["borders"]:
        c.setFillColor(colors.HexColor("#dfe9f0")); c.rect(36, y - 4, W - 72, rh, fill=1, stroke=0); c.setFillColor(colors.black)
    c.setFont(bold, fs)
    for x, h in zip(cols, heads):
        c.drawString(x, y + 3, h)
    y -= rh
    c.setFont(reg, fs)
    top = y + rh
    for r in rows:
        for x, v in zip(cols, r):
            c.drawString(x, y + 3, v)
        if style["borders"]:
            c.setLineWidth(0.6); c.line(36, y - 4, W - 36, y - 4)
        y -= rh
    if style["borders"]:
        c.setLineWidth(0.6); c.line(36, top + rh - 4 + 0, W - 36, top + rh - 4)
        for x in [36] + [x - 6 for x in cols[1:]] + [W - 36]:
            c.line(x, top + rh - 4, x, y + rh - 4)
    y -= 24
    c.setFont(reg, fs - 1)
    c.drawString(40, y, "Interpretation: please correlate clinically. SAMPLE report - not a real patient.")
    c.drawString(40, y - 40, "End of Report")
    c.showPage(); c.save()


def degrade(img: Image.Image, rng: random.Random, kind: str) -> Image.Image:
    a = np.array(img.convert("L"))
    if kind == "photo":
        h, w = a.shape
        ang = rng.uniform(-2.0, 2.0)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        a = cv2.warpAffine(a, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=235)
        grad = np.linspace(rng.uniform(-30, 10), rng.uniform(-10, 25), w, dtype=np.float32)[None, :]
        a = np.clip(a.astype(np.float32) + grad + np.random.default_rng(rng.randint(0, 9999)).normal(0, 5, a.shape), 0, 255).astype(np.uint8)
        a = cv2.GaussianBlur(a, (3, 3), 0.8)
    return Image.fromarray(a)


def render_set(out: Path, n: int, seed: int, formats=("pdf", "png", "jpg"), base_fonts: bool = True) -> list[dict]:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    fonts = [("Helvetica", "Helvetica-Bold"), ("Times-Roman", "Times-Bold"), ("Courier", "Courier-Bold")]
    if "PoppinsMedium" in _AVAILABLE:
        fonts.append(("PoppinsMedium", "PoppinsMedium"))
    if "FreeSerifBold" in _AVAILABLE:
        fonts.append(("FreeSerifBold", "FreeSerifBold"))
    truth = []
    for i in range(n):
        spec = make_spec(rng)
        reg, bold = fonts[i % len(fonts)]
        style = {"font": reg, "bold": bold, "size": rng.choice([9, 10, 11]), "dark_header": rng.random() < 0.5,
                 "borders": rng.random() < 0.6}
        fmt = formats[i % len(formats)]
        stem = f"sample_{i + 1:02d}_{spec['sex'].lower()}"
        pdf_path = out / f"{stem}.pdf"
        draw_report(pdf_path, spec, style)
        if fmt == "pdf":
            fname = pdf_path.name
        else:
            page = pdfium.PdfDocument(str(pdf_path))[0].render(scale=2.0).to_pil()
            img = degrade(page, rng, "photo" if fmt == "jpg" else "scan")
            fname = f"{stem}.{fmt}"
            if fmt == "png":
                img.save(out / fname)
            else:
                img.save(out / fname, quality=rng.randint(55, 85))
            pdf_path.unlink()
        truth.append({"file": fname, "font": reg, "borders": style["borders"], "dark_header": style["dark_header"],
                      "age": spec["age"], "gender": spec["sex"], "hba1c": spec["hba1c"], "glucose": spec["glucose"], "bmi": spec["bmi"]})
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    return truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    t = render_set(Path(a.out), a.n, a.seed)
    print(f"wrote {len(t)} sample reports to {a.out}")
