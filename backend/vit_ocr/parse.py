"""Turn recognised report lines into patient fields (HbA1c, glucose, BMI, age, sex, ...).

Only fields that are actually found are returned; nothing is guessed. Every field carries the
source line and its recognition confidence so the UI can ask the user to verify it. Names,
IDs, dates and other personal details are deliberately NOT extracted.
"""
from __future__ import annotations

import re

NUM = r"\d{1,4}(?:\.\d{1,2})?"

HBA1C_KW = re.compile(
    r"(?:glyc\w*\s+)?h(?:a)?emoglobin\s*a\s*1\s*c|hb\s*a\s*[1il|]\s*c|hba\s*1\s*c|\ba\s*1\s*c\b|glycohemoglobin|"
    r"glycat\w+\s+h\w+|glycosylat\w+\s+h\w+|glyco\w+", re.I)
HEMOGLOBIN_WORDS = re.compile(r"h(?:a)?emoglobin", re.I)
GLUC_KW = re.compile(r"gluc\w*|sugar|\bfbs\b|\brbs\b|\bppbs\b|\bfbg\b|\bppg\b|\bfpg\b", re.I)
GLUC_SKIP = re.compile(r"average|mean|\beag\b|tolerance|ogtt|urine|csf|glyc|tolerance|insulin|peptide", re.I)
BMI_KW = re.compile(r"\bbmi\b|\bbim\b|body\s*ma\w*\s*\w{0,3}dex", re.I)
_CUT = re.compile(
    r"\b(?:ref|reference|bio|range|prediabetes|desirable|optimal|target|interval)\b|normal\s*:|diabetes\s*:|[<>]|"
    r"\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?", re.I)
_UNIT_NOISE = re.compile(r"kg\s*/\s*m\s*2|mg\s*/\s*d\s*l|mmol\s*/\s*[lm]\w*|g\s*/\s*d\s*l", re.I)


def _fix_decimals(s: str) -> str:
    return re.sub(r"(\d),(\d{1,2})(?!\d)", r"\1.\2", s)


def _values_after(text: str, kw_end: int, strip_kw: re.Pattern | None = None) -> list[float]:
    """Numbers that follow the keyword, before any reference-range text."""
    s = _fix_decimals(text[kw_end:])
    if strip_kw is not None:
        s = strip_kw.sub(" ", s)
    s = HEMOGLOBIN_WORDS.sub(" ", s)
    m = _CUT.search(s)
    if m:
        s = s[:m.start()]
    s = _UNIT_NOISE.sub(" ", s)
    return [float(x) for x in re.findall(rf"(?<![\d.]){NUM}(?![\d])", s)]


def _field(value, line_idx: int, line: dict, note: str = "") -> dict:
    return {"value": value, "confidence": line["conf"], "line_index": line_idx,
            "source_text": line["text"], "note": note}


def _next_line_values(lines: list[dict], i: int, lo: float, hi: float):
    """Value on the line directly below the label (stacked layouts)."""
    if i + 1 < len(lines) and lines[i + 1]["page"] == lines[i]["page"]:
        for v in _values_after(lines[i + 1]["text"], 0):
            if lo <= v <= hi:
                return v, i + 1
    return None, None


def _sex_word(w: str):
    """Classify an OCR'd token as a sex word, tolerating misread characters ('Femrale', 'Pemale')."""
    import difflib
    if w in ("f", "female"):
        return "Female"
    if w in ("m", "male"):
        return "Male"
    if len(w) < 3 or len(w) > 7:
        return None
    rf = difflib.SequenceMatcher(None, w, "female").ratio()
    rm = difflib.SequenceMatcher(None, w, "male").ratio()
    if len(w) >= 5 and rf >= 0.7 and rf > rm:
        return "Female"
    if len(w) <= 5 and rm >= 0.85 and rm > rf:
        return "Male"
    return None


def parse_fields(lines: list[dict]) -> dict:
    out: dict = {k: None for k in ("age", "gender", "hba1c", "glucose", "bmi", "hypertension", "heart_disease", "smoking_history")}
    glucose_candidates: list[dict] = []

    height_cm = weight_kg = None
    height_line = weight_line = None

    for i, ln in enumerate(lines):
        t = ln["text"]
        low = t.lower()

        # ---- HbA1c ------------------------------------------------------------------
        if out["hba1c"] is None:
            m = HBA1C_KW.search(t)
            if m and not re.search(r"\bhb\b\s*[:\-]?\s*\d", low):
                vals = _values_after(t, m.end(), HBA1C_KW)
                val, src = None, i
                for v in vals:
                    if 3.0 <= v <= 20.0 or 20.0 < v <= 200.0:
                        val = v
                        break
                if val is None:
                    nv, ni = _next_line_values(lines, i, 3.0, 200.0)
                    if nv is not None:
                        val, src = nv, ni
                if val is not None:
                    note = ""
                    ifcc = "mmol/mol" in low or "mmol / mol" in low or (val > 20.0 and "%" not in low)
                    if ifcc:
                        val = round(val / 10.929 + 2.15, 1)
                        note = "Converted from mmol/mol (IFCC) to % (NGSP)."
                    out["hba1c"] = _field(round(val, 2), src, lines[src], note)
                    continue

        # ---- blood glucose ------------------------------------------------------------
        if GLUC_KW.search(t) and not GLUC_SKIP.search(t) and not HBA1C_KW.search(t):
            m = GLUC_KW.search(t)
            vals = _values_after(t, m.end(), GLUC_KW)
            val, src = None, i
            for v in vals:
                if 1.5 <= v <= 800:
                    val = v
                    break
            if val is None:
                nv, ni = _next_line_values(lines, i, 1.5, 800)
                if nv is not None:
                    val, src = nv, ni
            if val is not None:
                kind = ("fasting" if re.search(r"fast|\bfbs\b|\bfbg\b|\bfpg\b", low) else
                        "random" if re.search(r"random|\brbs\b|casual", low) else
                        "post-meal" if re.search(r"post|prandial|\bppbs\b|\bppg\b", low) else "unspecified")
                note = ""
                mmol = "mmol" in low or (val < 35 and "mg" not in low)
                if mmol:
                    val = round(val * 18.016, 0)
                    note = "Converted from mmol/L to mg/dL."
                glucose_candidates.append({"value": float(round(val)), "kind": kind, "confidence": lines[src]["conf"],
                                           "line_index": src, "source_text": lines[src]["text"], "note": note})
            continue

        # ---- BMI ----------------------------------------------------------------------
        if out["bmi"] is None:
            m = BMI_KW.search(t)
            if m:
                vals = _values_after(t, m.end(), BMI_KW)
                val, src = next((v for v in vals if 10 <= v <= 80), None), i
                if val is None:
                    val, src = _next_line_values(lines, i, 10, 80)
                    src = src if src is not None else i
                if val is not None:
                    out["bmi"] = _field(round(val, 1), src, lines[src])
                    continue

        # ---- height / weight (to derive BMI when it is not printed) ------------------------
        if height_cm is None and re.search(r"\bheight\b|\bht\b", low):
            vals = _values_after(t, re.search(r"\bheight\b|\bht\b", low).end())
            for v in vals:
                if re.search(r"\bm\b", low) and not re.search(r"cm", low) and 1.0 <= v <= 2.4:
                    height_cm, height_line = v * 100, i
                    break
                if 90 <= v <= 230:
                    height_cm, height_line = v, i
                    break
        if weight_kg is None and re.search(r"\bweight\b|\bwt\b", low):
            vals = _values_after(t, re.search(r"\bweight\b|\bwt\b", low).end())
            for v in vals:
                if re.search(r"lb", low) and 40 <= v <= 700:
                    weight_kg, weight_line = v * 0.45359237, i
                    break
                if 15 <= v <= 300:
                    weight_kg, weight_line = v, i
                    break

        # ---- age / sex ------------------------------------------------------------------
        if out["age"] is None:
            m = re.search(r"\bage\b[^0-9]{0,14}(\d{1,3})\b", low) or re.search(r"\b(\d{1,3})\s*(?:years|yrs|yr|y)\b", low)
            if m and 1 <= int(m.group(1)) <= 110:
                out["age"] = _field(float(m.group(1)), i, ln)
        if out["gender"] is None:
            g = word = None
            seg = None
            km = re.search(r"\b(?:sex|gender)\b", low)
            if km:
                seg = low[km.end():]
            elif re.search(r"\b\d{1,3}\s*(?:years|yrs|yr|y)\b", low):
                seg = low
            elif re.fullmatch(r"\s*[a-z]{3,7}\s*", low):
                seg = low
            for word in re.findall(r"[a-z]+", seg or ""):
                g = _sex_word(word)
                if g:
                    break
            if g:
                out["gender"] = _field(g, i, ln, "" if word in ("female", "male", "f", "m") else "Read with a spelling tolerance; check it.")

        # ---- explicit history fields -----------------------------------------------------------
        if out["hypertension"] is None and "hypertension" in low:
            yn = _yes_no(low)
            if yn is not None:
                out["hypertension"] = _field(yn, i, ln)
        if out["heart_disease"] is None and re.search(r"heart\s*disease|cardiac", low):
            yn = _yes_no(low)
            if yn is not None:
                out["heart_disease"] = _field(yn, i, ln)
        if out["smoking_history"] is None and "smok" in low:
            s = _smoking(low)
            if s:
                out["smoking_history"] = _field(s, i, ln)

    if glucose_candidates:
        order = {"fasting": 0, "random": 1, "unspecified": 2, "post-meal": 3}
        glucose_candidates.sort(key=lambda c: (order[c["kind"]], c["line_index"]))
        best = glucose_candidates[0]
        out["glucose"] = {k: best[k] for k in ("confidence", "line_index", "source_text", "note")}
        out["glucose"]["value"] = best["value"]
        out["glucose"]["kind"] = best["kind"]

    if out["bmi"] is None and height_cm and weight_kg:
        bmi = weight_kg / ((height_cm / 100) ** 2)
        if 10 <= bmi <= 80:
            li = height_line
            out["bmi"] = {"value": round(bmi, 1), "confidence": min(lines[height_line]["conf"], lines[weight_line]["conf"]),
                          "line_index": li, "source_text": f"{lines[height_line]['text']}  |  {lines[weight_line]['text']}",
                          "note": "Calculated from the height and weight printed on the report."}
    return {"fields": out, "glucose_candidates": glucose_candidates}


def _yes_no(low: str):
    if re.search(r"\b(yes|present|positive|known|diagnosed|history of)\b", low):
        return 1
    if re.search(r"\b(no|absent|nil|negative|none|not present)\b", low):
        return 0
    return None


def _smoking(low: str):
    if re.search(r"non[\s-]*smoker|never|does not smoke|no\b", low):
        return "never"
    if re.search(r"ex[\s-]*smoker|former|quit|stopped", low):
        return "former"
    if re.search(r"current|yes|smoker", low):
        return "current"
    return None
