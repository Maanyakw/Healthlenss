"""Doctor report (page 4): generative-AI narrative + deterministic data tables + PDF.

Design rule: every *number* in the tables comes from code (the patient's values and the risk
model); the generative model only writes prose around them. Its prose is checked for figures
that are not among the supplied facts, and any such figure is surfaced as a warning.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import re
from xml.sax.saxutils import escape

from . import bands, genai

ADA_CRITERIA = (
    "Diabetes (ADA): HbA1c 6.5% or higher; fasting plasma glucose 126 mg/dL or higher (fasting at least 8 hours); "
    "2-hour plasma glucose 200 mg/dL or higher during a 75 g OGTT; or random plasma glucose 200 mg/dL or higher with classic "
    "symptoms. Without unequivocal hyperglycaemia, confirm with two abnormal results (same or separate samples). "
    "Prediabetes: HbA1c 5.7-6.4%; fasting glucose 100-125 mg/dL; 2-hour OGTT glucose 140-199 mg/dL."
)

HB_LABEL = {"normal": "Normal (below 5.7%)", "prediabetes": "Prediabetes range (5.7-6.4%)", "diabetes": "Diabetes range (6.5% or higher)"}
def gl_label(gl: float) -> str:
    return "Low (below 70 mg/dL)" if gl < 70 else GL_LABEL[bands.glucose_band(gl)]


GL_LABEL = {"normal": "Normal (below 100 mg/dL)", "elevated": "Elevated (100-125 mg/dL)", "high": "High (126-199 mg/dL)",
            "very_high": "Very high (200 mg/dL or above)"}
BMI_LABEL = {"underweight": "Underweight (below 18.5)", "healthy": "Healthy weight (18.5-24.9)",
             "overweight": "Overweight (25-29.9)", "obese": "Obese (30 or above)"}

SECTION_KEYS = ["clinical_summary", "key_findings", "risk_drivers", "recommended_follow_up", "lifestyle_and_counselling", "limitations"]

# figures the prose may legitimately contain besides the patient's own values
_GENERIC_OK = {5.7, 6.4, 6.5, 100, 125, 126, 140, 199, 200, 18.5, 24.9, 25, 29.9, 30, 8, 5, 7, 10, 150, 3, 2, 12, 6, 1, 4}


def build_facts(patient: dict, pred: dict, provenance: dict | None, thresholds: dict) -> dict:
    p = patient
    hb, gl, bmi, age = float(p["hba1c"]), float(p["glucose"]), float(p["bmi"]), float(p["age"])
    prov = provenance or {}

    def src(k: str) -> str:
        return {"extracted": "Read from lab report (ViT)", "manual": "Entered manually"}.get(prov.get(k, "manual"), "Entered manually")

    rows = [
        {"name": "Sex", "value": p["gender"], "interpretation": "", "source": src("gender")},
        {"name": "Age", "value": f"{bands.fmt_num(age)} years", "interpretation": "", "source": src("age")},
        {"name": "HbA1c", "value": f"{hb:.1f} %", "interpretation": HB_LABEL[bands.hba1c_band(hb)], "source": src("hba1c")},
        {"name": "Blood glucose", "value": f"{bands.fmt_num(gl, 0)} mg/dL", "interpretation": gl_label(gl),
         "source": src("glucose")},
        {"name": "BMI", "value": f"{bmi:.1f} kg/m2", "interpretation": BMI_LABEL[bands.bmi_band(bmi)], "source": src("bmi")},
        {"name": "Hypertension", "value": "Yes" if int(p["hypertension"]) else "No", "interpretation": "", "source": src("hypertension")},
        {"name": "Heart disease", "value": "Yes" if int(p["heart_disease"]) else "No", "interpretation": "", "source": src("heart_disease")},
        {"name": "Smoking history", "value": bands.SMOKING_LABEL[p["smoking_history"]].capitalize(), "interpretation": "",
         "source": src("smoking_history")},
    ]
    return {
        "rows": rows,
        "probability": pred["probability"],
        "probability_text": bands.fmt_pct(pred["probability"]),
        "tier": pred["tier"],
        "thresholds": thresholds,
        "explanation": pred.get("explanation", [])[:5],
        "narrative": pred.get("narrative", ""),
        "warnings": pred.get("warnings", []),
    }


# --------------------------------------------------------------------------- generative report
def _facts_for_prompt(f: dict) -> str:
    lines = [f"- {r['name']}: {r['value']}" + (f" -> {r['interpretation']}" if r["interpretation"] else "") for r in f["rows"]]
    drv = "; ".join(f"{e['label']} {e['impact_pct_points']:+.1f} percentage points" for e in f["explanation"])
    return (
        "PATIENT PARAMETERS\n" + "\n".join(lines) +
        f"\n\nSCREENING MODEL OUTPUT\n- Estimated probability: {f['probability_text']} -> tier: {f['tier']} "
        f"(moderate from {f['thresholds']['moderate'] * 100:.1f}%, high from {f['thresholds']['high'] * 100:.0f}%)"
        f"\n- Contribution of each factor vs a typical non-diabetic profile: {drv}"
        f"\n- Plain-language narrative from the local dataset-trained language model: {f['narrative']}"
        + ("\n- Model warnings: " + " ".join(f["warnings"]) if f["warnings"] else ""))


REPORT_SYSTEM = f"""You are a clinical documentation assistant. Write a concise pre-consultation summary for a physician from the structured screening data supplied by the user.

Strict rules:
- Use ONLY the supplied facts. Never invent measurements, dates, names, medications, symptoms or history. If something was not provided, say "not provided".
- This is a screening-model output, not a diagnosis. Use cautious wording such as "suggests" or "is consistent with" and recommend confirmatory testing according to ADA criteria:
  {ADA_CRITERIA}
- Do not recommend specific drugs or doses.
- Describe glucose and HbA1c ranges as "range" (never "target"). A glucose below 70 mg/dL is low, not normal; mention it as a finding that needs separate clinical review.
- The risk model was trained on a public dataset whose labels are dominated by HbA1c and blood glucose; mention this once under limitations.
- Plain text only inside strings (no markdown, no bullet characters). Every list item is one sentence.

Return ONLY a JSON object with exactly these keys:
{{"clinical_summary": "3-5 sentences",
  "key_findings": ["..."],
  "risk_drivers": ["..."],
  "recommended_follow_up": ["..."],
  "lifestyle_and_counselling": ["..."],
  "limitations": ["..."]}}
Each list has 2-5 items."""


def _extract_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _clean_text(s: str) -> str:
    s = re.sub(r"[*_`#]+", "", str(s))
    return " ".join(s.split())[:900]


def _validate_sections(data: dict) -> dict:
    out = {}
    for k in SECTION_KEYS:
        v = data.get(k)
        if k == "clinical_summary":
            if not isinstance(v, str) or len(v.strip()) < 20:
                raise ValueError(f"bad {k}")
            out[k] = _clean_text(v)
        else:
            if not isinstance(v, list) or not v:
                raise ValueError(f"bad {k}")
            out[k] = [_clean_text(x) for x in v[:6] if str(x).strip()]
            if not out[k]:
                raise ValueError(f"empty {k}")
    return out


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", text)]


def unverified_figures(sections: dict, facts: dict) -> list[str]:
    allowed = set(_GENERIC_OK)
    for r in facts["rows"]:
        allowed.update(_numbers(r["value"]))
    allowed.update(_numbers(facts["probability_text"]))
    for e in facts["explanation"]:
        allowed.update({round(abs(e["impact_pct_points"]), 1), round(abs(e["impact_pct_points"]))})
    allowed.update(_numbers(facts["narrative"]))
    for w in facts.get("warnings", []):
        allowed.update(_numbers(w))
    allowed.update({round(facts["thresholds"]["moderate"] * 100, 1), 5.0, 50.0})
    bad = []
    blob = " ".join([sections["clinical_summary"]] + sum((sections[k] for k in SECTION_KEYS[1:]), []))
    for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%|mg/dL|mmol/L|kg/m2|kg/m²|years|year-old)", blob):
        v = float(m.group(1))
        if not any(abs(v - a) < 0.051 for a in allowed):
            bad.append(m.group(0))
    return sorted(set(bad))


# --------------------------------------------------------------------------- local fallback
def template_sections(f: dict, patient: dict) -> dict:
    rows = {r["name"]: r for r in f["rows"]}
    tier = f["tier"]
    hb, gl, bmi = float(patient["hba1c"]), float(patient["glucose"]), float(patient["bmi"])
    summary = (
        f"{patient['gender']} patient, {bands.fmt_num(float(patient['age']))} years old. The screening model estimates a {tier} probability "
        f"of diabetes ({f['probability_text']}). HbA1c is {hb:.1f}% ({HB_LABEL[bands.hba1c_band(hb)].lower()}), blood glucose is "
        f"{bands.fmt_num(gl, 0)} mg/dL ({gl_label(gl).lower()}) and BMI is {bmi:.1f} ({BMI_LABEL[bands.bmi_band(bmi)].lower()}). "
        "This is a screening estimate, not a diagnosis.")
    findings = [f"{rows[n]['name']}: {rows[n]['value']} - {rows[n]['interpretation']}" for n in ("HbA1c", "Blood glucose", "BMI")]
    comorb = [n.lower() for n in ("Hypertension", "Heart disease") if rows[n]["value"] == "Yes"]
    findings.append("Recorded comorbidities: " + (", ".join(comorb) if comorb else "none recorded") + ".")
    drivers = [f"{e['label']}: {e['impact_pct_points']:+.1f} percentage points on the estimated risk compared with a typical non-diabetic profile."
               for e in f["explanation"][:4]]
    follow = {
        "high": ["Confirm with laboratory testing per ADA criteria (repeat fasting plasma glucose and/or HbA1c on a separate sample; OGTT if results are discordant).",
                 "Assess for symptoms of hyperglycaemia and for complications at the clinician's discretion."],
        "moderate": ["Repeat fasting plasma glucose and HbA1c to clarify the result.",
                     "Review risk factors and agree a re-screening interval."],
        "low": ["Routine re-screening at an interval chosen by the clinician (ADA suggests at least every 3 years for adults).",
                "Reassess sooner if symptoms or new risk factors appear."],
    }[tier]
    life = []
    if bmi >= 25:
        life.append("Discuss gradual weight loss (about 5-7% of body weight) and about 150 minutes per week of moderate activity.")
    else:
        life.append("Encourage about 150 minutes per week of moderate activity and a balanced diet.")
    if patient["smoking_history"] == "current":
        life.append("Offer smoking-cessation support.")
    if int(patient["hypertension"]) or int(patient["heart_disease"]):
        life.append("Review blood-pressure and cardiovascular risk management in the context of possible diabetes.")
    life.append("Provide diet counselling, ideally with a registered dietitian.")
    limits = [
        "Risk estimate comes from a statistical model trained on a public dataset; its labels are dominated by HbA1c and blood glucose.",
        "The dataset does not state whether glucose was fasting or random; interpret the glucose value with that in mind.",
        "Values read from a lab report by the vision model should be verified against the original document.",
    ]
    return {"clinical_summary": summary, "key_findings": findings, "risk_drivers": drivers,
            "recommended_follow_up": follow, "lifestyle_and_counselling": life, "limitations": limits}


def generate_report(patient: dict, pred: dict, provenance: dict | None, thresholds: dict, patient_ref: str = "") -> dict:
    facts = build_facts(patient, pred, provenance, thresholds)
    meta = {"generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "patient_ref": patient_ref.strip()[:60]}
    warnings: list[str] = []
    mode, provider, model = "template", None, None
    sections = None
    if genai.available('report'):
        try:
            raw = genai.generate(REPORT_SYSTEM, [{"role": "user", "content": _facts_for_prompt(facts)}], purpose='report',
                                 json_mode=True, max_tokens=1800, temperature=0.2)
            sections = _validate_sections(_extract_json(raw))
            mode, provider, model = "genai", genai.provider('report'), genai.model_name('report')
            bad = unverified_figures(sections, facts)
            if bad:
                warnings.append("The generated text contains figures that are not among the supplied values: "
                                + ", ".join(bad) + ". Please verify before relying on them.")
        except (genai.GenAIError, ValueError, json.JSONDecodeError) as e:
            warnings.append(f"The generative-AI service did not return a usable report ({type(e).__name__}); "
                            "a template report built from the same data is shown instead.")
            sections = None
    if sections is None:
        sections = template_sections(facts, patient)
        if not genai.available('report'):
            warnings.append("No generative-AI key is configured, so this report was assembled from templates. "
                            "Add a GEMINI_API_KEY (free tier) to the .env file to enable generative writing.")
    return {"meta": meta, "facts": facts, "sections": sections, "mode": mode, "provider": provider, "model": model,
            "warnings": warnings, "ada_reference": ADA_CRITERIA}


# --------------------------------------------------------------------------- PDF
def _pdf_safe(s: str) -> str:
    s = (str(s).replace("≥", ">=").replace("≤", "<=").replace("→", "->").replace("²", "2")
         .replace("•", "-").replace("‑", "-"))
    return s.encode("cp1252", "replace").decode("cp1252")


def build_pdf(report: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13.5, alignment=TA_LEFT)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=11, textColor=colors.HexColor("#555555"))
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=17, textColor=colors.HexColor("#0b3d5c"), spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, textColor=colors.HexColor("#0b3d5c"),
                        spaceBefore=10, spaceAfter=4)
    cell = ParagraphStyle("cell", parent=body, fontSize=9, leading=12)

    def P(t, st=body):
        return Paragraph(escape(_pdf_safe(t)), st)

    def bullets(items):
        return ListFlowable([ListItem(P(i), leftIndent=10) for i in items], bulletType="bullet", start="-", leftIndent=12, bulletFontSize=9)

    f, s, meta = report["facts"], report["sections"], report["meta"]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title="Diabetes risk screening report", author="Diabetes GenAI screening app")
    el = [P("Diabetes Risk Screening Report", h1), P("Pre-consultation summary for clinician review", small), Spacer(1, 4)]
    meta_rows = [["Generated", meta["generated_at"]], ["Patient reference", meta.get("patient_ref") or "not provided"],
                 ["Report writer", {"genai": f"Generative AI ({report.get('provider')}, {report.get('model')})",
                                    "template": "Template (no generative-AI service used)"}[report["mode"]]]]
    t = Table([[P(a, cell), P(b, cell)] for a, b in meta_rows], colWidths=[40 * mm, 130 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d6df")), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef4f8"))]))
    el += [t, P("Clinical summary", h2), P(s["clinical_summary"])]

    el.append(P("Patient parameters", h2))
    rows = [[P("Parameter", cell), P("Value", cell), P("Interpretation", cell), P("Source", cell)]]
    rows += [[P(r["name"], cell), P(r["value"], cell), P(r["interpretation"] or "-", cell), P(r["source"], cell)] for r in f["rows"]]
    t = Table(rows, colWidths=[32 * mm, 30 * mm, 66 * mm, 42 * mm], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d6df")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe8f0")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    el.append(t)

    el.append(P("Screening-model result", h2))
    tier_col = {"low": "#1f8a4c", "moderate": "#b7791f", "high": "#c0392b"}[f["tier"]]
    el.append(Paragraph(f'Estimated probability: <b>{escape(f["probability_text"])}</b> &nbsp; Tier: '
                        f'<font color="{tier_col}"><b>{escape(f["tier"].upper())}</b></font> '
                        f'(moderate from {f["thresholds"]["moderate"] * 100:.1f}%, high from {f["thresholds"]["high"] * 100:.0f}%)', body))
    el.append(Spacer(1, 3))
    rows = [[P("Factor", cell), P("Value", cell), P("Effect on estimated risk*", cell)]]
    for e in f["explanation"]:
        v = e["value"]
        if isinstance(v, (int, float)) and e["feature"] in ("hypertension", "heart_disease"):
            v = "Yes" if int(v) else "No"
        rows.append([P(e["label"], cell), P(str(v), cell), P(f"{e['impact_pct_points']:+.1f} percentage points", cell)])
    t = Table(rows, colWidths=[50 * mm, 45 * mm, 75 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d6df")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe8f0"))]))
    el += [t, P("*Change in the estimate compared with a typical non-diabetic patient of the training data (marginal substitution).", small)]
    el += [P("Narrative from the locally trained language model", h2), P(f["narrative"])]

    for title, key in [("Key findings", "key_findings"), ("Risk drivers", "risk_drivers"), ("Suggested follow-up", "recommended_follow_up"),
                       ("Lifestyle and counselling points", "lifestyle_and_counselling"), ("Limitations", "limitations")]:
        el += [P(title, h2), bullets(s[key])]
    if report.get("warnings"):
        el += [P("Notes on this report", h2), bullets(report["warnings"])]
    el += [P("Reference criteria", h2), P(report["ada_reference"], small), Spacer(1, 6),
           P("This report is generated by a screening tool for educational use. It is not a diagnosis and does not replace clinical judgement. "
             "Values read from documents by the vision model must be checked against the source.", small)]
    doc.build(el)
    return buf.getvalue()
