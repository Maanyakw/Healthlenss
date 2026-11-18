"""Narrative corpus for the locally trained language model.

The dataset has no free text, so the training text is *derived from the dataset rows only*:
every row -> (control tokens describing the row, a plain-English narrative of that row).
Numbers that depend on the patient (age, HbA1c, glucose, BMI, probability) and the dataset
statistics (diabetes rate per band, computed from the training split) are written as
placeholders such as <hba1c> or <prev_hba1c>; they are filled in *after* generation from
the real values, so the language model can never invent a number.

The same module also provides `check()`, which verifies that a generated narrative is
consistent with the patient (right bands, right tier, all placeholders present, no stray
numbers) and `build_narrative()`, a deterministic fallback.
"""
from __future__ import annotations

import random
import re

from .. import bands

SPECIAL = ["<pad>", "<bos>", "<sep>", "<eos>"]

# ------------------------------------------------------------------ tokenisation
_TOK = re.compile(r"(\s*)(<[a-z0-9_]+>|\d+(?:\.\d+)?|[A-Za-z]+(?:'[A-Za-z]+)?|[^\sA-Za-z0-9])")


def tokenize(text: str) -> list[str]:
    return [("▁" if ws else "") + tok for ws, tok in _TOK.findall(text)]


def detokenize(tokens: list[str]) -> str:
    return "".join(tokens).replace("▁", " ").strip()


# ------------------------------------------------------------------ control fields
SMOKE_KEY = {"No Info": "noinfo", "never": "never", "former": "former", "current": "current",
             "not current": "notcurrent", "ever": "ever"}


def fields_from_values(gender: str, age: float, hba1c: float, glucose: float, bmi: float,
                       hypertension: int, heart_disease: int, smoking: str, tier: str) -> dict:
    return {
        "gender": gender.lower(),
        "age": bands.age_band(age),
        "hba1c": bands.hba1c_band(hba1c),
        "glucose": bands.glucose_band(glucose),
        "bmi": bands.bmi_band(bmi),
        "htn": "yes" if int(hypertension) else "no",
        "hd": "yes" if int(heart_disease) else "no",
        "smoke": SMOKE_KEY[smoking],
        "tier": tier,
    }


CONTROL_KEYS = ["gender", "age", "hba1c", "glucose", "bmi", "htn", "hd", "smoke", "tier"]


def control_tokens(f: dict) -> list[str]:
    return [f"{k}={f[k]}" for k in CONTROL_KEYS]


# ------------------------------------------------------------------ templates
AGE_PHRASE = {"young": "under 18", "adult": "18 to 39", "midlife": "40 to 59", "older": "60 and over"}
HBA1C_WORD = {"normal": "normal", "prediabetes": "prediabetes", "diabetes": "diabetes"}
HBA1C_RANGE = {"normal": "below 5.7%", "prediabetes": "5.7% to 6.4%", "diabetes": "6.5% or higher"}
GLUC_WORD = {"normal": "normal", "elevated": "elevated", "high": "high", "very_high": "very high"}
GLUC_RANGE = {"normal": "below 100 mg/dL", "elevated": "100 to 125 mg/dL",
              "high": "126 to 199 mg/dL", "very_high": "200 mg/dL or above"}
BMI_WORD = {"underweight": "underweight", "healthy": "healthy weight", "overweight": "overweight", "obese": "obese"}

S1 = [
    "This {sex} patient, aged <age>, has a {tier} estimated risk of diabetes (<prob>).",
    "For this {sex} patient aged <age>, the model estimates a {tier} diabetes risk of <prob>.",
    "The estimated diabetes risk for this {sex} patient, aged <age>, is {tier} at <prob>.",
]
S2 = [
    "The HbA1c of <hba1c>% is in the {w} range ({r}), and <prev_hba1c> of patients in this range had diabetes in the training data.",
    "An HbA1c of <hba1c>% falls in the {w} range ({r}); <prev_hba1c> of patients in this range had diabetes in the training data.",
]
S3 = [
    "The blood glucose of <glucose> mg/dL is {w} ({r}), and <prev_glucose> of patients in this range had diabetes in the training data.",
    "A blood glucose reading of <glucose> mg/dL is {w} ({r}); <prev_glucose> of patients in this range had diabetes in the training data.",
]
S4 = [
    "The BMI of <bmi> is in the {w} range, and <prev_bmi> of patients in this range had diabetes in the training data.",
    "A BMI of <bmi> falls in the {w} range; <prev_bmi> of patients in this range had diabetes in the training data.",
]
S5 = [
    "In the training data, the diabetes rate was <prev_age> for patients aged {ap} and <prev_gender> for {sex} patients, compared with <prev_all> overall.",
    "Among training patients aged {ap} the diabetes rate was <prev_age>, and among {sex} patients it was <prev_gender> (overall: <prev_all>).",
]
S6 = {
    ("yes", "yes"): [
        "The patient has both hypertension and heart disease; in the training data <prev_htn> of patients with hypertension and <prev_hd> of patients with heart disease had diabetes.",
        "Both hypertension and heart disease are recorded; the training-data diabetes rate was <prev_htn> with hypertension and <prev_hd> with heart disease.",
    ],
    ("yes", "no"): [
        "The patient has hypertension; in the training data <prev_htn> of patients with hypertension had diabetes.",
        "Hypertension is recorded, and <prev_htn> of patients with hypertension had diabetes in the training data.",
    ],
    ("no", "yes"): [
        "The patient has heart disease; in the training data <prev_hd> of patients with heart disease had diabetes.",
        "Heart disease is recorded, and <prev_hd> of patients with heart disease had diabetes in the training data.",
    ],
    ("no", "no"): [
        "The patient has no recorded hypertension or heart disease.",
        "No hypertension or heart disease is recorded for this patient.",
    ],
}
S7 = {
    "never": ["The patient has never smoked."],
    "former": ["The patient is a former smoker."],
    "current": ["The patient is a current smoker."],
    "notcurrent": ["The patient is not a current smoker."],
    "ever": ["The patient has smoked at some point."],
    "noinfo": ["Smoking history is not recorded."],
}
S8 = {
    "low": [
        "Overall, the profile is reassuring; healthy habits and routine repeat screening are advisable.",
        "Overall, the picture is reassuring, and keeping up healthy habits with routine screening is advisable.",
        "Overall, this is a reassuring profile; continue healthy habits and screen again at routine intervals.",
    ],
    "moderate": [
        "Overall, the profile warrants attention; a repeat fasting glucose or HbA1c test and a review of diet and activity are advisable.",
        "Overall, this warrants attention, and repeat testing together with a review of diet and activity is advisable.",
        "Overall, the picture warrants attention; repeat laboratory testing and lifestyle review are advisable.",
    ],
    "high": [
        "Overall, the profile suggests a high likelihood of diabetes; confirmatory laboratory testing and a clinical review by a doctor are strongly recommended.",
        "Overall, there is a high likelihood of diabetes, and confirmatory testing with a doctor's review is strongly recommended.",
        "Overall, the picture indicates a high likelihood of diabetes; please arrange confirmatory laboratory tests and a doctor's review.",
    ],
}
S9 = "This is a screening estimate from a statistical model and not a diagnosis."

# ------------------------------------------------------------------ markers used by check()
_RE_HBA1C = re.compile(r"\b(normal|prediabetes|diabetes) range")
_RE_GLUC = re.compile(r"mg/dL is (normal|elevated|high|very high)\b")
_RE_BMI = re.compile(r"\b(underweight|healthy weight|overweight|obese) range")
_RE_NUM = re.compile(r"(?<![<\w])\d+(?:\.\d+)?")
ALLOWED_NUMBERS = {"5.7", "6.4", "6.5", "100", "125", "126", "199", "200", "18", "39", "40", "59", "60"}
TIER_WORDS = ("low", "moderate", "high")
CLOSING_MARK = {"low": "reassuring", "moderate": "warrants attention", "high": "high likelihood"}
COMORBID_MARK = {
    ("yes", "yes"): ["both hypertension and heart disease"],
    ("yes", "no"): ["has hypertension", "hypertension is recorded"],
    ("no", "yes"): ["has heart disease", "heart disease is recorded"],
    ("no", "no"): ["no recorded hypertension", "no hypertension or heart disease"],
}
SMOKE_MARK = {"never": "never smoked", "former": "former smoker", "current": "a current smoker",
              "notcurrent": "not a current smoker", "ever": "smoked at some point", "noinfo": "not recorded"}


def _assert_templates() -> None:
    """Fail loudly at import if a template cannot satisfy its own validator."""
    for b in HBA1C_WORD:
        for t in S2:
            s = t.format(w=HBA1C_WORD[b], r=HBA1C_RANGE[b])
            assert _RE_HBA1C.search(s).group(1) == HBA1C_WORD[b], s
    for b in GLUC_WORD:
        for t in S3:
            s = t.format(w=GLUC_WORD[b], r=GLUC_RANGE[b])
            assert _RE_GLUC.search(s).group(1) == GLUC_WORD[b], s
    for b in BMI_WORD:
        for t in S4:
            s = t.format(w=BMI_WORD[b])
            assert _RE_BMI.search(s).group(1) == BMI_WORD[b], s
    for k, v in S6.items():
        for t in v:
            assert any(m in t.lower() for m in COMORBID_MARK[k]), t
    for k, v in S7.items():
        for t in v:
            assert SMOKE_MARK[k] in t, t
    for tier, v in S8.items():
        for t in v:
            assert CLOSING_MARK[tier] in t, t


_assert_templates()


# ------------------------------------------------------------------ building
def build_narrative(f: dict, rng: random.Random) -> str:
    """Narrative with placeholders for one row's control fields."""
    out = [
        rng.choice(S1).format(sex=f["gender"], tier=f["tier"]),
        rng.choice(S2).format(w=HBA1C_WORD[f["hba1c"]], r=HBA1C_RANGE[f["hba1c"]]),
        rng.choice(S3).format(w=GLUC_WORD[f["glucose"]], r=GLUC_RANGE[f["glucose"]]),
        rng.choice(S4).format(w=BMI_WORD[f["bmi"]]),
        rng.choice(S5).format(ap=AGE_PHRASE[f["age"]], sex=f["gender"]),
        rng.choice(S6[(f["htn"], f["hd"])]),
        rng.choice(S7[f["smoke"]]),
        rng.choice(S8[f["tier"]]),
        S9,
    ]
    return " ".join(out)


def required_placeholders(f: dict) -> set[str]:
    req = {"age", "hba1c", "glucose", "bmi", "prob", "prev_hba1c", "prev_glucose", "prev_bmi",
           "prev_age", "prev_gender", "prev_all"}
    if f["htn"] == "yes":
        req.add("prev_htn")
    if f["hd"] == "yes":
        req.add("prev_hd")
    return req


def check(f: dict, text: str) -> list[str]:
    """Return a list of problems (empty list == narrative is consistent with the patient)."""
    problems: list[str] = []
    t = text.strip()
    low = t.lower()
    # placeholders
    found = set(re.findall(r"<([a-z0-9_]+)>", t))
    req = required_placeholders(f)
    allowed = req | {"prev_htn", "prev_hd"}
    if not req <= found:
        problems.append(f"missing placeholders: {sorted(req - found)}")
    if not found <= allowed:
        problems.append(f"unexpected placeholders: {sorted(found - allowed)}")
    if f["htn"] == "no" and "prev_htn" in found:
        problems.append("prev_htn present although no hypertension")
    if f["hd"] == "no" and "prev_hd" in found:
        problems.append("prev_hd present although no heart disease")
    # bands
    m = _RE_HBA1C.findall(t)
    if m != [HBA1C_WORD[f["hba1c"]]]:
        problems.append(f"hba1c band wrong: {m}")
    m = _RE_GLUC.findall(t)
    if m != [GLUC_WORD[f["glucose"]]]:
        problems.append(f"glucose band wrong: {m}")
    m = _RE_BMI.findall(t)
    if m != [BMI_WORD[f["bmi"]]]:
        problems.append(f"bmi band wrong: {m}")
    for k, phrase in AGE_PHRASE.items():
        present = phrase in t
        if (k == f["age"]) != present:
            problems.append(f"age phrase mismatch for {k}")
    # sex / tier
    has_female = bool(re.search(r"\bfemale\b", low))
    has_male = bool(re.search(r"\bmale\b", low))  # \b stops "female" from matching
    if f["gender"] == "female" and (not has_female or has_male):
        problems.append("sex word wrong")
    if f["gender"] == "male" and (not has_male or has_female):
        problems.append("sex word wrong")
    first = re.split(r"(?<=[.!?])\s+", t)[0].lower()
    tiers_in_first = [w for w in TIER_WORDS if re.search(rf"\b{w}\b", first)]
    if tiers_in_first != [f["tier"]]:
        problems.append(f"tier wrong in first sentence: {tiers_in_first}")
    if CLOSING_MARK[f["tier"]] not in low:
        problems.append("closing sentence does not match tier")
    for other, mark in CLOSING_MARK.items():
        if other != f["tier"] and mark in low:
            problems.append(f"closing sentence for wrong tier ({other})")
    # comorbidities / smoking
    if not any(mk in low for mk in COMORBID_MARK[(f["htn"], f["hd"])]):
        problems.append("comorbidity sentence mismatch")
    if f["htn"] == "no" and "has hypertension" in low:
        problems.append("claims hypertension")
    if f["hd"] == "no" and "has heart disease" in low:
        problems.append("claims heart disease")
    if SMOKE_MARK[f["smoke"]] not in low:
        problems.append("smoking sentence mismatch")
    if f["smoke"] == "current" and "not a current smoker" in low:
        problems.append("smoking contradiction")
    # disclaimer and stray numbers
    if "not a diagnosis" not in low:
        problems.append("disclaimer missing")
    stray = [n for n in _RE_NUM.findall(re.sub(r"<[a-z0-9_]+>", " ", t)) if n not in ALLOWED_NUMBERS]
    if stray:
        problems.append(f"stray numbers: {stray}")
    if len(t) < 200 or len(t) > 2000:
        problems.append("length out of range")
    return problems


def _self_test() -> None:
    rng = random.Random(0)
    for _ in range(300):
        f = {
            "gender": rng.choice(["female", "male"]),
            "age": rng.choice(list(AGE_PHRASE)),
            "hba1c": rng.choice(list(HBA1C_WORD)),
            "glucose": rng.choice(list(GLUC_WORD)),
            "bmi": rng.choice(list(BMI_WORD)),
            "htn": rng.choice(["yes", "no"]),
            "hd": rng.choice(["yes", "no"]),
            "smoke": rng.choice(list(SMOKE_KEY.values())),
            "tier": rng.choice(list(TIER_WORDS)),
        }
        p = check(f, build_narrative(f, rng))
        assert not p, (f, p)


_self_test()
