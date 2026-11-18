"""Clinical banding helpers shared by the classifier, the narrator corpus and the report.

Cut-offs follow widely used screening conventions (ADA / WHO). They are used only to
*describe* a value in words; the risk estimate itself comes from the trained classifier.
"""
from __future__ import annotations

GENDERS = ["Female", "Male"]
SMOKING = ["No Info", "never", "former", "current", "not current", "ever"]

SMOKING_LABEL = {
    "No Info": "not recorded",
    "never": "never smoked",
    "former": "former smoker",
    "current": "current smoker",
    "not current": "not a current smoker",
    "ever": "has smoked in the past",
}


def age_band(age: float) -> str:
    if age < 18:
        return "young"
    if age < 40:
        return "adult"
    if age < 60:
        return "midlife"
    return "older"


def hba1c_band(x: float) -> str:
    if x < 5.7:
        return "normal"
    if x < 6.5:
        return "prediabetes"
    return "diabetes"


def glucose_band(x: float) -> str:
    if x < 100:
        return "normal"
    if x < 126:
        return "elevated"
    if x < 200:
        return "high"
    return "very_high"


def bmi_band(x: float) -> str:
    if x < 18.5:
        return "underweight"
    if x < 25:
        return "healthy"
    if x < 30:
        return "overweight"
    return "obese"


def tier_from_prob(p: float, thresholds: dict) -> str:
    if p >= thresholds["high"]:
        return "high"
    if p >= thresholds["moderate"]:
        return "moderate"
    return "low"


def fmt_num(x: float, nd: int = 1) -> str:
    """Whole numbers print without decimals (54.0 -> '54'), others with `nd` decimals."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.{nd}f}"


def fmt_pct(p: float, nd: int = 1) -> str:
    """p is a fraction in [0, 1]."""
    return f"{p * 100:.{nd}f}%"
