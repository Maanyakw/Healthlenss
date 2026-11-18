"""API tests. Run from the project root:  python3 -m pytest -q tests"""
import io
import os
os.environ.setdefault("RISK_BACKEND", "hgb")  # tests use the local stand-in; TabPFN weights are gated
import pytest
from fastapi.testclient import TestClient
from backend.app import app

P_F = dict(gender="Female", age=54, hypertension=1, heart_disease=0, smoking_history="never", bmi=31.2, hba1c=6.8, glucose=190)
P_M = dict(gender="Male", age=29, hypertension=0, heart_disease=0, smoking_history="current", bmi=22.4, hba1c=5.1, glucose=88)


@pytest.fixture(scope="module")
def c():
    with TestClient(app) as client:
        yield client


def test_root_health_cors(c):
    assert c.get("/").json()["service"].startswith("HealthLens")
    h = c.get("/api/health").json()
    assert h["status"] == "ok" and h["risk_backend"] in ("hgb", "tabpfn") and h["narrator_loaded"] and h["vit_loaded"]
    r = c.options("/api/predict", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST",
                                           "Access-Control-Request-Headers": "content-type"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "access-control-allow-origin" not in c.options("/api/predict", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}).headers


def test_model_info_comparison(c):
    info = c.get("/api/model-info").json()
    names = [r["name"] for r in info["comparison"]]
    assert names[0] == "TabPFN-3.5" and "XGBoost" in names and "Logistic regression" in names
    assert info["primary_model"] == "TabPFN-3.5"


def test_counterfactuals(c):
    r = c.post("/api/counterfactuals", json=P_F).json()
    assert r["tier"] == "high" and r["options"] and r["options"][0]["new_probability"] < 0.5
    assert c.post("/api/counterfactuals", json=P_M).json()["options"] == []


def test_predict_both_sexes(c):
    hi = c.post("/api/predict", json=P_F).json()
    lo = c.post("/api/predict", json=P_M).json()
    assert hi["tier"] == "high" and lo["tier"] == "low" and hi["probability"] > lo["probability"]
    assert hi["narrative"]["text"] and "not a diagnosis" in hi["narrative"]["text"]
    assert abs(sum(e["impact_pct_points"] for e in hi["explanation"])) > 50


@pytest.mark.parametrize("bad", [dict(P_F, age=0), dict(P_F, hba1c=25), dict(P_F, gender="Other"), dict(P_F, bmi="x"), {}])
def test_predict_rejects_bad_input(c, bad):
    assert c.post("/api/predict", json=bad).status_code in (400, 422)


def test_chat_local_whatif_and_safety(c):
    base = dict(patient=P_F, narrative="", history=[])
    r = c.post("/api/chat", json=dict(base, message="What does my HbA1c mean?")).json()
    assert r["reply"]
    w = c.post("/api/chat", json=dict(base, message="What if my BMI were 24?")).json()
    assert w.get("whatif") and "%" in w["reply"]
    s = c.post("/api/chat", json=dict(base, message="I feel confused and I am vomiting, very thirsty")).json()
    assert s["mode"] == "safety"
    assert c.post("/api/chat", json=dict(base, message="x" * 5000)).status_code in (400, 413, 422)


def test_report_and_pdf(c):
    r = c.post("/api/report", json=dict(patient=P_F, provenance={}, patient_ref="AB-12", narrative="")).json()
    assert r["sections"]["clinical_summary"] and r["facts"]["tier"] == "high"
    pdf = c.post("/api/report/pdf", json=dict(report=r))
    assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-"


def test_extract_validation(c):
    assert c.post("/api/extract", files={"file": ("a.txt", b"hello", "text/plain")}).status_code in (400, 415, 422)
    assert c.post("/api/extract", files={"file": ("a.png", b"notanimage", "image/png")}).status_code in (400, 415, 422)
    big = io.BytesIO(b"\x89PNG" + b"0" * (16 * 1024 * 1024))
    assert c.post("/api/extract", files={"file": ("a.png", big, "image/png")}).status_code in (400, 413)


def test_whatif_endpoint(c):
    base = dict(gender="Male", age=58, hypertension=1, heart_disease=0, smoking_history="former", bmi=32.1, hba1c=7.1, glucose=186)
    hi = c.post("/api/whatif", json=base).json()
    lo = c.post("/api/whatif", json={**base, "hba1c": 5.2, "glucose": 95}).json()
    assert set(hi) >= {"probability", "tier", "thresholds", "warnings"}
    assert lo["probability"] < hi["probability"]


def test_low_glucose_report_labels_and_no_false_alarm(c):
    p = dict(P_F, glucose=55, hba1c=5.8, bmi=27.7, age=23)
    pred = c.post("/api/predict", json=p).json()
    assert any("low (below 70)" in w for w in pred["warnings"])
    rep = c.post("/api/report", json={"patient": p, "provenance": {}, "patient_ref": "", "narrative": pred["narrative"]["text"]}).json()
    row = [r for r in rep["facts"]["rows"] if r["name"] == "Blood glucose"][0]
    assert row["interpretation"].startswith("Low")
