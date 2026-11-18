"""FastAPI application: JSON API under /api. The 4-page Next.js frontend (frontend/) is deployed separately.

Run:  uvicorn backend.app:app --host 127.0.0.1 --port 8000
Privacy: request bodies (patient values, uploaded reports) are never logged or stored.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import bands, chat, config, genai, report
from .risk_model import RiskModel
from .vit_ocr.parse import parse_fields

SMOKING_T = Literal["No Info", "never", "former", "current", "not current", "ever"]


class Patient(BaseModel):
    gender: Literal["Female", "Male"]
    age: float = Field(ge=1, le=110)
    hypertension: int = Field(ge=0, le=1)
    heart_disease: int = Field(ge=0, le=1)
    smoking_history: SMOKING_T
    bmi: float = Field(ge=10, le=80)
    hba1c: float = Field(ge=3, le=18)
    glucose: float = Field(ge=30, le=700)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=config.MAX_CHAT_CHARS)
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)
    patient: Patient
    narrative: str = Field(default="", max_length=3000)


class ReportRequest(BaseModel):
    patient: Patient
    narrative: str = Field(default="", max_length=3000)
    provenance: dict[str, str] = Field(default_factory=dict)
    patient_ref: str = Field(default="", max_length=60)


class PdfRequest(BaseModel):
    report: dict


# --------------------------------------------------------------------------- state
class State:
    stats: dict
    risk: RiskModel
    narrator = None
    reader = None
    errors: dict[str, str]


S = State()
S.errors = {}
_reader_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    S.stats = json.loads(config.STATS_PATH.read_text())
    S.risk = RiskModel()
    try:
        from .narrator.narrate import Narrator
        S.narrator = Narrator(S.stats)
    except Exception as e:  # missing / incompatible artifacts
        S.errors["narrator"] = f"{type(e).__name__}: {e}"
    try:
        from .vit_ocr.extract import LabReportReader
        S.reader = LabReportReader()
    except Exception as e:
        S.errors["vit"] = f"{type(e).__name__}: {e}"
    yield


app = FastAPI(title="HealthLens v3 API", lifespan=lifespan, docs_url="/api/docs", redoc_url=None,
              openapi_url="/api/openapi.json")


_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
                   allow_credentials=False, max_age=600)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Frame-Options"] = "DENY"
    if request.url.path.startswith("/api/") and not request.url.path.startswith("/api/docs") and "openapi" not in request.url.path:
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["Content-Security-Policy"] = "default-src 'none'"
    return resp


# --------------------------------------------------------------------------- tiny rate limiter
_hits: dict[tuple[str, str], deque] = defaultdict(deque)
_LIMITS = {"extract": (20, 60), "chat": (40, 60), "report": (12, 60), "predict": (120, 60), "whatif": (240, 60)}


def limit(request: Request, bucket: str) -> None:
    n, window = _LIMITS[bucket]
    key = (request.client.host if request.client else "?", bucket)
    now = time.time()
    q = _hits[key]
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= n:
        raise HTTPException(429, "Too many requests - please wait a moment and try again.")
    q.append(now)


# --------------------------------------------------------------------------- helpers
def _predict(p: Patient, with_narrative: bool = True, narrative_text: str = "") -> dict:
    d = p.model_dump()
    prob = S.risk.prob(d)
    tier = bands.tier_from_prob(prob, S.risk.thresholds)
    out = {
        "inputs": d,
        "probability": prob,
        "tier": tier,
        "thresholds": S.risk.thresholds,
        "explanation": S.risk.explain(d),
        "warnings": S.risk.warnings(d),
    }
    if with_narrative:
        if S.narrator is None:
            raise HTTPException(503, "The narrator language model is not available. Train it with: python -m backend.narrator.train_lm")
        out["narrative"] = S.narrator.narrate(d, prob, tier)
    else:
        out["narrative"] = narrative_text
    return out


# --------------------------------------------------------------------------- API
@app.get("/api/health")
def health():
    return {"status": "ok", "risk_backend": S.risk.backend, "chat_provider": genai.provider("chat"), "report_provider": genai.provider("report"), "narrator_loaded": S.narrator is not None,
            "vit_loaded": S.reader is not None, "errors": S.errors}


def _comparison() -> list[dict]:
    """Benchmark table: local rows from stats.json; the TabPFN row from model_card.json once the notebook was run."""
    rows = [dict(r) for r in S.stats.get("comparison", [])]
    if config.MODEL_CARD_PATH.exists():
        card = json.loads(config.MODEL_CARD_PATH.read_text())
        if card.get("tabpfn"):
            rows = [dict(card["tabpfn"], name="TabPFN-3.5", role="primary") if r["name"] == "TabPFN-3.5" else r for r in rows]
    return rows


@app.get("/api/model-info")
def model_info():
    st = S.stats
    out = {
        "dataset": st["dataset"],
        "thresholds": st["thresholds"],
        "classifier": st["metrics"],
        "primary_model": st.get("primary_model"),
        "comparison": _comparison(),
        "risk_backend": S.risk.backend,
        "risk_backend_note": S.risk.backend_note,
        "genai": {"chat": {"provider": genai.provider("chat"), "model": genai.model_name("chat")}, "report": {"provider": genai.provider("report"), "model": genai.model_name("report")}},
        "narrator": None, "vit": None,
    }
    for key, fname in (("narrator", "lm_metrics.json"), ("vit", "vit_metrics.json")):
        p = config.ART_DIR / fname
        if p.exists():
            m = json.loads(p.read_text())
            m.pop("history", None)
            out[key] = m
    return out


@app.post("/api/extract")
def extract(request: Request, file: UploadFile = File(...)):
    limit(request, "extract")
    data = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not (data[:5] == b"%PDF-" or data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff"):
        raise HTTPException(400, "Unsupported file. Upload a PDF, PNG or JPEG lab report.")
    if S.reader is None:
        raise HTTPException(503, "The Vision Transformer model is not available. Train it with: python -m backend.vit_ocr.train_vit")
    from .vit_ocr.extract import UnsupportedFile
    t0 = time.time()
    try:
        res = S.reader.read(data)
    except UnsupportedFile as e:
        raise HTTPException(400, str(e))
    except Exception:
        raise HTTPException(500, "The report could not be processed. Try a clearer scan or enter the values manually.")
    parsed = parse_fields(res["lines"])
    return {**res, **parsed, "elapsed_ms": int((time.time() - t0) * 1000)}


@app.post("/api/predict")
def predict(request: Request, patient: Patient):
    limit(request, "predict")
    return _predict(patient)


@app.post("/api/whatif")
def whatif(request: Request, patient: Patient):
    """Fast probability + tier for slider what-ifs (no narrative, no Shapley)."""
    limit(request, "whatif")
    d = patient.model_dump()
    prob = S.risk.prob(d)
    return {"probability": prob, "tier": bands.tier_from_prob(prob, S.risk.thresholds),
            "thresholds": S.risk.thresholds, "warnings": S.risk.warnings(d)}


@app.post("/api/counterfactuals")
def counterfactuals(request: Request, patient: Patient):
    limit(request, "predict")
    return S.risk.counterfactuals(patient.model_dump())


@app.post("/api/chat")
def chat_endpoint(request: Request, req: ChatRequest):
    limit(request, "chat")
    pred = _predict(req.patient, with_narrative=False, narrative_text=req.narrative)
    res = chat.answer(req.message, [t.model_dump() for t in req.history], req.patient.model_dump(), pred, S.risk, S.stats)
    return res


@app.post("/api/report")
def report_endpoint(request: Request, req: ReportRequest):
    limit(request, "report")
    pred = _predict(req.patient, with_narrative=False, narrative_text=req.narrative)
    return report.generate_report(req.patient.model_dump(), pred, req.provenance, S.risk.thresholds, req.patient_ref)


@app.post("/api/report/pdf")
def report_pdf(request: Request, req: PdfRequest):
    limit(request, "report")
    try:
        pdf = report.build_pdf(req.report)
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "The report data is incomplete; generate the report again.")
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="diabetes-risk-report.pdf"'})


@app.exception_handler(HTTPException)
async def http_exc(request: Request, exc: HTTPException):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


@app.get("/")
def root():
    return {"service": "HealthLens v3 API", "docs": "/api/docs", "health": "/api/health"}
