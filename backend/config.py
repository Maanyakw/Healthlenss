"""Central configuration: paths, environment variables and limits."""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
FRONTEND_DIR = ROOT / "frontend"

DATASET_CSV = DATA_DIR / "diabetes_prediction_dataset.csv"
RISK_MODEL_PATH = ART_DIR / "risk_model.joblib"
STATS_PATH = ART_DIR / "stats.json"
LM_WEIGHTS_PATH = ART_DIR / "narrator_lm.pt"
LM_VOCAB_PATH = ART_DIR / "narrator_vocab.json"
VIT_WEIGHTS_PATH = ART_DIR / "vit_ocr.pt"


def _load_dotenv() -> None:
    """Tiny .env loader (avoids an extra dependency). Real env vars win."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = re.split(r"\s+#", value, maxsplit=1)[0].strip().strip('"').strip("'")  # drop trailing "# comment"
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# ---- Generative-AI providers ------------------------------------------------------------
# Chat assistant (page 3) prefers Groq (free tier); doctor report (page 4) prefers Gemini (free tier).
# CHAT_PROVIDER / REPORT_PROVIDER: auto | groq | gemini | anthropic | none
CHAT_PROVIDER = os.environ.get("CHAT_PROVIDER", "auto").lower()
REPORT_PROVIDER = os.environ.get("REPORT_PROVIDER", "auto").lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
GENAI_TIMEOUT_S = float(os.environ.get("GENAI_TIMEOUT_S", "60"))

# ---- Risk model backend ---------------------------------------------------------------------
# RISK_BACKEND: auto (TabPFN if its context file and package exist) | tabpfn | hgb
RISK_BACKEND = os.environ.get("RISK_BACKEND", "auto").lower()
TABPFN_CONTEXT_CSV = ART_DIR / "tabpfn_context.csv"
MODEL_CARD_PATH = ART_DIR / "model_card.json"
TABPFN_CONTEXT_ROWS = int(os.environ.get("TABPFN_CONTEXT_ROWS", "5000"))

# ---- Upload limits --------------------------------------------------------------------
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_PDF_PAGES = 4
MAX_IMAGE_PIXELS = 60_000_000

# ---- Chat limits ----------------------------------------------------------------------
MAX_CHAT_CHARS = 1000
MAX_CHAT_TURNS = 12
