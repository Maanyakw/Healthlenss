FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# CPU-only torch keeps the image small
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch && grep -v '^torch' requirements.txt | grep -v '^pytest\|^httpx' > req.txt && pip install -r req.txt
COPY backend ./backend
COPY artifacts ./artifacts
COPY data ./data
COPY samples ./samples
EXPOSE 8000
# HF_TOKEN (secret) lets the server download the gated TabPFN-3.5 weights on first start; without it the server falls back to gradient boosting.
CMD ["sh","-c","uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
