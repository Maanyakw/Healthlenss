# HealthLens v3: explainable diabetes-risk screening (academic project)

Landing page plus four app pages: **/assess Read lab report, /result Risk result, /chat Ask HealthLens, /report Doctor report (PDF)**.
Extras: 3 themes (Clinic, Midnight, Meadow), live what-if sliders, an action-plan checklist for high values, and a saved-checks history (stored only in your browser).
Frontend: Next.js (Vercel). Backend: FastAPI (Azure Container Apps).

| Part | What is used |
|---|---|
| Data | Kaggle "Diabetes prediction dataset": 100,000 rows, 96,128 female and male patients after removing duplicates and 18 "Other" rows |
| Risk prediction | **TabPFN-3.5** (pretrained transformer, 5,000-row context from the training split); XGBoost and logistic regression as comparisons |
| Lab-report reading | Vision Transformer (CTC) trained here on synthetic report lines, then a rule-based parser; you confirm every value |
| Narration | 660k-parameter GPT trained only on text generated from the dataset, with a validator and template fallback |
| Chat | Groq (`openai/gpt-oss-120b`) for general questions; what-if is computed by the risk model and given to the assistant |
| Doctor report | Gemini (`gemini-2.5-flash`) or Claude (optional key), template fallback; PDF download |
| Explanation | Exact Shapley values (all 2^8 coalitions) and a counterfactual search |

## Honest status (read before the demo)
- **TabPFN has not been run in the build environment** (gated weights). Until you run the Colab notebook and add `artifacts/model_card.json`, the table shows TabPFN as "not benchmarked yet" and the server uses a gradient-boosting stand-in (`risk_backend` in `/api/health`). Nothing is claimed for TabPFN that was not measured.
- Local results (test split, 14.4k rows): XGBoost AUC 0.978, logistic regression 0.959. They are high because this dataset's label is almost fully determined by HbA1c and glucose; they do not mean real-world prediction is this easy.
- The ViT reader was trained and tested only on **synthetic** reports (about 92% of 60 unseen synthetic reports fully extracted). Real scans are not validated. Always check the values.
- Groq, Gemini and Claude calls were tested with mocked HTTP only, not live keys.
- Not a diagnosis. No real patient data should be uploaded.

## Run locally
```bash
python3 -m pip install -r requirements.txt
cp .env.example .env          # optional: add GROQ_API_KEY / GEMINI_API_KEY
RISK_BACKEND=hgb python3 -m uvicorn backend.app:app --port 8000
cd frontend && cp .env.example .env.local && npm install && npm run dev   # http://localhost:3000
python3 -m pytest -q tests    # 16 tests
```
Windows: use `py -m pip` / `py -m uvicorn`, and set the variable with `set RISK_BACKEND=hgb` first.

## 1. Benchmark TabPFN (Colab, ~15 min)
Open `ml/notebooks/HealthLens_v3_Benchmark.ipynb` in Colab, follow the first cell (accept the licence on Hugging Face, add `HF_TOKEN` as a Colab secret, T4 GPU), run all, and put the downloaded `model_card.json` into `artifacts/`.

## 2. Deploy the backend (Azure Container Apps)
1. Push this repo to GitHub. The workflow `.github/workflows/backend-image.yml` builds `ghcr.io/<you>/<repo>-api:latest` (make the package public, or add registry credentials in Azure).
2. Azure portal: create a **Container App** (Azure for Students; pick a region allowed for your subscription, Poland Central worked before), image from ghcr.io, ingress **external**, port **8000**, min replicas 0 or 1.
3. Environment variables / secrets: `CORS_ORIGINS=https://<your-vercel-domain>`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`, `RISK_BACKEND=auto`. Give it at least 2 vCPU and 4 GiB; TabPFN on CPU is slow (see the timings the notebook prints). If requests are too slow, set `RISK_BACKEND=hgb` and keep TabPFN as the benchmarked model.
4. Check `https://<app>.azurecontainerapps.io/api/health`.

## 3. Deploy the frontend (Vercel)
Import the repo, set **Root Directory** to `frontend`, add `NEXT_PUBLIC_API_URL=https://<app>.azurecontainerapps.io`, deploy. Then put the Vercel URL into the backend's `CORS_ORIGINS`.

## Retrain local models
`sh scripts/train_all.sh` (only needed if you change the data or code; the shipped artifacts already match).
