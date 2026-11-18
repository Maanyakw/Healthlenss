#!/usr/bin/env sh
# Retrain every local model from scratch. Run from the project root.
# TabPFN-3.5 is NOT trained here (it is pretrained); benchmark it with ml/notebooks/HealthLens_v3_Benchmark.ipynb.
set -e
python3 -m backend.risk_model                      # XGBoost, logistic regression, HGB, tiers, statistics, TabPFN context (~2 min)
python3 -m backend.narrator.train_lm --steps 2000 --batch 32 --lr 2e-3 --max_minutes 25   # narrator language model
python3 -m backend.vit_ocr.train_vit --train_n 80000 --steps 9000 --max_minutes 55         # ViT lab-report reader
python3 samples/make_samples.py --out frontend/public/samples --n 6 --seed 1
