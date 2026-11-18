"""Risk models, comparison benchmark, exact Shapley explanation and counterfactual search.

Primary model: TabPFN-3.5 (pretrained tabular transformer, fitted on a stratified context sample of the
training split). Comparison models: XGBoost and logistic regression, both trained on the full training split.
A histogram gradient-boosting model is kept as a fast local stand-in, used by the server when TabPFN is not
installed or its context/weights are unavailable, and for the unit tests.

Dataset: 'Diabetes prediction dataset' (100k records, female + male patients).
Run `python -m backend.risk_model` to (re)train the local models and write artifacts/*.
Run the Colab notebook ml/notebooks/HealthLens_v3_Benchmark.ipynb to benchmark TabPFN-3.5.
"""
from __future__ import annotations

import json
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import bands
from .bands import GENDERS, SMOKING
from . import config
from .config import ART_DIR, DATASET_CSV, RISK_MODEL_PATH, STATS_PATH

SEED = 42
FEATURES = ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi", "hba1c", "glucose"]
CATEGORICAL_IDX = [0, 4]
FEATURE_LABELS = {
    "gender": "Sex",
    "age": "Age",
    "hypertension": "Hypertension",
    "heart_disease": "Heart disease",
    "smoking_history": "Smoking history",
    "bmi": "BMI",
    "hba1c": "HbA1c",
    "glucose": "Blood glucose",
}

_G = {g: i for i, g in enumerate(GENDERS)}
_S = {s: i for i, s in enumerate(SMOKING)}


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def load_dataset() -> tuple[pd.DataFrame, dict]:
    """Load the CSV, keep female/male patients, drop exact duplicates."""
    raw = pd.read_csv(DATASET_CSV)
    raw = raw.rename(columns={"HbA1c_level": "hba1c", "blood_glucose_level": "glucose", "diabetes": "label"})
    info = {
        "raw_rows": int(len(raw)),
        "raw_gender_counts": {k: int(v) for k, v in raw["gender"].value_counts().items()},
    }
    df = raw[raw["gender"].isin(GENDERS)].copy()
    info["dropped_other_gender_rows"] = int(len(raw) - len(df))
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    info["dropped_duplicate_rows"] = int(before - len(df))
    info["rows_used"] = int(len(df))
    return df, info


def encode(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        df["gender"].map(_G).to_numpy(float),
        df["age"].to_numpy(float),
        df["hypertension"].to_numpy(float),
        df["heart_disease"].to_numpy(float),
        df["smoking_history"].map(_S).to_numpy(float),
        df["bmi"].to_numpy(float),
        df["hba1c"].to_numpy(float),
        df["glucose"].to_numpy(float),
    ])


def _new_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        categorical_features=CATEGORICAL_IDX, random_state=SEED,
    )


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def _metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
        "pr_auc": float(average_precision_score(y, p)) if y.sum() > 0 else None,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "specificity": tn / max(tn + fp, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# --------------------------------------------------------------------------------------
# data split, thresholds, metrics helpers
# --------------------------------------------------------------------------------------
def split_data(df: pd.DataFrame):
    """Stratified 70/15/15 split (by label x gender). Identical in training, notebook and tests."""
    strat = df["label"].astype(str) + "_" + df["gender"]
    trainval, test = train_test_split(df, test_size=0.15, stratify=strat, random_state=SEED)
    strat_tv = trainval["label"].astype(str) + "_" + trainval["gender"]
    train, val = train_test_split(trainval, test_size=0.1765, stratify=strat_tv, random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def screening_threshold(p_val: np.ndarray, y_val: np.ndarray, target_recall: float = 0.97) -> float:
    """Lowest cut-off that still catches >= target_recall of validation positives, clamped to [0.02, 0.25]."""
    order = np.argsort(-p_val)
    cum = np.cumsum(y_val[order]) / max(y_val.sum(), 1)
    k = int(np.searchsorted(cum, target_recall))
    return round(float(np.clip(p_val[order][min(k, len(order) - 1)], 0.02, 0.25)), 4)


def auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 300, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            vals.append(roc_auc_score(y[idx], p[idx]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def comparison_row(name: str, role: str, y_val, p_val, y_te, p_te, cv_auc: float | None = None, extra: dict | None = None) -> dict:
    thr = screening_threshold(p_val, y_val)
    m = _metrics(y_te, p_te, thr)
    return {"name": name, "role": role, "cv_auc": None if cv_auc is None else round(float(cv_auc), 4),
            "test_auc": round(m["roc_auc"], 4), "test_auc_ci95": auc_ci(y_te, p_te),
            "screening_threshold": thr, "recall": round(m["recall"], 4), "precision": round(m["precision"], 4),
            "brier": round(float(brier_score_loss(y_te, p_te)), 4), **(extra or {})}


def _new_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.9,
                         tree_method="hist", eval_metric="logloss", n_jobs=2, random_state=SEED)


def _new_lr():
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_IDX),
                             ("num", StandardScaler(), [1, 2, 3, 5, 6, 7])])
    return make_pipeline(pre, LogisticRegression(max_iter=2000))


# --------------------------------------------------------------------------------------
# training (local models) -- writes artifacts used by the backend and the Colab notebook
# --------------------------------------------------------------------------------------
def train() -> dict:
    t0 = time.time()
    df, info = load_dataset()
    train, val, test = split_data(df)
    Xtr, ytr = encode(train), train["label"].to_numpy()
    Xva, yva = encode(val), val["label"].to_numpy()
    Xte, yte = encode(test), test["label"].to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # --- local models: HGB (server stand-in), XGBoost and logistic regression (comparisons) --------
    hgb = _new_hgb().fit(Xtr, ytr)
    xgb = _new_xgb().fit(Xtr, ytr)
    lr = _new_lr().fit(Xtr, ytr)
    P = {k: (m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]) for k, m in (("hgb", hgb), ("xgb", xgb), ("lr", lr))}
    oof_hgb = cross_val_predict(_new_hgb(), Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
    cv = {"hgb": roc_auc_score(ytr, oof_hgb)}
    for k, maker in (("xgb", _new_xgb), ("lr", _new_lr)):
        cv[k] = roc_auc_score(ytr, cross_val_predict(maker(), Xtr, ytr, cv=skf, method="predict_proba")[:, 1])

    pva, pte = P["hgb"]
    thresholds = {"moderate": screening_threshold(pva, yva), "high": 0.5}
    train["oof_prob"] = oof_hgb
    train["tier"] = [bands.tier_from_prob(p, thresholds) for p in oof_hgb]

    comparison = [
        {"name": "TabPFN-3.5", "role": "primary", "status": "not benchmarked yet: run ml/notebooks/HealthLens_v3_Benchmark.ipynb"},
        comparison_row("XGBoost", "comparison", yva, P["xgb"][0], yte, P["xgb"][1], cv["xgb"]),
        comparison_row("Logistic regression", "comparison", yva, P["lr"][0], yte, P["lr"][1], cv["lr"]),
        comparison_row("Gradient boosting (HGB)", "server stand-in", yva, P["hgb"][0], yte, P["hgb"][1], cv["hgb"]),
    ]

    metrics: dict[str, Any] = {
        "decision_threshold_for_screening_metrics": thresholds["moderate"],
        "test_overall": _metrics(yte, pte, thresholds["moderate"]),
        "test_by_gender": {},
        "brier": float(brier_score_loss(yte, pte)),
        "ece": _ece(yte, pte),
        "n_trees": int(hgb.n_iter_),
    }
    for g in GENDERS:
        m = (test["gender"] == g).to_numpy()
        metrics["test_by_gender"][g] = _metrics(yte[m], pte[m], thresholds["moderate"])
    rule = ((test["hba1c"] >= 6.5) | (test["glucose"] >= 200)).to_numpy()
    metrics["baseline_rule_hba1c>=6.5_or_glucose>=200"] = {
        "recall": float((rule & (yte == 1)).sum() / max(yte.sum(), 1)),
        "precision": float((rule & (yte == 1)).sum() / max(rule.sum(), 1)),
    }
    metrics["tier_distribution_test"] = {
        t: float(np.mean([bands.tier_from_prob(p, thresholds) == t for p in pte])) for t in ("low", "moderate", "high")
    }
    metrics["tier_observed_prevalence_test"] = {}
    for t in ("low", "moderate", "high"):
        m = np.array([bands.tier_from_prob(p, thresholds) == t for p in pte])
        metrics["tier_observed_prevalence_test"][t] = float(yte[m].mean()) if m.any() else None

    def prev(mask: pd.Series) -> float:
        return float(train.loc[mask, "label"].mean()) if mask.sum() else 0.0

    prevalence = {
        "all": prev(train["label"].notna()),
        "hba1c": {b: prev(train["hba1c"].map(bands.hba1c_band) == b) for b in ("normal", "prediabetes", "diabetes")},
        "glucose": {b: prev(train["glucose"].map(bands.glucose_band) == b) for b in ("normal", "elevated", "high", "very_high")},
        "bmi": {b: prev(train["bmi"].map(bands.bmi_band) == b) for b in ("underweight", "healthy", "overweight", "obese")},
        "age": {b: prev(train["age"].map(bands.age_band) == b) for b in ("young", "adult", "midlife", "older")},
        "gender": {g: prev(train["gender"] == g) for g in GENDERS},
        "hypertension": prev(train["hypertension"] == 1),
        "heart_disease": prev(train["heart_disease"] == 1),
    }
    ranges = {c: [float(df[c].min()), float(df[c].max())] for c in ("age", "bmi", "hba1c", "glucose")}
    dataset_summary = {
        **info,
        "split": {"train": len(train), "validation": len(val), "test": len(test)},
        "by_gender": {g: {"rows": int((df["gender"] == g).sum()), "share": float((df["gender"] == g).mean()),
                          "diabetes_rate": float(df.loc[df["gender"] == g, "label"].mean())} for g in GENDERS},
        "diabetes_rate": float(df["label"].mean()),
        "feature_ranges": ranges,
        "source": "Diabetes prediction dataset (Kaggle, 100,000 records)",
    }

    ref = train[train["label"] == 0].sample(300, random_state=SEED)
    ART_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump({"model": hgb, "thresholds": thresholds, "reference": encode(ref), "ranges": ranges}, RISK_MODEL_PATH)
    stats = {"prevalence": prevalence, "thresholds": thresholds, "metrics": metrics, "dataset": dataset_summary,
             "comparison": comparison, "primary_model": "TabPFN-3.5"}
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    train.to_csv(ART_DIR / "train_split_with_tiers.csv", index=False)
    val.to_csv(ART_DIR / "val_split.csv", index=False)

    # TabPFN context: stratified sample of the TRAINING split only (test rows never enter it)
    n_ctx = min(config.TABPFN_CONTEXT_ROWS, len(train))
    ctx, _ = train_test_split(train, train_size=n_ctx, stratify=train["label"].astype(str) + "_" + train["gender"], random_state=SEED)
    ctx_out = pd.DataFrame(encode(ctx), columns=FEATURES)
    ctx_out["label"] = ctx["label"].to_numpy()
    ctx_out.to_csv(config.TABPFN_CONTEXT_CSV, index=False)
    print(f"trained in {time.time() - t0:.1f}s")
    for c in comparison:
        print(c)
    return stats


# --------------------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------------------
class RiskModel:
    """Risk estimate for one patient. backend = 'tabpfn' (primary) or 'hgb' (local stand-in)."""

    NOT_REALISTIC = {"glucose": 40.0, "hba1c": 1.0, "bmi": 5.0}

    def __init__(self, backend: str | None = None) -> None:
        import threading
        bundle = joblib.load(RISK_MODEL_PATH)
        self.hgb: HistGradientBoostingClassifier = bundle["model"]
        self.thresholds: dict = bundle["thresholds"]
        self.ranges: dict = bundle["ranges"]
        reference: np.ndarray = bundle["reference"]
        self.backend = "hgb"
        self.backend_note = ""
        self._clf = None
        self._lock = threading.Lock()
        want = (backend or config.RISK_BACKEND)
        if want in ("auto", "tabpfn"):
            try:
                self._load_tabpfn()
                self.backend = "tabpfn"
            except Exception as e:  # missing package, weights, context file, licence not accepted...
                self.backend_note = f"TabPFN unavailable ({type(e).__name__}: {str(e)[:120]}); using the gradient-boosting stand-in."
                if want == "tabpfn":
                    raise
        if self.backend == "tabpfn":
            # 4 representative non-diabetic rows keep exact Shapley affordable (256 coalitions x 4 rows)
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=4, n_init=5, random_state=SEED).fit(reference)
            idx = [int(np.argmin(((reference - c) ** 2).sum(1))) for c in km.cluster_centers_]
            self.reference = reference[sorted(set(idx))]
        else:
            self.reference = reference[:100]

    # ---- TabPFN ----------------------------------------------------------------------------
    def _load_tabpfn(self) -> None:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        if not config.TABPFN_CONTEXT_CSV.exists():
            raise FileNotFoundError("artifacts/tabpfn_context.csv missing")
        ctx = pd.read_csv(config.TABPFN_CONTEXT_CSV)
        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V3_5, n_estimators=1, fit_mode="fit_with_cache", categorical_features_indices=CATEGORICAL_IDX)
        clf.fit(ctx[FEATURES].to_numpy(float), ctx["label"].to_numpy(int))
        clf.predict_proba(ctx[FEATURES].to_numpy(float)[:2])  # warm-up
        self._clf = clf
        if config.MODEL_CARD_PATH.exists():
            card = json.loads(config.MODEL_CARD_PATH.read_text())
            if card.get("thresholds"):
                self.thresholds = card["thresholds"]

    def _proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is not None:
            with self._lock:
                return self._clf.predict_proba(X)[:, 1]
        return self.hgb.predict_proba(X)[:, 1]

    @staticmethod
    def _row(p: dict) -> np.ndarray:
        return np.array([[
            _G[p["gender"]], float(p["age"]), float(p["hypertension"]), float(p["heart_disease"]),
            _S[p["smoking_history"]], float(p["bmi"]), float(p["hba1c"]), float(p["glucose"]),
        ]])

    def prob(self, patient: dict) -> float:
        return float(self._proba(self._row(patient))[0])

    def prob_many(self, patients: list[dict]) -> np.ndarray:
        return self._proba(np.vstack([self._row(p) for p in patients]))

    def warnings(self, patient: dict) -> list[str]:
        out = []
        names = {"age": "Age", "bmi": "BMI", "hba1c": "HbA1c", "glucose": "Blood glucose"}
        for key, (lo, hi) in self.ranges.items():
            v = float(patient[key])
            if v < lo or v > hi:
                out.append(f"{names[key]} {bands.fmt_num(v)} is outside the range seen in the training data "
                           f"({bands.fmt_num(lo)}–{bands.fmt_num(hi)}); the estimate may be unreliable.")
        if float(patient["glucose"]) < 70:
            out.append(f"Blood glucose {bands.fmt_num(float(patient['glucose']))} mg/dL is low (below 70). This tool estimates diabetes risk "
                       "from high values, so a low reading is not reassuring and needs separate medical review.")
        # Discordance: the models lean mostly on HbA1c, so one marker in the diabetes range can be
        # outvoted by a normal one. Say so plainly, independent of the model's output.
        hb, gl = float(patient["hba1c"]), float(patient["glucose"])
        if (hb >= 6.5 or gl >= 126) and self.prob(patient) < self.thresholds["high"]:
            marker = f"HbA1c {bands.fmt_num(hb)}%" if hb >= 6.5 else f"Blood glucose {bands.fmt_num(gl)} mg/dL"
            out.append(f"{marker} is in the range used to diagnose diabetes, yet the model's estimate is below the high tier. "
                       "The model relies mostly on HbA1c and can be outvoted by a normal value. Do not treat this result as "
                       "reassuring; repeat testing with a doctor is advised.")
        return out

    def explain(self, patient: dict) -> list[dict]:
        """Exact Shapley attribution on the probability scale (all 2^8 feature coalitions).

        Baseline = average prediction over the reference rows (typical non-diabetic patients); each
        feature's value is its Shapley share of (estimate - baseline). Contributions sum exactly to that gap.
        """
        from math import factorial
        n = len(FEATURES)
        base = self._row(patient)[0]
        ref = self.reference
        masks = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)
        X = np.where(masks[:, None, :], base[None, None, :], ref[None, :, :])
        v = self._proba(X.reshape(-1, n)).reshape(2 ** n, len(ref)).mean(axis=1)
        sizes = masks.sum(axis=1)
        w = np.array([factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)])
        index = {tuple(m): i for i, m in enumerate(masks)}
        phi = np.zeros(n)
        for j in range(n):
            for i, m in enumerate(masks):
                if not m[j]:
                    m2 = m.copy(); m2[j] = True
                    phi[j] += w[sizes[i]] * (v[index[tuple(m2)]] - v[i])
        impacts = [{"feature": name, "label": FEATURE_LABELS[name], "value": patient[name],
                    "impact_pct_points": round(float(phi[j]) * 100, 2) + 0.0} for j, name in enumerate(FEATURES)]
        impacts.sort(key=lambda d: -abs(d["impact_pct_points"]))
        return impacts

    def counterfactuals(self, patient: dict, max_options: int = 3) -> dict:
        """Search for the smallest lowering of HbA1c, glucose and BMI that moves the estimate one tier down.

        Single-feature moves are tried first, then combinations on a coarse grid. Options beyond
        40 mg/dL, 1.0 HbA1c point or 5 BMI points are flagged as not realistic. These are what-if
        calculations of the model's output, not medical advice.
        """
        p0 = self.prob(patient)
        tier = bands.tier_from_prob(p0, self.thresholds)
        if tier == "low":
            return {"tier": tier, "target": None, "options": [], "note": "The estimate is already in the low tier."}
        target = self.thresholds["high"] if tier == "high" else self.thresholds["moderate"]
        lo = {"hba1c": max(float(self.ranges["hba1c"][0]), 4.0), "glucose": float(self.ranges["glucose"][0]),
              "bmi": 18.5}
        steps = {"hba1c": 0.1, "glucose": 5.0, "bmi": 0.5}
        cap = {"hba1c": 2.0, "glucose": 60.0, "bmi": 8.0}
        cur = {k: float(patient[k]) for k in steps}

        def grid(k, coarse):
            step = steps[k] * (4 if coarse else 1)
            vals, d = [], 0.0
            while d + step <= cap[k] + 1e-9 and cur[k] - (d + step) >= lo[k] - 1e-9:
                d += step
                vals.append(round(d, 3))
            return vals

        cands = []
        for k in steps:                                            # single-feature moves
            cands += [{k: d} for d in grid(k, False)]
        g = {k: [0.0] + grid(k, True) for k in steps}              # coarse combinations
        for a in g["hba1c"]:
            for b in g["glucose"]:
                for c in g["bmi"]:
                    nz = (a > 0) + (b > 0) + (c > 0)
                    if nz >= 2:
                        cands.append({k: d for k, d in (("hba1c", a), ("glucose", b), ("bmi", c)) if d > 0})
        probs = self.prob_many([{**patient, **{k: round(cur[k] - d, 2) for k, d in c.items()}} for c in cands])
        found = []
        for c, p in zip(cands, probs):
            if p < target:
                cost = sum(d / cap[k] for k, d in c.items())
                found.append((cost, c, float(p)))
        found.sort(key=lambda t: t[0])
        options, seen = [], set()
        for cost, c, p in found:
            key = tuple(sorted(c))
            if key in seen:
                continue
            seen.add(key)
            changes = [{"feature": k, "label": FEATURE_LABELS[k], "from": cur[k], "to": round(cur[k] - d, 2), "change": -d}
                       for k, d in c.items()]
            options.append({"changes": changes, "new_probability": p,
                            "new_tier": bands.tier_from_prob(p, self.thresholds),
                            "realistic": all(d <= self.NOT_REALISTIC[k] for k, d in c.items())})
            if len(options) >= max_options:
                break
        note = "" if options else "No combination of lower HbA1c, glucose and BMI within the searched range brings the estimate down a tier."
        return {"tier": tier, "target": target, "options": options, "note": note}


if __name__ == "__main__":
    train()
