"""Inference wrapper: patient -> plain-text narrative from the locally trained language model."""
from __future__ import annotations

import random
import threading

import torch

from .. import bands, config
from . import corpus as C
from .lm import GPT, Vocab, generate


class Narrator:
    def __init__(self, stats: dict) -> None:
        self.stats = stats
        self.vocab = Vocab.load(config.LM_VOCAB_PATH)
        ckpt = torch.load(config.LM_WEIGHTS_PATH, map_location="cpu")
        self.model = GPT(**ckpt["cfg"])
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()
        self._lock = threading.Lock()

    # ---- placeholder values come from the real patient and the real dataset statistics ----------
    def _values(self, p: dict, prob: float) -> dict[str, str]:
        pv = self.stats["prevalence"]
        if prob < 0.001:
            prob_s = "<0.1%"
        elif prob > 0.999:
            prob_s = ">99.9%"
        else:
            prob_s = bands.fmt_pct(prob)
        return {
            "age": bands.fmt_num(float(p["age"]), 1),
            "hba1c": f"{float(p['hba1c']):.1f}",
            "glucose": bands.fmt_num(float(p["glucose"]), 0),
            "bmi": f"{float(p['bmi']):.1f}",
            "prob": prob_s,
            "prev_hba1c": bands.fmt_pct(pv["hba1c"][bands.hba1c_band(float(p["hba1c"]))]),
            "prev_glucose": bands.fmt_pct(pv["glucose"][bands.glucose_band(float(p["glucose"]))]),
            "prev_bmi": bands.fmt_pct(pv["bmi"][bands.bmi_band(float(p["bmi"]))]),
            "prev_age": bands.fmt_pct(pv["age"][bands.age_band(float(p["age"]))]),
            "prev_gender": bands.fmt_pct(pv["gender"][p["gender"]]),
            "prev_all": bands.fmt_pct(pv["all"]),
            "prev_htn": bands.fmt_pct(pv["hypertension"]),
            "prev_hd": bands.fmt_pct(pv["heart_disease"]),
        }

    def _fill(self, text: str, values: dict[str, str]) -> str:
        for k, v in values.items():
            text = text.replace(f"<{k}>", v)
        return text

    def narrate(self, patient: dict, prob: float, tier: str) -> dict:
        f = C.fields_from_values(patient["gender"], float(patient["age"]), float(patient["hba1c"]),
                                 float(patient["glucose"]), float(patient["bmi"]), patient["hypertension"],
                                 patient["heart_disease"], patient["smoking_history"], tier)
        prompt = [self.vocab.bos] + self.vocab.encode(C.control_tokens(f)) + [self.vocab.sep]
        attempts = 0
        problems: list[str] = []
        raw = None
        with self._lock:
            for temperature, top_k in ((0.5, 5), (0.3, 3), (0.0, 0)):
                attempts += 1
                ids = generate(self.model, prompt, self.vocab.eos, temperature=temperature, top_k=top_k)
                raw = C.detokenize(self.vocab.decode(ids))
                problems = C.check(f, raw)
                if not problems:
                    break
        values = self._values(patient, prob)
        if not problems:
            return {"text": self._fill(raw, values), "source": "local-lm", "validated": True,
                    "attempts": attempts, "problems": []}
        # Safety net: the dataset-derived template (the very text the LM was trained on)
        text = C.build_narrative(f, random.Random(0))
        return {"text": self._fill(text, values), "source": "template-fallback", "validated": True,
                "attempts": attempts, "problems": problems[:3]}
