"""A small GPT (decoder-only transformer) trained from scratch -- no pretrained weights.

Vocabulary and training text come exclusively from the dataset-derived narrative corpus
(see corpus.py). KV-caching makes generation fast enough for interactive use on a CPU.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import corpus as C


# ------------------------------------------------------------------ vocabulary
def all_vocab_tokens() -> list[str]:
    """Deterministic vocabulary: every token any template / control field can produce."""
    toks: set[str] = set()

    def tk(text: str) -> list[str]:
        # tokens of a text both at the start of the narrative and mid-narrative
        # (where its first word carries the leading-space marker)
        return C.tokenize(text) + C.tokenize(" " + text)

    for tmpl in C.S1:
        for sex, tier in itertools.product(["female", "male"], C.TIER_WORDS):
            toks.update(tk(tmpl.format(sex=sex, tier=tier)))
    for tmpl in C.S2:
        for b in C.HBA1C_WORD:
            toks.update(tk(tmpl.format(w=C.HBA1C_WORD[b], r=C.HBA1C_RANGE[b])))
    for tmpl in C.S3:
        for b in C.GLUC_WORD:
            toks.update(tk(tmpl.format(w=C.GLUC_WORD[b], r=C.GLUC_RANGE[b])))
    for tmpl in C.S4:
        for b in C.BMI_WORD:
            toks.update(tk(tmpl.format(w=C.BMI_WORD[b])))
    for tmpl in C.S5:
        for ap, sex in itertools.product(C.AGE_PHRASE.values(), ["female", "male"]):
            toks.update(tk(tmpl.format(ap=ap, sex=sex)))
    for variants in C.S6.values():
        for t in variants:
            toks.update(tk(t))
    for variants in C.S7.values():
        for t in variants:
            toks.update(tk(t))
    for variants in C.S8.values():
        for t in variants:
            toks.update(tk(t))
    toks.update(tk(C.S9))
    ctrl = {
        "gender": ["female", "male"], "age": list(C.AGE_PHRASE), "hba1c": list(C.HBA1C_WORD),
        "glucose": list(C.GLUC_WORD), "bmi": list(C.BMI_WORD), "htn": ["yes", "no"], "hd": ["yes", "no"],
        "smoke": list(C.SMOKE_KEY.values()), "tier": list(C.TIER_WORDS),
    }
    for k, vals in ctrl.items():
        toks.update(f"{k}={v}" for v in vals)
    return C.SPECIAL + sorted(toks)


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.pad, self.bos, self.sep, self.eos = (self.stoi[t] for t in C.SPECIAL)

    def encode(self, toks: list[str]) -> list[int]:
        return [self.stoi[t] for t in toks]  # KeyError == a token outside the dataset-derived vocab

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.itos))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(path.read_text()))


# ------------------------------------------------------------------ model
class Attention(nn.Module):
    def __init__(self, d: int, h: int, p: float):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(p)
        self.p = p

    def forward(self, x, past=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v)
        if past is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0)
        else:  # single new token attends to the whole cache
            y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.drop(self.proj(y)), present


class Block(nn.Module):
    def __init__(self, d: int, h: int, p: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, h, p)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(p))

    def forward(self, x, past=None):
        a, present = self.attn(self.ln1(x), past)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, vocab_size: int, d: int = 128, n_layer: int = 3, n_head: int = 4, max_len: int = 288, p: float = 0.1):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, d=d, n_layer=n_layer, n_head=n_head, max_len=max_len, p=p)
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_len, d)
        self.drop = nn.Dropout(p)
        self.blocks = nn.ModuleList([Block(d, n_head, p) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, past=None):
        B, T = idx.shape
        offset = 0 if past is None else past[0][0].shape[2]
        pos = torch.arange(offset, offset + T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        presents = []
        for i, blk in enumerate(self.blocks):
            x, pr = blk(x, None if past is None else past[i])
            presents.append(pr)
        return self.head(self.ln_f(x)), presents

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def generate(model: GPT, prompt: list[int], eos: int, max_new: int = 270, temperature: float = 0.5,
             top_k: int = 5, gen: torch.Generator | None = None) -> list[int]:
    model.eval()
    max_len = model.cfg["max_len"]
    ids = torch.tensor([prompt], dtype=torch.long)
    logits, past = model(ids)
    out: list[int] = []
    for _ in range(min(max_new, max_len - len(prompt))):
        step = logits[0, -1]
        if temperature <= 0:
            nxt = int(step.argmax())
        else:
            step = step / temperature
            if top_k:
                v, _ = torch.topk(step, top_k)
                step = step.masked_fill(step < v[-1], float("-inf"))
            nxt = int(torch.multinomial(F.softmax(step, dim=-1), 1, generator=gen))
        if nxt == eos:
            break
        out.append(nxt)
        logits, past = model(torch.tensor([[nxt]]), past)
    return out
