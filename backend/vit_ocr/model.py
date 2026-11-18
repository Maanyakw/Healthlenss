"""Vision Transformer with a CTC head for reading a text chunk (ViTSTR / TrOCR-style encoder).

The chunk image (1 x 32 x 512) is cut into 32x8 pixel patches (64 tokens), linearly embedded,
given learned position embeddings and passed through a transformer encoder. Each output token
predicts a character distribution; CTC aligns them to the text, so no character segmentation
is needed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import preprocess as P
from .charset import IDX2CHAR, N_CLASSES


class ViTCTC(nn.Module):
    def __init__(self, d: int = 128, depth: int = 4, heads: int = 4, patch_w: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cfg = dict(d=d, depth=depth, heads=heads, patch_w=patch_w, dropout=dropout)
        self.patch = nn.Conv2d(1, d, kernel_size=(P.H, patch_w), stride=(P.H, patch_w))  # linear patch embedding
        self.n_tokens = P.W // patch_w
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 32, 512) in [0, 1] (dark text = low). Returns logits (B, T, n_classes)."""
        t = self.patch(1.0 - x).flatten(2).transpose(1, 2) + self.pos  # ink = high activation
        return self.head(self.norm(self.encoder(t)))


def greedy_decode(logits: torch.Tensor) -> list[tuple[str, float]]:
    """CTC greedy decoding. Confidence = mean max-probability over the emitted (non-blank) frames."""
    probs = logits.softmax(-1)
    conf, idx = probs.max(-1)
    out = []
    for b in range(idx.shape[0]):
        seq, cs, prev = [], [], 0
        for t in range(idx.shape[1]):
            k = int(idx[b, t])
            if k != 0 and k != prev:
                seq.append(IDX2CHAR[k])
                cs.append(float(conf[b, t]))
            prev = k
        out.append(("".join(seq), float(np.mean(cs)) if cs else 0.0))
    return out
