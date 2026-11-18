"""Train the ViT-CTC line recogniser on synthetic lab-report chunks.

    python -m backend.vit_ocr.train_vit --train_n 60000 --steps 6000

Model selection uses validation chunks rendered with the *training* fonts; chunks rendered with
HELD-OUT fonts are used only for reporting, so the reported numbers reflect unseen typefaces.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..config import ART_DIR, VIT_WEIGHTS_PATH
from . import synth
from .charset import encode
from .model import ViTCTC, greedy_decode


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def to_tensor(X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(X).float().div_(255.0).unsqueeze(1)


@torch.no_grad()
def evaluate(model, X: np.ndarray, labels: list[str]) -> dict:
    model.eval()
    exact, err, tot = 0, 0, 0
    bad = []
    for i in range(0, len(X), 128):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(to_tensor(X[i:i + 128]))
        for (pred, _), gold in zip(greedy_decode(logits.float()), labels[i:i + 128]):
            exact += pred == gold
            err += edit_distance(pred, gold)
            tot += len(gold)
            if pred != gold and len(bad) < 8:
                bad.append((gold, pred))
    model.train()
    return {"exact_match": exact / len(X), "cer": err / max(tot, 1), "examples_wrong": bad}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_n", type=int, default=60000)
    ap.add_argument("--val_n", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--max_minutes", type=float, default=60)
    args = ap.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(0)
    t0 = time.time()
    print("generating synthetic data ...", flush=True)
    Xtr, Ltr = synth.generate_parallel(args.train_n, 1, synth.TRAIN_FONTS, procs=2)
    Xvs, Lvs = synth.generate_dataset(args.val_n, 777, synth.TRAIN_FONTS)
    Xvh, Lvh = synth.generate_dataset(args.val_n, 888, synth.HELD_OUT_FONTS)
    print(f"data ready in {time.time() - t0:.0f}s  train={len(Xtr)}", flush=True)

    model = ViTCTC(d=args.d, depth=args.depth)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"ViT-CTC parameters: {n_par:,}  tokens/chunk: {model.n_tokens}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    tgt_all = [torch.tensor(encode(l), dtype=torch.long) for l in Ltr]

    rng = np.random.default_rng(0)
    warm = 300
    best, hist = 1e9, []
    t1 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = rng.integers(0, len(Xtr), args.batch)
        x = to_tensor(Xtr[idx])
        targets = [tgt_all[i] for i in idx]
        tl = torch.tensor([len(t) for t in targets])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(x)
        lp = logits.float().log_softmax(-1).transpose(0, 1)  # (T, B, C)
        il = torch.full((args.batch,), lp.shape[0], dtype=torch.long)
        loss = F.ctc_loss(lp, torch.cat(targets), il, tl, blank=0, zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"step {step:5d} ctc {loss.item():.4f} lr {lr:.5f} elapsed {time.time() - t1:6.0f}s", flush=True)
        done = step == args.steps or (time.time() - t1) / 60 > args.max_minutes
        if step % 500 == 0 or done:
            es = evaluate(model, Xvs, Lvs)
            print(f"   val(train fonts) exact {es['exact_match']:.3f} cer {es['cer']:.4f}", flush=True)
            hist.append({"step": step, "val_seen": es["exact_match"], "cer_seen": es["cer"]})
            if es["cer"] < best:
                best = es["cer"]
                torch.save({"cfg": model.cfg, "state": model.state_dict()}, VIT_WEIGHTS_PATH)
        if done:
            if step != args.steps:
                print("time budget reached; stopping", flush=True)
            break

    ckpt = torch.load(VIT_WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(ckpt["state"])
    final_seen = evaluate(model, Xvs, Lvs)
    final_held = evaluate(model, Xvh, Lvh)
    metrics = {
        "parameters": n_par, "steps_run": step, "train_samples": len(Xtr),
        "train_fonts": len(synth.TRAIN_FONTS), "held_out_fonts": len(synth.HELD_OUT_FONTS),
        "chunk_exact_match_train_fonts": final_seen["exact_match"], "chunk_cer_train_fonts": final_seen["cer"],
        "chunk_exact_match_heldout_fonts": final_held["exact_match"], "chunk_cer_heldout_fonts": final_held["cer"],
        "examples_wrong_heldout": final_held["examples_wrong"], "history": hist,
        "train_minutes": round((time.time() - t0) / 60, 1),
    }
    (ART_DIR / "vit_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
