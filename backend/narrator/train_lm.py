"""Train the narrator language model from scratch on dataset-derived narratives.

    python -m backend.narrator.train_lm --steps 1500

Training text is generated on the fly from rows of the *training split only* (labelled with
out-of-fold risk tiers by the classifier). 5% of the distinct control-field combinations are
held out entirely to test whether the model generalises to unseen patient profiles.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..config import ART_DIR, LM_VOCAB_PATH, LM_WEIGHTS_PATH
from . import corpus as C
from .lm import GPT, Vocab, all_vocab_tokens, generate


def row_fields(r) -> dict:
    return C.fields_from_values(r.gender, r.age, r.hba1c, r.glucose, r.bmi, r.hypertension,
                                r.heart_disease, r.smoking_history, r.tier)


def build_example(vocab: Vocab, f: dict, rng: random.Random, max_len: int) -> tuple[list[int], int]:
    prompt = [vocab.bos] + vocab.encode(C.control_tokens(f)) + [vocab.sep]
    body = vocab.encode(C.tokenize(C.build_narrative(f, rng))) + [vocab.eos]
    ids = prompt + body
    assert len(ids) <= max_len, f"sequence too long: {len(ids)}"
    return ids, len(prompt)


def make_batch(vocab: Vocab, items: list[tuple[list[int], int]]):
    L = max(len(i) for i, _ in items)
    x = torch.full((len(items), L), vocab.pad, dtype=torch.long)
    mask = torch.zeros((len(items), L), dtype=torch.bool)
    for b, (ids, plen) in enumerate(items):
        x[b, :len(ids)] = torch.tensor(ids)
        mask[b, plen:len(ids)] = True  # loss on narrative (+eos) only
    return x, mask


def lm_loss(model: GPT, x, mask):
    logits, _ = model(x[:, :-1])
    tgt = x[:, 1:]
    m = mask[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none").view_as(tgt)
    return (loss * m).sum() / m.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--max_minutes", type=float, default=40)
    args = ap.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(0)
    rng = random.Random(0)
    max_len = 288

    itos = all_vocab_tokens()
    vocab = Vocab(itos)
    vocab.save(LM_VOCAB_PATH)

    train = pd.read_csv(ART_DIR / "train_split_with_tiers.csv")
    rows = [row_fields(r) for r in train.itertuples(index=False)]
    combos: dict[tuple, int] = {}
    for f in rows:
        combos[tuple(f[k] for k in C.CONTROL_KEYS)] = combos.get(tuple(f[k] for k in C.CONTROL_KEYS), 0) + 1
    keys = sorted(combos)
    rng.shuffle(keys)
    n_hold = max(1, int(0.05 * len(keys)))
    held = set(keys[:n_hold])
    train_rows = [f for f in rows if tuple(f[k] for k in C.CONTROL_KEYS) not in held]
    train_combos = [dict(zip(C.CONTROL_KEYS, k)) for k in keys[n_hold:]]
    held_combos = [dict(zip(C.CONTROL_KEYS, k)) for k in keys[:n_hold]]
    print(f"vocab={len(itos)} rows={len(rows)} distinct combos={len(keys)} held-out combos={len(held)}", flush=True)

    model = GPT(len(itos), d=args.d, n_layer=args.layers, max_len=max_len)
    print(f"parameters: {model.n_params():,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    def sample_fields():
        # half frequency-weighted (follows the dataset), half uniform over distinct profiles (coverage)
        return rng.choice(train_rows) if rng.random() < 0.5 else rng.choice(train_combos)

    val_rng = random.Random(123)
    val_items = [build_example(vocab, val_rng.choice(held_combos), val_rng, max_len) for _ in range(96)]
    val_seen = [build_example(vocab, val_rng.choice(train_combos), val_rng, max_len) for _ in range(96)]

    def evaluate(items):
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(items), 32):
                x, m = make_batch(vocab, items[i:i + 32])
                tot += float(lm_loss(model, x, m)) * (len(items[i:i + 32]))
        model.train()
        return tot / len(items)

    t0 = time.time()
    warm = 60
    best = 1e9
    model.train()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, m = make_batch(vocab, [build_example(vocab, sample_fields(), rng, max_len) for _ in range(args.batch)])
        loss = lm_loss(model, x, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == 1:
            print(f"step {step:5d} loss {loss.item():.4f} lr {lr:.5f} elapsed {time.time() - t0:6.0f}s", flush=True)
        if step % 150 == 0 or step == args.steps:
            vh, vs = evaluate(val_items), evaluate(val_seen)
            print(f"   val loss  held-out profiles {vh:.4f} | seen profiles {vs:.4f}", flush=True)
            if vh < best:
                best = vh
                torch.save({"cfg": model.cfg, "state": model.state_dict()}, LM_WEIGHTS_PATH)
        if (time.time() - t0) / 60 > args.max_minutes:
            print("time budget reached; stopping", flush=True)
            torch.save({"cfg": model.cfg, "state": model.state_dict()}, LM_WEIGHTS_PATH)
            break

    # ---- faithfulness evaluation with the exact decoding settings used in the app -------------
    ckpt = torch.load(LM_WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(ckpt["state"])
    model.eval()

    def faithfulness(combos_list, n):
        ok, problems = 0, []
        g = torch.Generator().manual_seed(7)
        for f in [combos_list[i % len(combos_list)] for i in range(n)]:
            prompt = [vocab.bos] + vocab.encode(C.control_tokens(f)) + [vocab.sep]
            ids = generate(model, prompt, vocab.eos, temperature=0.5, top_k=5, gen=g)
            text = C.detokenize(vocab.decode(ids))
            p = C.check(f, text)
            if not p:
                ok += 1
            else:
                problems.append(p[0])
        return ok / n, problems[:5]

    n_eval = 120
    r_seen, pr_seen = faithfulness(train_combos, n_eval)
    r_held, pr_held = faithfulness(held_combos, n_eval)
    metrics = {
        "parameters": model.n_params(), "vocab_size": len(itos), "steps": step,
        "best_val_loss_heldout_profiles": best,
        "valid_narratives_seen_profiles": r_seen, "valid_narratives_heldout_profiles": r_held,
        "example_problems_seen": pr_seen, "example_problems_heldout": pr_held,
        "distinct_profiles_total": len(keys), "heldout_profiles": len(held),
        "train_minutes": round((time.time() - t0) / 60, 1),
    }
    (ART_DIR / "lm_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
