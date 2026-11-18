"""Measure ViT extraction on NEW synthetic reports (different seed from the demo files; some fonts held out).
    python3 tests/eval_extraction.py --n 30 --seed 777
"""
import argparse, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "samples"))
from make_samples import render_set
from backend.vit_ocr.extract import LabReportReader
from backend.vit_ocr.parse import parse_fields

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30); ap.add_argument("--seed", type=int, default=777)
a = ap.parse_args()
ex = LabReportReader()
with tempfile.TemporaryDirectory() as d:
    truth = render_set(Path(d), a.n, a.seed)
    hits = {k: 0 for k in ("age", "gender", "hba1c", "glucose", "bmi")}
    allok = 0; by_fmt = {}; misses = []
    for t in truth:
        res = ex.read((Path(d) / t["file"]).read_bytes())
        f = parse_fields(res["lines"])["fields"]
        ok = {}
        for k in hits:
            v = f.get(k); v = v["value"] if isinstance(v, dict) else v
            if k == "bmi": ok[k] = v is not None and abs(float(v) - t[k]) <= 0.15
            elif k == "gender": ok[k] = v == t[k]
            else: ok[k] = v is not None and abs(float(v) - t[k]) < 1e-6
            hits[k] += ok[k]
        allok += all(ok.values()); fmt = t["file"].rsplit(".", 1)[1]
        s = by_fmt.setdefault(fmt, [0, 0]); s[0] += all(ok.values()); s[1] += 1
        if not all(ok.values()): misses.append((t["file"], t["font"], {k: (t[k], (f[k]["value"] if isinstance(f[k], dict) else f[k])) for k in ok if not ok[k]}))
n = len(truth)
print(f"documents: {n}; all five fields correct: {allok}/{n} ({100*allok/n:.0f}%)")
for k, v in hits.items(): print(f"  {k:8s} {v}/{n} ({100*v/n:.0f}%)")
for fmt, (g, m) in by_fmt.items(): print(f"  {fmt}: {g}/{m} fully correct")
for m in misses[:12]: print("  miss", m)
