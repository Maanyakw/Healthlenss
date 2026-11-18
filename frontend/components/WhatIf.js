"use client";
import { useEffect, useRef, useState } from "react";
import { postJSON } from "../lib/api";
import { TIER_LABEL, pct } from "../lib/ui";

const DEFS = [
  { k: "hba1c", label: "HbA1c", unit: "%", min: 4, max: 12, step: 0.1, d: 1 },
  { k: "glucose", label: "Blood glucose", unit: "mg/dL", min: 60, max: 300, step: 1, d: 0 },
  { k: "bmi", label: "BMI", unit: "", min: 16, max: 45, step: 0.1, d: 1 },
];
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

export default function WhatIf({ patient, base }) {
  const start = () => Object.fromEntries(DEFS.map((d) => [d.k, clamp(Number(patient[d.k]), d.min, d.max)]));
  const [vals, setVals] = useState(start);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const seq = useRef(0);
  const changed = DEFS.some((d) => vals[d.k] !== clamp(Number(patient[d.k]), d.min, d.max));

  useEffect(() => {
    if (!changed) { setRes(null); setErr(""); return; }
    const id = ++seq.current; setBusy(true);
    const h = setTimeout(async () => {
      try { const r = await postJSON("/api/whatif", { ...patient, ...vals }); if (id === seq.current) { setRes(r); setErr(""); } }
      catch (e) { if (id === seq.current) setErr(e.message); }
      if (id === seq.current) setBusy(false);
    }, 350);
    return () => clearTimeout(h);
  }, [vals]); // eslint-disable-line react-hooks/exhaustive-deps

  const delta = res ? (res.probability - base) * 100 : 0;
  return (
    <>
      <div className="sliders">
        {DEFS.map((d) => (
          <div className="sl" key={d.k}>
            <label htmlFor={"wi-" + d.k}>{d.label}<span>{vals[d.k].toFixed(d.d)} {d.unit}</span></label>
            <input id={"wi-" + d.k} type="range" min={d.min} max={d.max} step={d.step} value={vals[d.k]} onChange={(e) => setVals({ ...vals, [d.k]: Number(e.target.value) })} />
            <div className="ticks"><span>{d.min}</span><span>your value: {Number(patient[d.k]).toFixed(d.d)}</span><span>{d.max}</span></div>
          </div>
        ))}
      </div>
      <div className="wi-out" aria-live="polite">
        {!changed && <span className="muted">Move a slider to see how the estimate would change.</span>}
        {changed && res && <>
          <span className="big">{pct(res.probability)}</span>
          <span className={"tier " + res.tier}>{TIER_LABEL[res.tier]}</span>
          <span className={"delta " + (delta <= 0 ? "down" : "up")}>{Math.abs(delta) < 0.05 ? "no change" : (delta > 0 ? "+" : "") + delta.toFixed(1) + " points"}</span>
          {busy && <span className="spinner" />}
        </>}
        {changed && !res && !err && <span className="spinner" />}
        {changed && res && res.warnings.length > 0 && <span className="small muted" style={{ flexBasis: "100%" }}>{res.warnings[0]}</span>}
        {err && <span className="small" style={{ color: "var(--high-ink)" }}>{err}</span>}
      </div>
      <div className="btn-row"><button type="button" className="btn ghost sm" onClick={() => setVals(start())}>Reset sliders</button></div>
      <p className="small muted">A what-if shows how the model reacts to different numbers. It does not predict what will happen to you or promise that a change is achievable.</p>
    </>
  );
}
