"use client";
import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "../../lib/api";
import { useSession } from "../../lib/session";
import { RANGES, SMOKING, EXAMPLES, Notice } from "../../lib/ui";

const KEYS = ["gender", "age", "hba1c", "glucose", "bmi", "hypertension", "heart_disease", "smoking_history"];
const NUMERIC = ["age", "hba1c", "glucose", "bmi"];
const LABELS = { gender: "Sex", age: "Age", hba1c: "HbA1c", glucose: "Blood glucose", bmi: "BMI", hypertension: "Hypertension", heart_disease: "Heart disease", smoking_history: "Smoking history" };
const UNITS = { age: "years", hba1c: "%", glucose: "mg/dL", bmi: "kg/m²" };
const SAMPLES = [["Sample PDF", "sample_01_male.pdf", "application/pdf"], ["Sample scan (PNG)", "sample_02_female.png", "image/png"], ["Sample phone photo (JPEG)", "sample_03_female.jpg", "image/jpeg"]];
const EMPTY = Object.fromEntries(KEYS.map((k) => [k, ""]));

export default function ReadPage() {
  const router = useRouter();
  const { s, ready, setPatient } = useSession();
  const [vals, setVals] = useState(EMPTY);
  const [prov, setProv] = useState({});
  const [ext, setExt] = useState(null);
  const [page, setPage] = useState(0);
  const [status, setStatus] = useState(null);
  const [errors, setErrors] = useState({});
  const [synthetic, setSynthetic] = useState(false);
  const [over, setOver] = useState(false);
  const [active, setActive] = useState(null);
  const fileRef = useRef(null);

  useEffect(() => {
    if (ready && s.patient) { setVals(Object.fromEntries(KEYS.map((k) => [k, String(s.patient[k])]))); setProv(s.provenance || {}); }
  }, [ready]); // eslint-disable-line react-hooks/exhaustive-deps

  const setVal = (k, v) => {
    setVals((o) => ({ ...o, [k]: v }));
    const f = ext && ext.fields[k];
    setProv((p) => ({ ...p, [k]: f && String(f.value) === String(v).trim() ? "extracted" : "manual" }));
  };

  async function upload(file) {
    if (!/\.(pdf|png|jpe?g)$/i.test(file.name)) return setStatus({ kind: "error", msg: "Unsupported file type. Choose a PDF, PNG or JPEG file." });
    if (file.size > 15 * 1024 * 1024) return setStatus({ kind: "error", msg: "That file is larger than 15 MB. Choose a smaller file." });
    setStatus({ kind: "info", busy: true, msg: "Reading " + file.name + "… this can take up to a minute on a CPU." });
    const fd = new FormData(); fd.append("file", file);
    try {
      const res = await (await api("/api/extract", { method: "POST", body: fd })).json();
      setExt(res); setPage(0);
      const nv = {}; const np = {};
      KEYS.forEach((k) => { const f = res.fields[k]; nv[k] = f ? String(f.value) : ""; np[k] = f ? "extracted" : "missing"; });
      setVals(nv); setProv(np); setErrors({});
      const found = KEYS.filter((k) => res.fields[k]).length;
      setStatus(found
        ? { kind: "info", msg: "Read " + res.lines.length + " lines in " + (res.elapsed_ms / 1000).toFixed(1) + " s and found " + found + " of " + KEYS.length + " values. Check each value against the report before continuing." }
        : { kind: "warn", msg: "The report was read, but none of the needed values were found. Enter them below." });
    } catch (e) { setStatus({ kind: "error", msg: e.message }); }
  }

  async function sample([, name, type]) {
    try { const blob = await (await fetch("/samples/" + name)).blob(); upload(new File([blob], name, { type })); }
    catch (e) { setStatus({ kind: "error", msg: "Could not load the sample file." }); }
  }

  function example(kind) {
    setExt(null); setVals(EXAMPLES[kind]); setProv(Object.fromEntries(KEYS.map((k) => [k, "manual"]))); setErrors({}); setStatus(null);
  }

  function submit(e) {
    e.preventDefault();
    const patient = {}, errs = {};
    KEYS.forEach((k) => {
      const raw = String(vals[k]).trim();
      if (k === "gender" || k === "smoking_history") { if (!raw) errs[k] = "Choose an option."; else patient[k] = raw; }
      else if (NUMERIC.includes(k)) {
        const v = Number(raw.replace(",", ".")); const [lo, hi, unit] = RANGES[k];
        if (!raw || !isFinite(v)) errs[k] = "Enter a number.";
        else if (v < lo || v > hi) errs[k] = "Enter a value between " + lo + " and " + hi + (unit ? " " + unit : "") + ".";
        else patient[k] = v;
      } else { if (raw === "") errs[k] = "Choose Yes or No."; else patient[k] = Number(raw); }
    });
    setErrors(errs);
    if (Object.keys(errs).length) { const first = KEYS.find((k) => errs[k]); document.getElementById("f-" + first)?.focus(); return; }
    setPatient(patient, prov);
    router.push("/result");
  }

  const chip = (k) => {
    const f = ext && ext.fields[k]; const p = prov[k];
    if (p === "extracted" && f) return f.confidence >= 0.95 ? <span className="chip ok">Read by the model, verify</span> : <span className="chip check">Low confidence, check carefully</span>;
    if (p === "manual") return <span className="chip manual">Entered manually</span>;
    if (p === "missing") return <span className="chip missing">Not found on report, enter it</span>;
    return null;
  };

  const boxes = [];
  if (ext) KEYS.forEach((k) => {
    const f = ext.fields[k]; if (!f || f.line_index == null) return;
    const ln = ext.lines[f.line_index]; if (!ln || ln.page !== page) return;
    const pad = 0.004;
    boxes.push(<div key={k} className={"box" + (active === k ? " active" : "")} style={{ left: Math.max(0, ln.x0 - pad) * 100 + "%", top: Math.max(0, ln.y0 - pad) * 100 + "%", width: (ln.x1 - ln.x0 + 2 * pad) * 100 + "%", height: (ln.y1 - ln.y0 + 2 * pad) * 100 + "%" }} />);
  });

  const alts = ext ? ext.glucose_candidates || [] : [];

  const input = (k) => {
    const common = { id: "f-" + k, value: vals[k], onChange: (e) => setVal(k, e.target.value), onFocus: () => setActive(k), onBlur: () => setActive(null), "aria-describedby": "e-" + k };
    if (k === "gender") return <select {...common}><option value="">Choose…</option><option>Female</option><option>Male</option></select>;
    if (k === "smoking_history") return <select {...common}><option value="">Choose…</option>{SMOKING.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>;
    if (k === "hypertension" || k === "heart_disease") return <select {...common}><option value="">Choose…</option><option value="0">No</option><option value="1">Yes</option></select>;
    return <input {...common} inputMode="decimal" autoComplete="off" />;
  };

  return (
    <>
      <div className="page-head">
        <h1>Read your lab report</h1>
        <p className="lede">A Vision Transformer trained for this project reads the report; a rule-based parser then picks out the values. You confirm every value before anything is calculated.</p>
      </div>

      <section className="card" aria-labelledby="up-h">
        <h2 id="up-h">Read values from a lab report</h2>
        <label className="consent"><input type="checkbox" checked={synthetic} onChange={(e) => setSynthetic(e.target.checked)} />
          This is a synthetic sample, not a real patient&apos;s report</label>
        <div className={"drop" + (over ? " over" : "")}
          onDragEnter={(e) => { e.preventDefault(); setOver(true); }} onDragOver={(e) => e.preventDefault()} onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); if (synthetic && e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); }}>
          <p><strong>Drop a PDF, PNG or JPEG here</strong></p>
          <input ref={fileRef} type="file" accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg" hidden onChange={(e) => e.target.files[0] && upload(e.target.files[0])} />
          <button type="button" className="btn" disabled={!synthetic} onClick={() => fileRef.current.click()}>Choose file</button>
          <div className="samples"><span className="small muted">Or try one:</span>
            {SAMPLES.map((x) => <button key={x[1]} type="button" className="btn ghost sm" onClick={() => sample(x)}>{x[0]}</button>)}</div>
          {!synthetic && <p className="small muted">Tick the box above to enable uploads. Sample files can always be used.</p>}
        </div>
        {status && <Notice kind={status.kind}>{status.busy && <span className="spinner" />}<span>{status.msg}</span></Notice>}
      </section>

      <div className={ext ? "grid2" : ""}>
        {ext && (
          <section className="card" aria-label="Uploaded report">
            <h2>Your report</h2>
            {ext.pages.length > 1 && <div className="tabs">{ext.pages.map((_, i) => <button key={i} type="button" aria-pressed={i === page} onClick={() => setPage(i)}>Page {i + 1}</button>)}</div>}
            <div className="viewer">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={ext.pages[page].image} alt={"Uploaded lab report, page " + (page + 1)} />{boxes}
            </div>
            <details className="raw"><summary className="small">Text the model read</summary>
              <pre>{ext.lines.map((l) => "p" + (l.page + 1) + "  " + l.text + "   (" + Math.round(l.conf * 100) + "%)").join("\n")}</pre></details>
          </section>
        )}
        <section className="card" aria-labelledby="f-h">
          <h2 id="f-h">Your values</h2>
          <div className="quick">Or fill an example:
            <button type="button" className="btn ghost sm" onClick={() => example("low")}>Lower-risk example</button>
            <button type="button" className="btn ghost sm" onClick={() => example("high")}>Higher-risk example</button>
            <button type="button" className="btn ghost sm" onClick={() => { setExt(null); setVals(EMPTY); setProv({}); setErrors({}); setStatus(null); }}>Clear</button>
          </div>
          <form className="form" onSubmit={submit} noValidate>
            {KEYS.map((k) => (
              <div key={k} className={"field" + (errors[k] ? " invalid" : "")}>
                <label htmlFor={"f-" + k}>{LABELS[k]}{UNITS[k] ? " (" + UNITS[k] + ")" : ""} {chip(k)}</label>
                {input(k)}
                {k === "glucose" && alts.length > 1 && (
                  <select aria-label="Other glucose values found" style={{ marginTop: 6 }} defaultValue="" onChange={(e) => e.target.value && setVal("glucose", e.target.value)}>
                    <option value="">Several glucose values found, choose one</option>
                    {alts.map((c, i) => <option key={i} value={String(c.value)}>{c.value} mg/dL, {c.kind}</option>)}
                  </select>
                )}
                {ext && ext.fields[k] && ext.fields[k].note && <div className="hint">{ext.fields[k].note}</div>}
                <div className="err" id={"e-" + k}>{errors[k] || ""}</div>
              </div>
            ))}
            <div style={{ gridColumn: "1 / -1" }}><button className="btn" type="submit">Get my screening result</button></div>
          </form>
        </section>
      </div>
    </>
  );
}
