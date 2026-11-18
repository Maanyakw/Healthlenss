"use client";
import { useCallback, useEffect, useState } from "react";
import { getJSON, postJSON } from "../../lib/api";
import { useSession } from "../../lib/session";
import { Gate, Notice, TIER_LABEL, pct } from "../../lib/ui";
import HeroGauge from "../../components/HeroGauge";
import WhatIf from "../../components/WhatIf";
import ActionPlan from "../../components/ActionPlan";
import History from "../../components/History";
import { addEntry, loadHistory, saveHistory } from "../../lib/history";

const SMOKE = { "No Info": "Not known", never: "Never smoked", former: "Former smoker", current: "Current smoker", "not current": "Not a current smoker", ever: "Has smoked in the past" };
const fmt = (f, v) => (f === "hypertension" || f === "heart_disease" ? (Number(v) ? "Yes" : "No") : f === "smoking_history" ? SMOKE[v] || v : v);

export default function ResultPage() {
  const { s, ready, update } = useSession();
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const [about, setAbout] = useState(null);
  const [cfLoading, setCfLoading] = useState(false);
  const [cfErr, setCfErr] = useState("");
  const [hist, setHist] = useState([]);

  const run = useCallback(async (patient) => {
    setErr(""); setLoading(true);
    try { update({ prediction: await postJSON("/api/predict", patient) }); }
    catch (e) { setErr(e.message); }
    setLoading(false);
  }, [update]);

  useEffect(() => { if (ready && s.patient && !s.prediction && !loading && !err) run(s.patient); }, [ready, s.patient, s.prediction]); // eslint-disable-line
  useEffect(() => { setHist(loadHistory()); }, []);
  useEffect(() => { if (s.prediction) setHist(addEntry(s.prediction)); }, [s.prediction]); // eslint-disable-line
  useEffect(() => { getJSON("/api/model-info").then(setAbout).catch(() => setAbout(false)); }, []);

  async function findChanges() {
    setCfErr(""); setCfLoading(true);
    try { update({ counterfactuals: await postJSON("/api/counterfactuals", s.patient) }); }
    catch (e) { setCfErr(e.message); }
    setCfLoading(false);
  }

  if (!ready) return null;
  if (!s.patient) return <><div className="page-head"><h1>Your risk result</h1></div><Gate text="Add your lab values first so there is something to estimate." /></>;

  const pred = s.prediction;
  const cf = s.counterfactuals;
  const max = pred ? Math.max(...pred.explanation.map((f) => Math.abs(f.impact_pct_points)), 0.01) : 1;
  const n = pred && pred.narrative;
  const th = pred && pred.thresholds;

  return (
    <>
      <div className="page-head">
        <h1>Your risk result</h1>
        <p className="lede">A pretrained transformer estimates the risk, a small language model written from dataset text explains it, and exact Shapley values show what drove it.</p>
      </div>
      {loading && <Notice kind="info"><span className="spinner" /><span>Calculating your estimate and writing the explanation… the first request can take a minute while the server wakes up.</span></Notice>}
      {err && <><Notice kind="error">{err}</Notice><div className="btn-row"><button className="btn" onClick={() => run(s.patient)}>Try again</button><a className="btn ghost" href="/assess">Edit values</a></div></>}

      {pred && (<>
        {pred.warnings.map((w, i) => <Notice key={i} kind="warn">{w}</Notice>)}
        <section className="card hero-result" style={{ gridTemplateColumns: "minmax(260px, 420px) 1fr" }} aria-labelledby="r-h">
          <HeroGauge p={pred.probability} tier={pred.tier} label={"Diabetes risk: " + pred.tier} ariaLabel={"Estimated risk " + pct(pred.probability) + ", " + TIER_LABEL[pred.tier]}
            chips={[{ icon: "A", label: "HbA1c", value: pred.inputs.hba1c + " %" }, { icon: "G", label: "Glucose", value: pred.inputs.glucose + " mg/dL" }, { icon: "B", label: "BMI", value: String(pred.inputs.bmi) }]} />
          <div>
            <h2 id="r-h">Estimated risk</h2>
            <span className={"tier " + pred.tier}>{TIER_LABEL[pred.tier]}</span>
            <div className="zones" aria-hidden="true"><i /><i /><i /></div>
            <p className="small muted">Low below {(th.moderate * 100).toFixed(1)}%, moderate to {Math.round(th.high * 100)}%, high above. Zones are drawn equal in width for readability.</p>
            <div className="badges">
              <span className="badge">{about ? (about.risk_backend === "tabpfn" ? "Risk model: TabPFN-3.5" : "Risk model: gradient boosting stand-in (TabPFN not loaded on this server)") : "Risk model"}</span>
            </div>
            <p className="small muted">A screening estimate from a statistical model, not a diagnosis. Only a doctor can confirm diabetes with laboratory tests.</p>
          </div>
        </section>

        <section className="card" aria-labelledby="n-h">
          <h2 id="n-h">What this means</h2>
          <div className="badges">
            <span className="badge">{n.source === "local-lm" ? "Written by the locally trained language model" : "Dataset template (model output failed a check)"}</span>
            <span className="badge">Checked: bands, tier and figures match your values</span>
            {n.attempts > 1 && n.source === "local-lm" && <span className="badge">Regenerated {n.attempts - 1}× to pass the check</span>}
          </div>
          <p className="narr">{n.text}</p>
        </section>

        <section className="card bars" aria-labelledby="s-h">
          <h2 id="s-h">What drives the estimate</h2>
          <p className="small muted">Exact Shapley values: each bar is that value&apos;s share of the difference between this estimate and a typical non-diabetic patient. Red raises the risk, green lowers it.</p>
          {pred.explanation.map((f) => (
            <div className="row" key={f.feature}>
              <div>{f.label}<small>{String(fmt(f.feature, f.value))}</small></div>
              <div className="axis" role="img" aria-label={f.label + (f.impact_pct_points >= 0 ? " raises" : " lowers") + " the estimate by " + Math.abs(f.impact_pct_points).toFixed(1) + " percentage points"}>
                <div className={"fill " + (f.impact_pct_points >= 0 ? "up" : "down")} style={{ width: (Math.abs(f.impact_pct_points) / max) * 50 + "%" }} />
              </div>
              <div className="val">{f.impact_pct_points > 0 ? "+" : ""}{f.impact_pct_points.toFixed(1)} pts</div>
            </div>
          ))}
        </section>

        <section className="card" aria-labelledby="w-h">
          <h2 id="w-h">What if my values change?</h2>
          <p className="small muted">Drag the sliders. The risk model recalculates your estimate for the new numbers.</p>
          <WhatIf key={JSON.stringify(s.patient)} patient={s.patient} base={pred.probability} />
        </section>

        <section className="card" aria-labelledby="c-h">
          <h2 id="c-h">What would lower it?</h2>
          <p className="small muted">A counterfactual search: the smallest drops in HbA1c, glucose and BMI that would move the model&apos;s estimate down one tier. This shows how the model behaves; it is not treatment advice.</p>
          {!cf && <button className="btn" onClick={findChanges} disabled={cfLoading}>{cfLoading ? "Searching…" : "Search for changes"}</button>}
          {cfErr && <Notice kind="error">{cfErr}</Notice>}
          {cf && cf.options.length === 0 && <p>{cf.note}</p>}
          {cf && cf.options.map((o, i) => (
            <div key={i} className={"cf" + (o.realistic ? "" : " unreal")}>
              <strong className="t">{o.changes.map((c) => c.label + " " + c.from + " → " + c.to).join(", ")}</strong>
              <div className="small muted">Estimate becomes {pct(o.new_probability)} ({TIER_LABEL[o.new_tier]}).{o.realistic ? "" : " This is a large change and may not be realistic."}</div>
            </div>
          ))}
        </section>

        <section className="card" aria-labelledby="ap-h">
          <h2 id="ap-h">Your action plan</h2>
          <p className="small muted">Steps matched to the values that are above the usual ranges. Tick them off as you go.</p>
          <ActionPlan patient={s.patient} tier={pred.tier} done={s.plan || {}} onToggle={(id) => update({ plan: { ...(s.plan || {}), [id]: !(s.plan || {})[id] } })} />
        </section>

        {hist.length > 0 && (
          <section className="card" aria-labelledby="h-h">
            <h2 id="h-h">Your saved checks</h2>
            <History items={hist} onClear={() => { saveHistory([]); setHist([]); }} />
          </section>
        )}

        <section className="card" aria-labelledby="m-h">
          <h2 id="m-h">About the data and the models</h2>
          {about === null && <p className="muted">Loading…</p>}
          {about === false && <p className="muted">Model details are unavailable right now.</p>}
          {about && <Models about={about} />}
        </section>

        <div className="btn-row"><a className="btn" href="/chat">Ask questions about this result</a><a className="btn ghost" href="/report">Create the doctor report</a><a className="btn ghost" href="/assess">Edit values</a></div>
      </>)}
    </>
  );
}

function Models({ about }) {
  const d = about.dataset, rows = about.comparison || [];
  const f = (x) => (x == null ? "n/a" : (x * 100).toFixed(1) + "%");
  const a = (x) => (x == null ? "n/a" : Number(x).toFixed(3));
  return (
    <>
      <p>{d.source}: {d.rows_used.toLocaleString()} records of female and male patients after removing {d.dropped_duplicate_rows.toLocaleString()} duplicates and {d.dropped_other_gender_rows} records with another gender value.</p>
      <div style={{ overflowX: "auto" }}><table>
        <thead><tr><th>Model</th><th>Test AUC</th><th>95% interval</th><th>Recall at cut-off</th><th>Status</th></tr></thead>
        <tbody>{rows.map((r, i) => (
          <tr key={i}><td>{r.name}{r.role ? <small className="muted"> ({r.role})</small> : null}</td>
            <td className="num">{a(r.test_auc)}</td>
            <td className="num">{r.test_auc_ci95 ? a(r.test_auc_ci95[0]) + " to " + a(r.test_auc_ci95[1]) : "n/a"}</td>
            <td className="num">{f(r.recall)}</td><td>{r.status || ""}</td></tr>))}</tbody></table></div>
      <p className="small muted">In this dataset almost all of the signal comes from HbA1c and blood glucose, so the scores look better than they would for predicting future diabetes in the real world. Models score about the same because the label is nearly determined by those two values.</p>
      {about.risk_backend_note && <Notice kind="warn">{about.risk_backend_note}</Notice>}
    </>
  );
}
