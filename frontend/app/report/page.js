"use client";
import { useState } from "react";
import { api, postJSON } from "../../lib/api";
import { useSession } from "../../lib/session";
import { Gate, Notice } from "../../lib/ui";

const yn = (f, v) => (f === "hypertension" || f === "heart_disease" ? (Number(v) ? "Yes" : "No") : String(v));
const List = ({ items }) => <ul>{(items || []).map((t, i) => <li key={i}>{t}</li>)}</ul>;

export default function ReportPage() {
  const { s, ready, update } = useSession();
  const [ref, setRef] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [pdfBusy, setPdfBusy] = useState(false);

  if (!ready) return null;
  if (!s.patient) return <><div className="page-head"><h1>Doctor report</h1></div><Gate text="Add your lab values first so there is something to report." /></>;

  async function generate() {
    setBusy(true); setErr("");
    try {
      const rep = await postJSON("/api/report", {
        patient: s.patient, provenance: s.provenance || {}, patient_ref: ref,
        narrative: (s.prediction && s.prediction.narrative && s.prediction.narrative.text) || "",
      });
      update({ report: rep });
    } catch (e) { setErr(e.message); }
    setBusy(false);
  }

  async function pdf(rep) {
    setPdfBusy(true); setErr("");
    try {
      const res = await api("/api/report/pdf", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ report: rep }) });
      const url = URL.createObjectURL(await res.blob());
      const a = document.createElement("a"); a.href = url; a.download = "healthlens-report.pdf"; document.body.append(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    } catch (e) { setErr(e.message); }
    setPdfBusy(false);
  }

  const rep = s.report;
  const f = rep && rep.facts, sec = rep && rep.sections;

  return (
    <>
      <div className="page-head no-print">
        <h1>Doctor report</h1>
        <p className="lede">A generative-AI model (Gemini or Claude, if the server has a key) writes the clinical wording from the verified numbers. Without a key, a template version is used. Download it as a PDF.</p>
      </div>
      <section className="card no-print">
        <div className="field" style={{ maxWidth: 420 }}>
          <label htmlFor="ref">Patient reference (optional)</label>
          <input id="ref" maxLength={60} autoComplete="off" value={ref} onChange={(e) => setRef(e.target.value)} placeholder="Initials or clinic number" />
          <div className="hint">Printed on the report. The app never asks for your name or other identifying details.</div>
        </div>
        <div className="btn-row"><button className="btn" onClick={generate} disabled={busy}>{rep ? "Write it again" : "Generate report"}</button></div>
      </section>
      {busy && <Notice kind="info"><span className="spinner" /><span>Writing the report… this can take 10 to 30 seconds.</span></Notice>}
      {err && <Notice kind="error">{err}</Notice>}

      {rep && (<>
        {rep.warnings.map((w, i) => <div className="no-print" key={i}><Notice kind="warn">{w}</Notice></div>)}
        <div className="btn-row no-print" style={{ marginBottom: 14 }}>
          <button className="btn" onClick={() => pdf(rep)} disabled={pdfBusy}>{pdfBusy ? "Preparing…" : "Download PDF"}</button>
          <button className="btn ghost" onClick={() => window.print()}>Print</button>
        </div>
        <article className="doc" aria-label="Doctor report">
          <h1>Diabetes risk screening report</h1>
          <p className="doc-meta">Generated {rep.meta.generated_at}. Patient reference: {rep.meta.patient_ref || "not provided"}. {rep.mode === "genai" ? "Written by generative AI (" + rep.provider + ", " + rep.model + ")." : "Assembled from templates (no generative-AI service used)."}</p>
          <h2>Clinical summary</h2><p>{sec.clinical_summary}</p>
          <h2>Patient parameters</h2>
          <table><thead><tr><th>Parameter</th><th>Value</th><th>Interpretation</th><th>Source</th></tr></thead>
            <tbody>{f.rows.map((r, i) => <tr key={i}><th scope="row">{r.name}</th><td>{r.value}</td><td>{r.interpretation || "-"}</td><td>{r.source}</td></tr>)}</tbody></table>
          <h2>Screening-model result</h2>
          <p>Estimated probability <strong>{f.probability_text}</strong>, tier <strong>{f.tier.toUpperCase()}</strong> (moderate from {(f.thresholds.moderate * 100).toFixed(1)}%, high from {Math.round(f.thresholds.high * 100)}%).</p>
          <table><thead><tr><th>Factor</th><th>Value</th><th>Effect on estimated risk (Shapley)</th></tr></thead>
            <tbody>{f.explanation.map((e, i) => <tr key={i}><td>{e.label}</td><td>{yn(e.feature, e.value)}</td><td>{e.impact_pct_points > 0 ? "+" : ""}{e.impact_pct_points.toFixed(1)} percentage points</td></tr>)}</tbody></table>
          <h2>Narrative from the locally trained language model</h2><p>{f.narrative}</p>
          <h2>Key findings</h2><List items={sec.key_findings} />
          <h2>Risk drivers</h2><List items={sec.risk_drivers} />
          <h2>Suggested follow-up</h2><List items={sec.recommended_follow_up} />
          <h2>Lifestyle and counselling points</h2><List items={sec.lifestyle_and_counselling} />
          <h2>Limitations</h2><List items={sec.limitations} />
          <h2>Reference criteria</h2><p className="small">{rep.ada_reference}</p>
          <p className="small muted">Educational screening output, not a diagnosis. Values read from documents by the vision model must be checked against the source.</p>
        </article>
      </>)}
    </>
  );
}
