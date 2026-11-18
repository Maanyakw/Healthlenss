"use client";
import { buildPlan } from "../lib/plan";

export default function ActionPlan({ patient, tier, done = {}, onToggle }) {
  const steps = buildPlan(patient, tier);
  const n = steps.filter((s) => done[s.id]).length;
  return (
    <>
      <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={steps.length} aria-valuenow={n} aria-label="Steps completed"><i style={{ width: (n / steps.length) * 100 + "%" }} /></div>
      <p className="small muted">{n} of {steps.length} steps ticked off</p>
      <ul className="plan">
        {steps.map((s) => (
          <li key={s.id} className={done[s.id] ? "done" : ""}>
            <input id={"p-" + s.id} type="checkbox" checked={!!done[s.id]} onChange={() => onToggle(s.id)} />
            <label htmlFor={"p-" + s.id}><div className="t"><span className="tag">{s.tag}</span>{s.title}</div><div className="w">{s.why}</div></label>
          </li>
        ))}
      </ul>
      <p className="small muted">General educational steps matched to your values, not personal medical advice. Ask your doctor before changing your diet, exercise or medicines.</p>
    </>
  );
}
