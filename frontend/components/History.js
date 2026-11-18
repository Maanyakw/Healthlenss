"use client";
import { TIER_LABEL, pct } from "../lib/ui";

export default function History({ items, onClear }) {
  if (items.length < 1) return null;
  const W = 600, H = 120, pad = 14;
  const pts = items.map((e, i) => [items.length === 1 ? W / 2 : pad + (i * (W - 2 * pad)) / (items.length - 1), H - pad - Math.min(1, e.p) * (H - 2 * pad)]);
  return (
    <div className="hist">
      {items.length > 1 && (
        <svg viewBox={"0 0 " + W + " " + H} role="img" aria-label="Estimated risk over your saved checks" preserveAspectRatio="none">
          <polyline points={pts.map((p) => p.join(",")).join(" ")} fill="none" stroke="var(--teal)" strokeWidth="3" vectorEffect="non-scaling-stroke" />
          {pts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r="4" fill="var(--teal)" />)}
        </svg>
      )}
      <div style={{ overflowX: "auto" }}><table>
        <thead><tr><th>Date</th><th>Risk</th><th>HbA1c</th><th>Glucose</th><th>BMI</th></tr></thead>
        <tbody>{[...items].reverse().map((e, i) => (
          <tr key={i}><td>{new Date(e.t).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })}</td>
            <td>{pct(e.p)} <span className={"tier " + e.tier} style={{ fontSize: ".7rem", padding: "1px 8px" }}>{TIER_LABEL[e.tier]}</span></td>
            <td className="num">{e.hba1c}</td><td className="num">{e.glucose}</td><td className="num">{e.bmi}</td></tr>))}</tbody></table></div>
      <div className="btn-row"><button type="button" className="btn ghost sm" onClick={onClear}>Clear saved history</button></div>
      <p className="small muted">Saved only in this browser on this device. Checks use whatever values you entered, so they are only as good as those values.</p>
    </div>
  );
}
