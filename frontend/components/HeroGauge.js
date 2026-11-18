import { pct } from "../lib/ui";

// 270-degree arc gauge with orbit rings and floating value chips (same look as the HealthLens landing page).
export default function HeroGauge({ p, tier, label, chips = [], caption, ariaLabel }) {
  const R = 46, C = 2 * Math.PI * R, sweep = 0.75;
  const shown = Math.max(0.015, Math.min(1, p));
  const color = { low: "var(--low)", moderate: "var(--moderate)", high: "var(--high)" }[tier] || "var(--teal)";
  const cls = ["a", "b", "c"];
  return (
    <div className="orbit" role="img" aria-label={ariaLabel || "Estimated risk " + pct(p)}>
      <div className="r1" /><div className="r2" /><div className="disc" />
      <svg viewBox="0 0 100 100" aria-hidden="true" style={{ transform: "rotate(135deg)" }}>
        <circle cx="50" cy="50" r={R} fill="none" stroke="var(--track)" strokeWidth="6" strokeLinecap="round" strokeDasharray={C} strokeDashoffset={C * (1 - sweep)} />
        <circle cx="50" cy="50" r={R} fill="none" stroke={color} strokeWidth="6" strokeLinecap="round" strokeDasharray={C} strokeDashoffset={C * (1 - sweep * shown)} />
      </svg>
      <div className="disc" style={{ background: "none", boxShadow: "none" }}><div className="mid"><strong>{pct(p)}</strong><span>{label}</span></div></div>
      {chips.slice(0, 3).map((c, i) => (
        <div key={i} className={"chip2 " + cls[i]}><i aria-hidden="true">{c.icon}</i><span><small>{c.label}</small><b>{c.value}</b></span></div>
      ))}
      {caption && <span className="cap">{caption}</span>}
    </div>
  );
}
