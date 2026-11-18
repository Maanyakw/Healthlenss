import React from "react";

export const RANGES = { age: [1, 110, "years"], hba1c: [3, 18, "%"], glucose: [30, 700, "mg/dL"], bmi: [10, 80, ""] };
export const SMOKING = [["never", "Never smoked"], ["former", "Former smoker"], ["current", "Current smoker"], ["not current", "Not current"], ["ever", "Ever smoked"], ["No Info", "No information"]];
export const TIER_LABEL = { low: "Low risk", moderate: "Moderate risk", high: "High risk" };
export const EXAMPLES = {
  low: { gender: "Female", age: "32", hypertension: "0", heart_disease: "0", smoking_history: "never", bmi: "23.4", hba1c: "5.2", glucose: "92" },
  high: { gender: "Male", age: "58", hypertension: "1", heart_disease: "0", smoking_history: "former", bmi: "32.1", hba1c: "7.1", glucose: "186" },
};

export function pct(p) {
  const v = p * 100;
  if (v < 0.1) return "<0.1%";
  if (v > 99.9) return ">99.9%";
  return v.toFixed(1) + "%";
}

// Safe rendering of **bold** and "- " lists from assistant text (React escapes everything else).
export function RichText({ text }) {
  const blocks = String(text || "").split(/\n{2,}/);
  const inline = (t) => t.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    /^\*\*[^*]+\*\*$/.test(part) ? <strong key={i}>{part.slice(2, -2)}</strong> : <React.Fragment key={i}>{part}</React.Fragment>);
  return blocks.map((b, i) => {
    const lines = b.split("\n");
    if (lines.every((l) => /^\s*[-*]\s+/.test(l))) return <ul key={i}>{lines.map((l, j) => <li key={j}>{inline(l.replace(/^\s*[-*]\s+/, ""))}</li>)}</ul>;
    return <p key={i}>{lines.map((l, j) => <React.Fragment key={j}>{j > 0 && <br />}{inline(l)}</React.Fragment>)}</p>;
  });
}

export function Notice({ kind = "info", children }) {
  return <div className={"notice " + kind} role={kind === "error" ? "alert" : "status"}>{children}</div>;
}

export function Gate({ text }) {
  return (
    <div className="card gate">
      <h3>Start with your values</h3>
      <p className="muted">{text}</p>
      <a className="btn" href="/assess">Go to step 1</a>
    </div>
  );
}
