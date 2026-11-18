import Link from "next/link";
import HeroGauge from "../components/HeroGauge";
import ThemePicker from "../components/ThemePicker";

const FEATURES = [
  ["◔", "Risk assessment", "TabPFN-3.5, a pretrained transformer, estimates diabetes risk. XGBoost and logistic regression are shown as comparisons."],
  ["✦", "Explainable AI", "Exact Shapley values show how much HbA1c, glucose, BMI and age each moved your estimate."],
  ["▤", "Report analysis", "A Vision Transformer trained for this project reads a synthetic sample lab report; you check every value."],
  ["⇆", "What-if sliders", "Move HbA1c, glucose or BMI and watch the estimate change live, plus a search for the smallest change that lowers your tier."],
  ["✎", "Ask HealthLens", "Chat in plain English, including what-if questions that are calculated by the risk model."],
  ["☑", "Action plan and report", "A checklist of steps for your high values, and a doctor report you can download as a PDF."],
];
const STEPS = [
  ["Add health data", "Enter values, or pre-fill them from a synthetic sample lab report."],
  ["AI analysis", "The risk model scores the values and a small language model writes the explanation."],
  ["Understand results", "See your risk, what drove it, and what a change would do."],
  ["Take action", "Follow the checklist, ask questions, and download the doctor report."],
];

export default function Landing() {
  return (
    <div className="land">
      <header className="topbar">
        <Link href="/" className="brand" aria-label="HealthLens home"><span className="logo" aria-hidden="true">H</span><strong style={{ fontSize: "1.2rem" }}>HealthLens</strong></Link>
        <nav aria-label="Sections"><a href="#features">Features</a><a href="#how-it-works">How it works</a><a href="#about">About</a></nav>
        <div className="right"><ThemePicker /><Link className="btn glow" href="/assess">Get started</Link></div>
      </header>
      <main className="wrap">
        <section className="hero">
          <div>
            <span className="pill">✦ AI-powered health insights</span>
            <h1>Understand your health with clarity.</h1>
            <p className="lede">HealthLens helps you understand health data, identify diabetes risk, explain AI predictions, and test what would change your result.</p>
            <div className="btn-row">
              <Link className="btn glow" href="/assess">Start health assessment →</Link>
              <Link className="btn ghost" href="/result">View my result</Link>
            </div>
            <p className="note">ⓘ HealthLens is a health screening and awareness tool. It does not replace professional medical advice or diagnosis.</p>
          </div>
          <HeroGauge p={0.24} tier="low" label="Diabetes risk: low" caption="Sample preview" ariaLabel="Sample diabetes risk: 24 percent, low"
            chips={[{ icon: "G", label: "Glucose", value: "118 mg/dL" }, { icon: "B", label: "BMI", value: "24.8" }, { icon: "A", label: "HbA1c", value: "5.4 %" }]} />
        </section>
        <section className="sect" id="features">
          <h2>Everything you need to understand your health data</h2>
          <p className="lede">A simple interface backed by explainable machine learning.</p>
          <div className="cards">{FEATURES.map(([ic, t, d]) => <article className="fcard" key={t}><div className="ic" aria-hidden="true">{ic}</div><h3>{t}</h3><p>{d}</p></article>)}</div>
        </section>
        <section className="sect" id="how-it-works">
          <h2>How HealthLens works</h2>
          <ol className="steps">{STEPS.map(([t, d], i) => <li key={t}><div className="n">{i + 1}</div><h3>{t}</h3><p className="muted" style={{ margin: 0 }}>{d}</p></li>)}</ol>
        </section>
        <section className="sect" id="about">
          <h2>Making AI health insights easier to understand</h2>
          <p className="lede">HealthLens combines a pretrained tabular transformer, explainable AI, sample-report reading and a chat assistant in one platform. It is an academic AI in healthcare project trained on a public Kaggle dataset; its predictions are not a medical diagnosis.</p>
        </section>
        <section className="cta"><h2>Start understanding your health</h2><p>Explore your health data with a simple, explainable AI experience.</p><Link className="btn" href="/assess">Get started →</Link></section>
        <footer className="foot2"><strong>HealthLens</strong><span>© 2026 HealthLens. Built as an academic project.</span></footer>
      </main>
    </div>
  );
}
