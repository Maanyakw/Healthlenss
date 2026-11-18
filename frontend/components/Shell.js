"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSession } from "../lib/session";
import ThemePicker from "./ThemePicker";

const STEPS = [
  { href: "/assess", title: "Read lab report", sub: "Vision Transformer" },
  { href: "/result", title: "Risk result", sub: "TabPFN + what-if + plan" },
  { href: "/chat", title: "Ask HealthLens", sub: "Chat assistant" },
  { href: "/report", title: "Doctor report", sub: "Generative AI" },
];

export default function Shell({ children }) {
  const path = usePathname();
  const { s, clear } = useSession();
  if (path === "/") return <>{children}</>;      // landing page has its own header
  const done = [!!s.patient, !!s.prediction, !!(s.chat && s.chat.length), !!s.report];
  return (
    <div className="shell">
      <aside className="side">
        <Link href="/" className="brand" aria-label="HealthLens home">
          <span className="logo" aria-hidden="true">H</span>
          <span><strong>HealthLens</strong><span>AI Health Companion</span></span>
        </Link>
        <nav className="nav" aria-label="Steps">
          {STEPS.map((st, i) => (
            <Link key={st.href} href={st.href} className={done[i] && path !== st.href ? "done" : ""} aria-current={path === st.href ? "page" : undefined}>
              <span className="n" aria-hidden="true">{done[i] && path !== st.href ? "✓" : i + 1}</span>
              <span>{st.title}<span className="sub">{st.sub}</span></span>
            </Link>
          ))}
        </nav>
        <ThemePicker />
        <div className="foot">
          Screening aid, not a diagnosis. Your values stay in this browser tab and are sent only to the HealthLens server to compute results.{" "}
          <button type="button" onClick={() => { clear(); window.location.href = "/assess"; }}>Clear session</button>
        </div>
      </aside>
      <main className="main" id="main">{children}</main>
    </div>
  );
}
