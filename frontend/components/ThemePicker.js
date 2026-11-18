"use client";
import { useEffect, useRef, useState } from "react";

export const THEMES = [
  { id: "clinic", name: "Clinic", kind: "Light", sw: "#0e7a6d" },
  { id: "midnight", name: "Midnight", kind: "Dark", sw: "#3ad6c5" },
  { id: "meadow", name: "Meadow", kind: "Light", sw: "#3b8a3a" },
];

export default function ThemePicker() {
  const [theme, setTheme] = useState("clinic");
  const [open, setOpen] = useState(false);
  const box = useRef(null);
  useEffect(() => { setTheme(document.documentElement.dataset.theme || "clinic"); }, []);
  useEffect(() => {
    if (!open) return;
    const off = (e) => { if (box.current && !box.current.contains(e.target)) setOpen(false); };
    const esc = (e) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", off); document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("mousedown", off); document.removeEventListener("keydown", esc); };
  }, [open]);
  const pick = (id) => {
    setTheme(id); setOpen(false); document.documentElement.dataset.theme = id;
    try { localStorage.setItem("hl3_theme", id); } catch (e) { /* private mode */ }
  };
  const cur = THEMES.find((t) => t.id === theme) || THEMES[0];
  return (
    <div className="theme" ref={box}>
      <button type="button" aria-haspopup="true" aria-expanded={open} onClick={() => setOpen(!open)}><span className="dot" aria-hidden="true" />{cur.name}</button>
      {open && (
        <ul role="group" aria-label="Choose a theme">
          {THEMES.map((t) => (
            <li key={t.id}><button type="button" aria-pressed={t.id === theme} onClick={() => pick(t.id)}>
              <span className="sw" style={{ background: t.sw }} />{t.name}<small>{t.kind}</small></button></li>
          ))}
        </ul>
      )}
    </div>
  );
}
