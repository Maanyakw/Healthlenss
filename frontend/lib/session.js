"use client";
import { useCallback, useEffect, useState } from "react";

const KEY = "hl3_session_v1";
const read = () => { try { return JSON.parse(sessionStorage.getItem(KEY)) || {}; } catch (e) { return {}; } };

// Patient state lives only in this browser tab.
export function useSession() {
  const [s, setS] = useState({});
  const [ready, setReady] = useState(false);
  useEffect(() => { setS(read()); setReady(true); const on = () => setS(read()); window.addEventListener("hl3-session", on); return () => window.removeEventListener("hl3-session", on); }, []);
  const update = useCallback((patch) => {
    const next = { ...read(), ...patch };
    try { sessionStorage.setItem(KEY, JSON.stringify(next)); } catch (e) { /* storage full or blocked */ }
    setS(next); window.dispatchEvent(new Event("hl3-session"));
  }, []);
  const setPatient = useCallback((patient, provenance) => {
    const cur = read();
    const same = cur.patient && JSON.stringify(cur.patient) === JSON.stringify(patient);
    update(same ? { patient, provenance } : { patient, provenance, prediction: null, counterfactuals: null, chat: [], report: null, plan: {} });
  }, [update]);
  const clear = useCallback(() => { try { sessionStorage.removeItem(KEY); } catch (e) {} setS({}); window.dispatchEvent(new Event("hl3-session")); }, []);
  return { s, ready, update, setPatient, clear };
}
