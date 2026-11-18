"use client";
const KEY = "hl3_history_v1";
export const loadHistory = () => { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch (e) { return []; } };
export const saveHistory = (h) => { try { localStorage.setItem(KEY, JSON.stringify(h.slice(-12))); } catch (e) { /* ignore */ } };
export function addEntry(pred) {
  const h = loadHistory(); const i = pred.inputs;
  const e = { t: Date.now(), p: pred.probability, tier: pred.tier, hba1c: i.hba1c, glucose: i.glucose, bmi: i.bmi };
  const last = h[h.length - 1];
  if (last && last.hba1c === e.hba1c && last.glucose === e.glucose && last.bmi === e.bmi && Math.abs(last.p - e.p) < 1e-6) return h;
  const next = [...h, e]; saveHistory(next); return next;
}
