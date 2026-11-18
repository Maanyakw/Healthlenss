export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export async function api(path, opts) {
  let res;
  try { res = await fetch(API_URL + path, opts); }
  catch (e) { throw new Error("Could not reach the HealthLens server. If it was idle it may be waking up; wait a moment and try again."); }
  if (!res.ok) {
    let detail = "";
    try { const j = await res.json(); detail = j.detail; } catch (e) { /* not json */ }
    if (Array.isArray(detail)) detail = detail.map((d) => (d.loc ? d.loc.slice(-1)[0] + ": " : "") + d.msg).join("; ");
    throw new Error(detail || "The server returned an error (" + res.status + ").");
  }
  return res;
}

export const postJSON = (path, body) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => r.json());
export const getJSON = (path) => api(path).then((r) => r.json());
