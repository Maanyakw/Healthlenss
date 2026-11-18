"use client";
import { useEffect, useRef, useState } from "react";
import { postJSON } from "../../lib/api";
import { useSession } from "../../lib/session";
import { Gate, Notice, RichText } from "../../lib/ui";

const SUGGESTIONS = ["Why is my risk at this level?", "What does my HbA1c mean?", "What if my BMI were 24?", "Which lifestyle changes help most?", "How is diabetes confirmed?"];
const GREETING = "Hello. I can explain your HbA1c, glucose and BMI, why your estimate came out the way it did, and how changes might shift it. I can also answer general questions. I can't diagnose or advise on medicines. What would you like to know?";

function metaFor(r) {
  let m = r.mode === "genai" ? "Generative AI (" + r.provider + "), general education, not medical advice"
    : r.mode === "safety" ? "Safety message"
    : r.mode === "local-fallback" ? "Local assistant (generative AI was unavailable)"
    : "Local assistant. Add a GROQ_API_KEY on the server for open-ended answers";
  if (r.whatif) m += ". What-if calculated by the risk model";
  return m;
}

export default function ChatPage() {
  const { s, ready, update } = useSession();
  const [msgs, setMsgs] = useState([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);
  const log = useRef(null);

  useEffect(() => { if (ready) setMsgs(s.chat || []); }, [ready]); // eslint-disable-line
  useEffect(() => { if (log.current) log.current.scrollTop = log.current.scrollHeight; }, [msgs, busy]);

  if (!ready) return null;
  if (!s.patient) return <><div className="page-head"><h1>Ask HealthLens</h1></div><Gate text="Add your lab values first so the assistant knows what to talk about." /></>;

  async function ask(q) {
    q = q.trim(); if (!q || busy) return;
    setBusy(true); setText(""); setNotice(null);
    const before = msgs; const withUser = [...before, { role: "user", content: q }];
    setMsgs(withUser);
    try {
      const res = await postJSON("/api/chat", {
        message: q, history: before.slice(-12).map(({ role, content }) => ({ role, content })), patient: s.patient,
        narrative: (s.prediction && s.prediction.narrative && s.prediction.narrative.text) || "",
      });
      const next = [...withUser, { role: "assistant", content: res.reply, meta: metaFor(res) }];
      setMsgs(next); update({ chat: next });
      if (res.notice) setNotice(res.notice);
    } catch (e) { setNotice({ error: e.message }); }
    setBusy(false);
  }

  return (
    <>
      <div className="page-head">
        <h1>Ask HealthLens</h1>
        <p className="lede">Ask about your result or anything general. What-if questions (for example, &quot;what if my BMI were 24?&quot;) are calculated by the risk model, then explained by the assistant.</p>
      </div>
      <p className="small muted">In an emergency (confusion, vomiting with very high sugar, chest pain, trouble breathing, fainting) call your local emergency number instead of using this chat.</p>
      {msgs.length === 0 && <div className="suggest" aria-label="Suggested questions">{SUGGESTIONS.map((q) => <button key={q} type="button" onClick={() => ask(q)}>{q}</button>)}</div>}
      <div className="card">
        <div className="chat-log" ref={log} role="log" aria-live="polite" aria-label="Conversation">
          <div className="msg bot"><RichText text={GREETING} /></div>
          {msgs.map((m, i) => (
            <div key={i} className={"msg " + (m.role === "user" ? "user" : "bot")}>
              {m.role === "user" ? m.content : <RichText text={m.content} />}
              {m.meta && <div className="meta">{m.meta}</div>}
            </div>
          ))}
          {busy && <div className="msg bot"><span className="spinner" style={{ display: "inline-block" }} /> Thinking…</div>}
        </div>
        {notice && (notice.error ? <Notice kind="error">{notice.error}</Notice> : <Notice kind="warn">{notice}</Notice>)}
        <form className="chat-form" onSubmit={(e) => { e.preventDefault(); ask(text); }}>
          <textarea rows={1} maxLength={1000} value={text} placeholder="Type a question…" aria-label="Your question"
            onChange={(e) => setText(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(text); } }} />
          <button className="btn" type="submit" disabled={busy || !text.trim()}>Send</button>
        </form>
      </div>
      <div className="btn-row">
        <a className="btn" href="/report">Create the doctor report</a>
        {msgs.length > 0 && <button className="btn ghost" type="button" onClick={() => { setMsgs([]); update({ chat: [] }); }}>Clear conversation</button>}
      </div>
    </>
  );
}
