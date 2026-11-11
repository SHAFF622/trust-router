/*
  Where the run comes from. Both pages (app.html and brain.html) use this file.

  There are three ways to open the app:
    local  On this laptop, with `python3 app.py`. The server holds the run.
    demo   The public site. It shows one recorded run. Nobody can start a run,
           so no AI key is used and it costs nothing.
    live   The public site at /live. It needs a password. The server streams
           the run, and this browser keeps it, so the brain tab can read it.

  For a test, add ?mode=demo or ?mode=live to the address.
*/
(function () {
  const params = new URLSearchParams(location.search);
  const local = ["localhost", "127.0.0.1", "::1", ""].includes(location.hostname);
  const mode = params.get("mode") ||
    (local ? "local" : location.pathname.startsWith("/live") ? "live" : "demo");

  const KEY_STATE = "tr-live-state", KEY_PW = "tr-live-password";
  const EMPTY = () => ({status: "idle", question: "", steps: [], claims: [], papers: [],
    lit_stats: {}, spend: 0, calls: 0, error: "", trace: {shared: [], lanes: []}});
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private window */ } },
    del(k) { try { localStorage.removeItem(k); } catch (e) { /* private window */ } },
  };

  let demo = null, demoChanges = {};   // the recorded run, and dismissals made while viewing it
  let liveMemory = null;               // a copy in memory, if the browser blocks storage

  async function demoState() {
    if (!demo) {
      try {
        const r = await fetch("/demo-run.json", {cache: "no-store"});
        demo = r.ok ? await r.json() : null;
        // A file saved before a run finished has no findings. Treat it as not ready.
        if (!demo || !(demo.trace && demo.trace.lanes && demo.trace.lanes.length))
          demo = {...EMPTY(), demoMissing: true, error: "The recorded run is not ready yet."};
      } catch (e) { demo = {...EMPTY(), demoMissing: true, error: "The recorded run is not ready yet."}; }
    }
    const s = JSON.parse(JSON.stringify(demo));
    (s.trace.lanes || []).forEach((ln, i) => { if (demoChanges[i]) ln.dismissed = true; });
    return s;
  }

  function liveState() {
    const raw = store.get(KEY_STATE);
    if (raw) { try { return JSON.parse(raw); } catch (e) { /* fall through */ } }
    return liveMemory || EMPTY();
  }
  function saveLive(s) { liveMemory = s; store.set(KEY_STATE, JSON.stringify(s)); }

  // Read the server's stream: one full copy of the run per line.
  async function stream(path, body, onUpdate) {
    const r = await fetch(path, {method: "POST",
      headers: {"Content-Type": "application/json", "X-Live-Password": TR.password() || ""},
      body: JSON.stringify(body)});
    if (!r.ok || !r.body) {
      let msg = `The server said ${r.status}.`;
      try { msg = (await r.json()).error || msg; } catch (e) { /* not JSON */ }
      throw new Error(msg);
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "", last = null;
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
        if (!line) continue;
        last = JSON.parse(line);
        saveLive(last); if (onUpdate) onUpdate(last);
      }
    }
    // If the line dropped mid-run, say so, rather than look busy forever.
    if (last && (last.status === "running" || last.fixing != null)) {
      last.status = "error"; last.fixing = null;
      last.error = "The connection stopped before the run ended. Start the run again.";
      saveLive(last); if (onUpdate) onUpdate(last);
    }
    return last;
  }

  const post = async (path, body) =>
    (await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body)})).json();

  const TR = window.TR = {
    mode,
    // Where this mode's pages live, for links between the main page and the brain view.
    home: mode === "live" ? "/live" : "/",
    brain: mode === "live" ? "/live/brain" : "/brain",

    async getState() {
      if (mode === "local") return (await fetch("/api/state")).json();
      if (mode === "demo") return demoState();
      return liveState();
    },

    async run(question, budget, onUpdate) {
      if (mode === "local") {
        const b = await post("/api/run", {question, budget});
        if (b.error) throw new Error(b.error);
        return;
      }
      if (mode === "demo") throw new Error("This is a recorded run. Open the live app to ask a new question.");
      saveLive({...EMPTY(), status: "running", question, started: Date.now() / 1000});
      return stream("/api/live/run", {question, budget}, onUpdate);
    },

    async fix(i, claim, budget, onUpdate) {
      if (mode === "local") {
        const b = await post("/api/fix", {i, claim, budget});
        if (b.error) throw new Error(b.error);
        return;
      }
      if (mode === "demo") throw new Error("You can only correct a finding in the live app, at /live/brain.");
      const s = liveState();
      s.fixing = i; saveLive(s);
      return stream("/api/live/fix", {state: s, i, claim}, onUpdate);
    },

    async dismiss(i) {
      if (mode === "local") {
        const b = await post("/api/dismiss", {i});
        if (b.error) throw new Error(b.error);
        return;
      }
      if (mode === "demo") { demoChanges[i] = true; return; }
      const s = liveState();
      if (s.trace && s.trace.lanes && s.trace.lanes[i]) s.trace.lanes[i].dismissed = true;
      saveLive(s);
    },

    // The live password stays in this browser only.
    password() { return store.get(KEY_PW) || TR._pw || ""; },
    async signIn(pw) {
      const r = await fetch("/api/live/check", {method: "POST",
        headers: {"Content-Type": "application/json", "X-Live-Password": pw}, body: "{}"});
      const b = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(b.error || `The server said ${r.status}.`);
      TR._pw = pw; store.set(KEY_PW, pw);
    },
    signOut() { TR._pw = ""; store.del(KEY_PW); },
  };
})();
