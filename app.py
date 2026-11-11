"""
Ask a research question, watch the agents work, see what to believe.

    ANTHROPIC_API_KEY=sk-ant-... python3 app.py      then http://localhost:8800

You type a question. Three agents work on it, each given different material so
they fail in different ways:

  Data analyst   Proposes specific findings the question implies, from what it
                 knows. Each one is written as a claim you could check.
  Librarian      Searches published papers for real passages about the question
                 and writes a cited answer. This reuses Quorum's literature
                 code: Europe PMC search, full text from PubMed Central, and
                 passage ranking.
  Critic         Reads each claim with the paper passages beside it and says
                 whether the evidence actually supports it. Never sees who
                 wrote the claim.

Then plain code sorts each claim into act on it, check first, or set aside.
No model assigns a tier; a referee can lower trust but never raise it.

Everything streams to the page as it happens.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8800"))
LABELS = {"ACT_ON": "Act on it", "CHECK": "Check first", "SET_ASIDE": "Set aside"}
# Which AI service to use. If there is an Anthropic key, the agents run on
# Claude, as before. If there is only an OpenAI key, they run on three
# different OpenAI models instead. Three different models still disagree in
# useful ways; three copies of the same model would just agree with each other.
USE_OPENAI = bool(os.environ.get("OPENAI_API_KEY")) and not os.environ.get("ANTHROPIC_API_KEY")
if USE_OPENAI:
    ANALYST, LIBRARIAN, CRITIC = "gpt-4.1", "gpt-4.1-mini", "gpt-4o"
else:
    ANALYST, LIBRARIAN, CRITIC = "claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"
# Dollars per million tokens (in, out), so the spending cap counts correctly.
PRICES = {"claude-sonnet-5-5": (2, 10), "claude-haiku-5-5": (0.10, 0.50),
          "claude-opus-5-5": (4, 20),
          "gpt-4.1": (2, 8), "gpt-4.1-mini": (0.40, 1.60), "gpt-4o": (2.50, 10)}

# Quorum already solved paper search, so borrow it rather than writing a worse
# version here. vendor/quorum holds a copy of that one file, so the app works the
# same on this laptop and on the hosted site (see vendor/quorum/__init__.py).
sys.path.insert(0, str(HERE / "vendor"))
try:
    from quorum import literature as LIT
    if os.environ.get("VERCEL"):
        # On Vercel only /tmp can be written to, so the paper cache goes there.
        LIT.CACHE = Path("/tmp/quorum-papers")
except Exception:  # noqa: BLE001 - the app still works without papers
    LIT = None
import knowledge  # noqa: E402 - the world model (Quorum's), optional

try:
    WORLD = knowledge.open_world()
except Exception:  # noqa: BLE001 - the app works without it, as before
    WORLD = None

STATE: dict = {"status": "idle", "question": "", "steps": [], "claims": [],
               "papers": [], "lit_stats": {}, "spend": 0.0, "calls": 0, "error": "",
               "trace": {"shared": [], "lanes": []}}
LOCK = threading.Lock()
PASSAGES = ""  # the paper passages from the last run, so a fix is checked against the same evidence
LIB: dict = {}  # the last paper search, so each finding can be matched to its own best passages


def step(text: str, done: bool = False) -> None:
    with LOCK:
        STATE["steps"].append({"t": text, "done": done, "at": time.time()})
        STATE["tick"] = time.time()   # proof the run is still alive


STALE_AFTER = 180  # seconds without a step before a run counts as dead


def alive() -> bool:
    """Is a run genuinely in progress, or did one die and leave the flag set?"""
    return (STATE.get("status") == "running"
            and time.time() - STATE.get("tick", 0) < STALE_AFTER)


def shared_node(key: str, label: str, agent: str, state: str, detail: str = "") -> None:
    """
    A step every finding shares: reading papers, proposing the findings.

    These sit above the per-finding lanes in the map, because they happen once
    and everything downstream depends on them.
    """
    with LOCK:
        nodes = STATE["trace"]["shared"]
        now = time.time()
        for n in nodes:
            if n["key"] == key:
                n.update(state=state, detail=detail or n["detail"], label=label)
                if state != "running":
                    n["done_at"] = now
                return
        nodes.append({"key": key, "label": label, "agent": agent,
                      "state": state, "detail": detail, "at": now,
                      "done_at": None if state == "running" else now})


def lane(i: int, claim: str) -> dict:
    """One finding's path, from the claim to the tier it ends up in."""
    with LOCK:
        while len(STATE["trace"]["lanes"]) <= i:
            STATE["trace"]["lanes"].append(
                {"claim": "", "steps": [], "tier": None, "trust": None})
        ln = STATE["trace"]["lanes"][i]
        if claim:
            ln["claim"] = claim
            ln.setdefault("at", time.time())
        return ln


def lane_step(i: int, key: str, label: str, agent: str, state: str,
              detail: str = "", vote: int | None = None, why: str = "",
              saw: str = "") -> None:
    """
    One step on a finding's path.

    Beyond what happened, this records who did it, when, how long it took, what
    evidence it was shown, and the reason it gave. A scientist auditing an
    answer needs all five; "the agent decided X" on its own is not checkable.
    """
    ln = lane(i, "")
    now = time.time()
    with LOCK:
        for s in ln["steps"]:
            if s["key"] == key:
                s.update(state=state, label=label, detail=detail or s["detail"], vote=vote)
                if state != "running":
                    s["took"] = round(now - s.get("started", now), 1)
                    s["at"] = now
                if why:
                    s["why"] = why
                if saw:
                    s["saw"] = saw
                return
        ln["steps"].append({"key": key, "label": label, "agent": agent,
                            "state": state, "detail": detail, "vote": vote,
                            "started": now, "at": now, "took": 0,
                            "why": why, "saw": saw})


class Client:
    """
    Talks to Claude, keeps the run inside its budget, and survives a dropped
    connection.

    A single timeout used to end the whole run: one call would hang, the error
    would travel up, and every finding after it was lost. A network blip is no
    reason to throw away work already paid for, so each call gets three tries
    with a growing wait. Errors that retrying cannot fix, such as a bad key,
    fail immediately.
    """

    TRIES = 3
    FATAL = ("AuthenticationError", "PermissionDeniedError",
             "BadRequestError", "NotFoundError")

    def __init__(self, cap: float = 2.0):
        # A call with no answer in two minutes is not coming back. Fail fast
        # and retry rather than holding a dead connection open.
        lib = "openai" if USE_OPENAI else "anthropic"
        try:
            if USE_OPENAI:
                from openai import OpenAI
                self.api = OpenAI(timeout=120.0, max_retries=0)
            else:
                from anthropic import Anthropic
                self.api = Anthropic(timeout=120.0, max_retries=0)
        except ImportError:
            raise RuntimeError(f"The {lib} library is missing from this Python. "
                               f"Run: {sys.executable} -m pip install {lib}") from None
        self.cap = cap

    def ask(self, model: str, prompt: str, max_tokens: int = 900) -> str:
        if STATE["spend"] >= self.cap:
            raise RuntimeError(f"Budget cap of ${self.cap:.2f} reached.")
        for attempt in range(1, self.TRIES + 1):
            try:
                if USE_OPENAI:
                    r = self.api.chat.completions.create(
                        model=model, max_completion_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}])
                else:
                    r = self.api.messages.create(
                        model=model, max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}])
                break
            except Exception as e:  # noqa: BLE001 - judged by name, not by type
                name = e.__class__.__name__
                if name in self.FATAL or attempt == self.TRIES:
                    raise
                wait = 2 ** attempt            # 2s, then 4s
                step(f"{model} gave an error ({name}). Next try in {wait}s "
                     f"(try {attempt + 1} of {self.TRIES}).")
                time.sleep(wait)
        pin, pout = PRICES.get(model, (4, 20))
        # The two services name their token counts and answer text differently.
        if USE_OPENAI:
            tin, tout = r.usage.prompt_tokens, r.usage.completion_tokens
            text = r.choices[0].message.content or ""
        else:
            tin, tout = r.usage.input_tokens, r.usage.output_tokens
            text = "".join(b.text for b in r.content if b.type == "text")
        with LOCK:
            STATE["spend"] += (tin * pin + tout * pout) / 1e6
            STATE["calls"] += 1
        return text.strip()


def cycle_node(cycle: int, key: str, label: str, agent: str, state: str,
               detail: str = "") -> None:
    """A step in one round of the loop, for the map to draw."""
    with LOCK:
        cycles = STATE["trace"].setdefault("cycles", [])
        while len(cycles) <= cycle:
            cycles.append({"n": len(cycles) + 1, "nodes": [], "kept": 0, "dropped": 0})
        for n in cycles[cycle]["nodes"]:
            if n["key"] == key:
                n.update(state=state, detail=detail or n["detail"], label=label)
                return
        cycles[cycle]["nodes"].append({"key": key, "label": label, "agent": agent,
                                       "state": state, "detail": detail})


def parse_claims(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if len(line) > 30:
            out.append(line)
    return out[:6]


def check_claim(c: "Client", i: int, claim: str, passages: str) -> None:
    """
    Check one finding: a second opinion from published papers, a review from a
    different model, and a plain-code test of whether it says anything
    measurable. Then plain code sorts it into a tier.

    This is its own function so that when a person corrects a finding in the
    brain view, exactly these checks re-run on the corrected wording.
    """
    # This finding's own evidence: paper findings about what it names, then the
    # passages that match its words best. Before, every finding got the same
    # first 3,000 characters of passages, whatever it said.
    known = knowledge.known_about(WORLD, claim) if WORLD is not None else []
    own = knowledge.evidence_for(claim, known, LIB, LIT, chars=3000) if (known or LIB) else ""
    passages = own or passages
    step(f"Librarian: checking finding {i} against the papers")
    lane_step(i-1, "lit", "Papers check",
              f"Librarian · {LIBRARIAN}", "running")
    second = c.ask(LIBRARIAN, (
        "Judge whether published work supports this statement. Reply "
        "SUPPORTED, UNSUPPORTED or UNCLEAR, a dash, then under 20 words.\n\n"
        f"Statement: {claim}\n\n"
        f"Passages from papers:\n{passages[:3000] or '(none available)'}"
    ), max_tokens=120)

    step(f"Critic: checking finding {i}")
    lane_step(i-1, "critic", "Critic check",
              f"Critic · {CRITIC}", "running")
    review = c.ask(CRITIC, (
        "You are reviewing a proposed research finding. You do not know "
        "whether it is true. Judge only whether it is specific enough to "
        "test and whether the passages support it. Reply SOUND or SHAKY, "
        "a dash, then under 20 words.\n\n"
        f"Finding: {claim}\n\n"
        f"Passages:\n{passages[:3000] or '(none available)'}"
    ), max_tokens=120)

    votes = []
    if second.upper().startswith("SUPPORTED"):
        votes.append({"n": "The papers support it", "v": 1, "d": second[:120]})
    elif second.upper().startswith("UNSUPPORTED"):
        votes.append({"n": "The papers contradict it", "v": -1, "d": second[:120]})
    else:
        votes.append({"n": "The papers are unclear", "v": 0, "d": second[:120]})

    if review.upper().lstrip().startswith("SHAKY"):
        votes.append({"n": "The critic rejects it", "v": -1, "d": review[:120]})
    else:
        votes.append({"n": "The critic accepts it", "v": 1, "d": review[:120]})

    # Is it specific enough that an experiment could settle it?
    testable = any(w in claim.lower() for w in
                   ("increase", "decrease", "higher", "lower", "%", "more", "less",
                    "correlat", "predict", "associat", "reduce", "greater"))
    votes.append({"n": "It states something measurable" if testable
                  else "It states nothing measurable",
                  "v": 1 if testable else -1,
                  "d": "It names a direction or an amount that a test can check."
                  if testable else "It names no direction or amount. No test can settle it."})

    # Each check becomes a node on this finding's path, so the map shows
    # exactly which evidence pushed it up or down.
    saw = (f"{len(known)} findings from the world model and "
           f"{len(passages)//1000}k characters of evidence chosen for this finding"
           if known else
           f"{len(STATE['papers'])} papers and "
           f"{len(passages)//1000}k characters of the passages that best match this finding"
           if passages else "no papers")
    lane_step(i-1, "lit", "Papers check", f"Librarian · {LIBRARIAN}",
              "done", votes[0]["n"], votes[0]["v"],
              why=second.strip()[:300], saw=saw)
    lane_step(i-1, "critic", "Critic check", f"Critic · {CRITIC}",
              "done", votes[1]["n"], votes[1]["v"],
              why=review.strip()[:300], saw=f"the finding and {saw}")
    lane_step(i-1, "testable", "Measurable check", "Plain code", "done",
              votes[2]["n"], votes[2]["v"],
              why=votes[2]["d"], saw="only the words of the finding")

    score = sum(v["v"] for v in votes)
    doubts = sum(1 for v in votes if v["v"] < 0)
    tier = ("ACT_ON" if doubts == 0 and score >= 2
            else "SET_ASIDE" if doubts >= 2 else "CHECK")
    trust = round(max(0, min(1, .5 + score / 6)), 2)
    rule = ("no doubts, and a score of 2 or more" if tier == "ACT_ON"
            else "2 or more doubts" if tier == "SET_ASIDE"
            else "some support and some doubt")
    lane_step(i-1, "route", "Jev router", "Plain code \u00b7 no model", "done",
              f"Score {score:+d} from {len(votes)} checks. {doubts} doubt"
              f"{'' if doubts == 1 else 's'}.", None,
              why=f"Verdict: {LABELS[tier]}. The rule for this verdict is {rule}. "
                  f"An AI model does not choose the verdict.",
              saw="only the 3 checks above")
    ln = lane(i-1, "")
    with LOCK:
        ln["world"] = [f.id for f in known]  # the world-model findings this finding was judged against
        ln["tier"], ln["trust"] = tier, trust
        ln["problem"] = any(v["v"] < 0 for v in votes[:2])  # papers or reviewer object
        ln["votes"] = votes
        while len(STATE["claims"]) < i:
            STATE["claims"].append(None)
        STATE["claims"][i-1] = {"text": claim, "tier": tier,
                                "trust": trust, "votes": votes}
        run_id, question = STATE.get("run_id", ""), STATE.get("question", "")
    if WORLD is not None and run_id:
        try:
            WORLD.record_routed(run_id, question, f"{run_id}-{i}-{len(ln.get('fixes', []))}", claim, tier, votes)
        except Exception:  # noqa: BLE001 - storing helps later runs; it must not stop this one
            traceback.print_exc()
    step(f"Finding {i}: {LABELS[tier]}", done=True)


def run(question: str, cap: float) -> None:
    try:
        with LOCK:
            STATE.update({"status": "running", "tick": time.time(), "started": time.time(),
                          "fixing": None,
                          "question": question, "steps": [],
                          "claims": [], "papers": [], "lit_stats": {}, "spend": 0.0, "calls": 0,
                          "error": "", "trace": {"shared": [], "lanes": []},
                          "run_id": f"app-{time.strftime('%Y%m%d-%H%M%S')}", "world": None})
        c = Client(cap)
        global PASSAGES, LIB
        PASSAGES, LIB = "", {}

        # 1. Papers first, so the critic has real evidence to judge against.
        passages = ""
        shared_node("papers", "Searching the papers", "Librarian", "running")
        if LIT is not None:
            step("Librarian: searching the papers")
            try:
                lib = LIT.library(question, extra=12, shelf=False, general=True)
                with LOCK:
                    # Keep what the panel shows: how well each paper matched,
                    # whether the full text was read, and the best passage.
                    STATE["papers"] = [{
                        "pmid": p.get("pmid", ""),
                        "title": (p.get("title") or "")[:220],
                        "authors": (p.get("authors") or "")[:90],
                        "journal": p.get("journal", ""), "year": p.get("year", ""),
                        "shelf": p.get("shelf", ""),
                        "relevance": p.get("relevance", 0),
                        "full_text": bool(p.get("full_text")),
                        "paragraphs": p.get("paragraphs", 0),
                        "passage": ((p.get("best_passage") or {}).get("text") or "")[:300],
                    } for p in sorted(lib.get("papers", []),
                                      key=lambda x: -x.get("relevance", 0))[:14]]
                    STATE["lit_stats"] = {
                        "papers": len(lib.get("papers", [])),
                        "full": sum(1 for p in lib.get("papers", []) if p.get("full_text")),
                        "passages": len(lib.get("passages", [])),
                        "ranked_for": question,
                    }
                passages = LIT.context_for(lib)[:6000]
                PASSAGES, LIB = passages, lib
                found = (f"{len(lib.get('papers', []))} papers and "
                         f"{len(lib.get('passages', []))} good passages")
                shared_node("papers", "Read the papers", "Librarian", "done", found)
                step(f"Librarian: found {found}", done=True)
            except Exception as e:  # noqa: BLE001 - no internet must not stop the run
                shared_node("papers", "No papers", "Librarian", "skipped",
                            f"The paper search failed ({e.__class__.__name__}).")
                step(f"Librarian: the paper search failed ({e.__class__.__name__}). "
                     f"The run continues without papers.", done=True)
        else:
            shared_node("papers", "No papers", "Librarian", "skipped",
                        "The Quorum paper code is missing.")
            step("Librarian: the Quorum paper code is missing. "
                 "The run continues without papers.", done=True)

        # 1b. The world model reads new passages into findings (each passage once,
        #     ever), so findings about the same things are known in later runs.
        if WORLD is not None and LIB:
            shared_node("world", "Reading into the world model", f"Librarian · {LIBRARIAN}", "running")
            try:
                got = knowledge.learn(WORLD, LIB, c.ask, LIBRARIAN)
                total = WORLD.stats()
                with LOCK:  # what the brain view draws: the findings it holds for this question
                    STATE["world"] = {"findings": knowledge.findings_from(WORLD, got.get("ids", [])),
                                      "stats": total.model_dump()}
                shared_node("world", "World model", f"Librarian · {LIBRARIAN}", "done",
                            f"{got['kept']} new findings from {got['passages']} new passages; "
                            f"{got['skipped']} passages read before. It holds {total.findings} findings "
                            f"from {total.papers} papers.")
                step(f"World model: {got['kept']} new findings, {total.findings} in all", done=True)
            except Exception as e:  # noqa: BLE001 - optional
                shared_node("world", "World model", "Librarian", "skipped",
                            f"Could not read the passages ({e.__class__.__name__}).")

        # 2. The analyst proposes specific, checkable claims.
        shared_node("propose", "Writing findings", f"Analyst · {ANALYST}", "running")
        step("Analyst: writing findings")
        claims = parse_claims(c.ask(ANALYST, (
            "You are a research analyst. For the question below, write 4 to 6 "
            "specific findings that could be true, each one a single sentence "
            "stating something concrete enough to be checked. No hedging, no "
            "preamble, one per line.\n\n"
            f"Question: {question}"
        )))
        shared_node("propose", "Wrote findings", f"Analyst · {ANALYST}", "done",
                    f"{len(claims)} findings. Each finding gets its own checks.")
        step(f"Analyst: wrote {len(claims)} findings", done=True)
        for i, claim in enumerate(claims):
            lane(i, claim)

        # 3. A second opinion on each claim, from a different model family, so
        #    it does not make the same mistakes as the analyst.
        #    The findings are checked at the same time, not one after another:
        #    each check waits on the network, so running them side by side
        #    finishes in roughly the time of the slowest one.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, min(6, len(claims)))) as pool:
            list(pool.map(lambda ic: check_claim(c, ic[0], ic[1], passages), enumerate(claims, 1)))

        step("Done", done=True)
        with LOCK:
            STATE["status"] = "done"
    except Exception as e:  # noqa: BLE001 - surface the failure on the page
        traceback.print_exc()
        with LOCK:
            STATE["status"] = "error"
            STATE["error"] = f"{e.__class__.__name__}: {e}"


def fix_claim(i: int, claim: str, cap: float) -> None:
    """
    A person corrected finding i in the brain view. Keep a record of what it
    said before, then re-run only that finding's checks on the new wording.
    Nothing else in the run changes, the same way fixing one wrong step should
    not mean redoing the whole piece of work.
    """
    try:
        ln = lane(i, "")
        with LOCK:
            ln.setdefault("fixes", []).append(
                {"before": ln["claim"], "after": claim, "at": time.time()})
            ln["claim"], ln["steps"], ln["dismissed"] = claim, [], False
            STATE["fixing"], STATE["tick"] = i, time.time()
        step(f"You corrected finding {i+1}. The librarian and the critic check it again.")
        check_claim(Client(cap), i + 1, claim, PASSAGES)
    except Exception as e:  # noqa: BLE001 - show the failure on the page
        traceback.print_exc()
        with LOCK:
            STATE["error"] = f"The fix failed: {e.__class__.__name__}: {e}"
    finally:
        with LOCK:
            STATE["fixing"] = None


def page(name: str = "app.html") -> str:
    """
    Read the page from disk on every request.

    Loading it once at startup means an edit to app.html does nothing until the
    server is restarted, which is a confusing way to lose ten minutes.
    """
    f = HERE / name
    return f.read_text() if f.exists() else f"<h1>{name} is missing</h1>"


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            with LOCK:
                out = dict(STATE)
                out["now"] = time.time()   # so the brain view can line its clock up with ours
                if out.get("status") == "running" and not alive():
                    out["status"] = "error"
                    out["error"] = ("The run stopped before the end. "
                                    "Click Run the panel again.")
                self._send(json.dumps(out).encode(), "application/json")
        elif path == "/source.js":
            self._send(page("source.js").encode(), "text/javascript; charset=utf-8")
        elif path == "/brain":
            self._send(page("brain.html").encode(), "text/html; charset=utf-8")
        else:
            self._send(page().encode(), "text/html; charset=utf-8")

    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        reply = lambda d: self._send(json.dumps(d).encode(), "application/json")
        if path in ("/api/fix", "/api/dismiss"):
            i = int(body.get("i", -1))
            if not 0 <= i < len(STATE["trace"]["lanes"]):
                return reply({"error": "This finding does not exist."})
            if alive() or STATE.get("fixing") is not None:
                return reply({"error": "A run is in progress. Wait until it stops."})
            if path == "/api/dismiss":
                with LOCK:
                    STATE["trace"]["lanes"][i]["dismissed"] = True
                return reply({"ok": True})
            claim = (body.get("claim") or "").strip()
            if len(claim) < 10:
                return reply({"error": "Type the corrected finding first."})
            with LOCK:
                STATE["fixing"] = i   # claim the slot now, so a double click cannot start two
            threading.Thread(target=fix_claim, args=(i, claim, float(body.get("budget", 2))),
                             daemon=True).start()
            return reply({"ok": True})
        if path != "/api/run":
            return self._send(b"{}", "application/json")
        q = (body.get("question") or "").strip()
        if alive():
            return self._send(json.dumps({"error": "A run is in progress. Wait until it stops."}).encode(),
                              "application/json")
        if not q:
            return self._send(json.dumps({"error": "Type a question first."}).encode(),
                              "application/json")
        threading.Thread(target=run, args=(q, float(body.get("budget", 2))),
                         daemon=True).start()
        self._send(json.dumps({"ok": True}).encode(), "application/json")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        sys.exit("Set an API key first, either one:\n"
                 "  ANTHROPIC_API_KEY=sk-ant-... python3 app.py\n"
                 "  OPENAI_API_KEY=sk-...        python3 app.py")
    print(f"Agents on {'OpenAI' if USE_OPENAI else 'Claude'}: "
          f"{ANALYST}, {LIBRARIAN}, {CRITIC}")
    print(f"Open http://localhost:{PORT}  (ctrl-C to stop)")
    print(f"Papers: {'on, via Quorum' if LIT else 'off'}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
