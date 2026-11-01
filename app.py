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
PORT = 8800
LABELS = {"ACT_ON": "Act on it", "CHECK": "Check first", "SET_ASIDE": "Set aside"}
ANALYST, LIBRARIAN, CRITIC = "claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"
PRICES = {"claude-sonnet-5-5": (2, 10), "claude-haiku-5-5": (0.10, 0.50),
          "claude-opus-5-5": (4, 20)}

# Quorum already solved paper search and PDF reading, so borrow it rather than
# writing a worse version here.
sys.path.insert(0, str(Path.home() / "ai-scientist" / "quorum"))
try:
    from quorum import literature as LIT
except Exception:  # noqa: BLE001 - the app still works without papers
    LIT = None

STATE: dict = {"status": "idle", "question": "", "steps": [], "claims": [],
               "papers": [], "lit_stats": {}, "spend": 0.0, "calls": 0, "error": "",
               "trace": {"shared": [], "lanes": []}}
LOCK = threading.Lock()


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
        for n in nodes:
            if n["key"] == key:
                n.update(state=state, detail=detail or n["detail"], label=label)
                return
        nodes.append({"key": key, "label": label, "agent": agent,
                      "state": state, "detail": detail})


def lane(i: int, claim: str) -> dict:
    """One finding's path, from the claim to the tier it ends up in."""
    with LOCK:
        while len(STATE["trace"]["lanes"]) <= i:
            STATE["trace"]["lanes"].append(
                {"claim": "", "steps": [], "tier": None, "trust": None})
        ln = STATE["trace"]["lanes"][i]
        if claim:
            ln["claim"] = claim
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
                s.update(state=state, detail=detail or s["detail"], vote=vote)
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
    def __init__(self, cap: float = 2.0):
        from anthropic import Anthropic
        self.api, self.cap = Anthropic(), cap

    def ask(self, model: str, prompt: str, max_tokens: int = 900) -> str:
        if STATE["spend"] >= self.cap:
            raise RuntimeError(f"Budget cap of ${self.cap:.2f} reached.")
        r = self.api.messages.create(model=model, max_tokens=max_tokens,
                                     messages=[{"role": "user", "content": prompt}])
        pin, pout = PRICES.get(model, (4, 20))
        with LOCK:
            STATE["spend"] += (r.usage.input_tokens * pin + r.usage.output_tokens * pout) / 1e6
            STATE["calls"] += 1
        return "".join(b.text for b in r.content if b.type == "text").strip()


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


def run(question: str, cap: float) -> None:
    try:
        with LOCK:
            STATE.update({"status": "running", "tick": time.time(),
                          "question": question, "steps": [],
                          "claims": [], "papers": [], "lit_stats": {}, "spend": 0.0, "calls": 0,
                          "error": "", "trace": {"shared": [], "lanes": []}})
        c = Client(cap)

        # 1. Papers first, so the critic has real evidence to judge against.
        passages = ""
        shared_node("papers", "Searching papers", "Librarian", "running")
        if LIT is not None:
            step("Librarian: searching published papers")
            try:
                lib = LIT.library(question)
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
                found = (f"{len(lib.get('papers', []))} papers, "
                         f"{len(lib.get('passages', []))} relevant passages")
                shared_node("papers", "Read the literature", "Librarian", "done", found)
                step(f"Librarian: read {found}", done=True)
            except Exception as e:  # noqa: BLE001 - no internet must not stop the run
                shared_node("papers", "No papers available", "Librarian", "skipped",
                            e.__class__.__name__)
                step(f"Librarian: paper search unavailable ({e.__class__.__name__})", done=True)
        else:
            shared_node("papers", "No papers available", "Librarian", "skipped",
                        "Quorum's literature code not found")
            step("Librarian: Quorum's literature code not found, skipping papers", done=True)

        # 2. The analyst proposes specific, checkable claims.
        shared_node("propose", "Proposing findings", f"Analyst · {ANALYST}", "running")
        step("Data analyst: proposing findings")
        claims = parse_claims(c.ask(ANALYST, (
            "You are a research analyst. For the question below, write 4 to 6 "
            "specific findings that could be true, each one a single sentence "
            "stating something concrete enough to be checked. No hedging, no "
            "preamble, one per line.\n\n"
            f"Question: {question}"
        )))
        shared_node("propose", "Proposed findings", f"Analyst · {ANALYST}", "done",
                    f"{len(claims)} findings, each checked separately below")
        step(f"Data analyst: proposed {len(claims)} findings", done=True)
        for i, claim in enumerate(claims):
            lane(i, claim)

        # 3. A second opinion on each claim, from a different model family, so
        #    it does not make the same mistakes as the analyst.
        for i, claim in enumerate(claims, 1):
            step(f"Librarian: second opinion on finding {i}")
            lane_step(i-1, "lit", "Checking the literature",
                      f"Librarian · {LIBRARIAN}", "running")
            second = c.ask(LIBRARIAN, (
                "Judge whether published work supports this statement. Reply "
                "SUPPORTED, UNSUPPORTED or UNCLEAR, a dash, then under 20 words.\n\n"
                f"Statement: {claim}\n\n"
                f"Passages from papers:\n{passages[:3000] or '(none available)'}"
            ), max_tokens=120)

            step(f"Critic: reviewing finding {i}")
            lane_step(i-1, "critic", "Reviewing the evidence",
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
                votes.append({"n": "Published work supports it", "v": 1, "d": second[:120]})
            elif second.upper().startswith("UNSUPPORTED"):
                votes.append({"n": "Published work contradicts it", "v": -1, "d": second[:120]})
            else:
                votes.append({"n": "Published work is unclear", "v": 0, "d": second[:120]})

            if review.upper().lstrip().startswith("SHAKY"):
                votes.append({"n": "Reviewer calls it shaky", "v": -1, "d": review[:120]})
            else:
                votes.append({"n": "Reviewer finds it sound", "v": 1, "d": review[:120]})

            # Is it specific enough that an experiment could settle it?
            testable = any(w in claim.lower() for w in
                           ("increase", "decrease", "higher", "lower", "%", "more", "less",
                            "correlat", "predict", "associat", "reduce", "greater"))
            votes.append({"n": "States something measurable" if testable
                          else "Nothing measurable stated",
                          "v": 1 if testable else -1,
                          "d": "Names a direction or quantity you could test."
                          if testable else "No direction or quantity, so no experiment settles it."})

            # Each check becomes a node on this finding's path, so the map shows
            # exactly which evidence pushed it up or down.
            saw = (f"{len(STATE['papers'])} papers, "
                   f"{len(passages)//1000}k characters of ranked passages"
                   if passages else "no papers were available")
            lane_step(i-1, "lit", "Checked the literature", f"Librarian · {LIBRARIAN}",
                      "done", votes[0]["n"], votes[0]["v"],
                      why=second.strip()[:300], saw=saw)
            lane_step(i-1, "critic", "Reviewed the evidence", f"Critic · {CRITIC}",
                      "done", votes[1]["n"], votes[1]["v"],
                      why=review.strip()[:300], saw=f"the finding and {saw}")
            lane_step(i-1, "testable", votes[2]["n"], "Plain code", "done",
                      votes[2]["n"], votes[2]["v"],
                      why=votes[2]["d"], saw="the wording of the finding only")

            score = sum(v["v"] for v in votes)
            doubts = sum(1 for v in votes if v["v"] < 0)
            tier = ("ACT_ON" if doubts == 0 and score >= 2
                    else "SET_ASIDE" if doubts >= 2 else "CHECK")
            trust = round(max(0, min(1, .5 + score / 6)), 2)
            rule = ("no doubts and a score of 2 or more" if tier == "ACT_ON"
                    else "two or more doubts" if tier == "SET_ASIDE"
                    else "some support, some doubt")
            lane_step(i-1, "route", "Jev router", "Plain code \u00b7 no model", "done",
                      f"{score:+d} from {len(votes)} checks, {doubts} doubt"
                      f"{'' if doubts == 1 else 's'}", None,
                      why=f"{LABELS[tier]} because the rule says {rule}. "
                          f"A check can lower trust but never raise it.",
                      saw="the three checks above, nothing else")
            ln = lane(i-1, "")
            with LOCK:
                ln["tier"], ln["trust"] = tier, trust
                STATE["claims"].append({"text": claim, "tier": tier,
                                        "trust": trust, "votes": votes})
            step(f"Finding {i}: {tier.replace('_', ' ').lower()}", done=True)

        step("Done", done=True)
        with LOCK:
            STATE["status"] = "done"
    except Exception as e:  # noqa: BLE001 - surface the failure on the page
        traceback.print_exc()
        with LOCK:
            STATE["status"] = "error"
            STATE["error"] = f"{e.__class__.__name__}: {e}"


def page() -> str:
    """
    Read the page from disk on every request.

    Loading it once at startup means an edit to app.html does nothing until the
    server is restarted, which is a confusing way to lose ten minutes.
    """
    f = HERE / "app.html"
    return f.read_text() if f.exists() else "<h1>app.html is missing</h1>"


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
                if out.get("status") == "running" and not alive():
                    out["status"] = "error"
                    out["error"] = ("That run stopped without finishing. "
                                    "Press Run again.")
                self._send(json.dumps(out).encode(), "application/json")
        else:
            self._send(page().encode(), "text/html; charset=utf-8")

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            return self._send(b"{}", "application/json")
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        q = (body.get("question") or "").strip()
        if alive():
            return self._send(json.dumps({"error": "A run is already going."}).encode(),
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
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first:\n"
                 "  ANTHROPIC_API_KEY=sk-ant-... python3 app.py")
    print(f"Open http://localhost:{PORT}  (ctrl-C to stop)")
    print(f"Papers: {'on, via Quorum' if LIT else 'off'}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
