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
               "papers": [], "spend": 0.0, "calls": 0, "error": ""}
LOCK = threading.Lock()


def step(text: str, done: bool = False) -> None:
    with LOCK:
        STATE["steps"].append({"t": text, "done": done, "at": time.time()})


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
            STATE.update({"status": "running", "question": question, "steps": [],
                          "claims": [], "papers": [], "spend": 0.0, "calls": 0,
                          "error": ""})
        c = Client(cap)

        # 1. Papers first, so the critic has real evidence to judge against.
        passages = ""
        if LIT is not None:
            step("Librarian: searching published papers")
            try:
                lib = LIT.library(question)
                with LOCK:
                    STATE["papers"] = [{"pmid": p.get("pmid", ""),
                                        "title": (p.get("title") or "")[:160]}
                                       for p in lib.get("papers", [])[:8]]
                passages = LIT.context_for(lib)[:6000]
                step(f"Librarian: read {len(lib.get('papers', []))} papers, "
                     f"{len(lib.get('passages', []))} relevant passages", done=True)
            except Exception as e:  # noqa: BLE001 - no internet must not stop the run
                step(f"Librarian: paper search unavailable ({e.__class__.__name__})", done=True)
        else:
            step("Librarian: Quorum's literature code not found, skipping papers", done=True)

        # 2. The analyst proposes specific, checkable claims.
        step("Data analyst: proposing findings")
        claims = parse_claims(c.ask(ANALYST, (
            "You are a research analyst. For the question below, write 4 to 6 "
            "specific findings that could be true, each one a single sentence "
            "stating something concrete enough to be checked. No hedging, no "
            "preamble, one per line.\n\n"
            f"Question: {question}"
        )))
        step(f"Data analyst: proposed {len(claims)} findings", done=True)

        # 3. A second opinion on each claim, from a different model family, so
        #    it does not make the same mistakes as the analyst.
        for i, claim in enumerate(claims, 1):
            step(f"Librarian: second opinion on finding {i}")
            second = c.ask(LIBRARIAN, (
                "Judge whether published work supports this statement. Reply "
                "SUPPORTED, UNSUPPORTED or UNCLEAR, a dash, then under 20 words.\n\n"
                f"Statement: {claim}\n\n"
                f"Passages from papers:\n{passages[:3000] or '(none available)'}"
            ), max_tokens=120)

            step(f"Critic: reviewing finding {i}")
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

            score = sum(v["v"] for v in votes)
            doubts = sum(1 for v in votes if v["v"] < 0)
            tier = ("ACT_ON" if doubts == 0 and score >= 2
                    else "SET_ASIDE" if doubts >= 2 else "CHECK")
            with LOCK:
                STATE["claims"].append({"text": claim, "tier": tier,
                                        "trust": round(max(0, min(1, .5 + score / 6)), 2),
                                        "votes": votes})
            step(f"Finding {i}: {tier.replace('_', ' ').lower()}", done=True)

        step("Done", done=True)
        with LOCK:
            STATE["status"] = "done"
    except Exception as e:  # noqa: BLE001 - surface the failure on the page
        traceback.print_exc()
        with LOCK:
            STATE["status"] = "error"
            STATE["error"] = f"{e.__class__.__name__}: {e}"


PAGE = (HERE / "app.html").read_text() if (HERE / "app.html").exists() else "<h1>app.html missing</h1>"


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
                self._send(json.dumps(STATE).encode(), "application/json")
        else:
            self._send(PAGE.encode(), "text/html; charset=utf-8")

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            return self._send(b"{}", "application/json")
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        q = (body.get("question") or "").strip()
        if STATE["status"] == "running":
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
