"""
The live mode of the hosted site, on Vercel.

On the laptop, app.py is one long-running server that keeps the run in memory
and the page asks it for updates. Vercel works differently: each request is
short-lived and nothing stays in memory between requests. So here a run is one
long request that streams the run back as it happens, one line of JSON per
update. The browser keeps the run. To fix a finding, the browser sends the run
back with the corrected wording, and the server streams the re-check.

Three rules keep this safe, because the site is public:
  1. Every request needs the password in LIVE_PASSWORD (a Vercel secret).
  2. A run can spend at most $2, and a fix at most $0.50, of the AI key.
  3. One run at a time per server copy, so two runs cannot mix their state.

Routes (vercel.json sends /api/live/* here):
  POST /api/live/check   is the password right, and is live mode set up?
  POST /api/live/run     {question, budget}          streams the run
  POST /api/live/fix     {state, i, claim}           streams the re-check
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import app as core  # noqa: E402 - the same run logic the laptop app uses

RUN_CAP, FIX_CAP = 2.0, 0.5          # dollars
MAX_QUESTION, MAX_PASSAGES = 500, 20000  # characters
BUSY = threading.Lock()              # one run at a time in this server copy


def _setup_problem() -> str:
    if not os.environ.get("LIVE_PASSWORD"):
        return "Live mode is not set up. The site owner must set LIVE_PASSWORD."
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        return "Live mode is not set up. The site owner must set an AI key."
    return ""


def _password_ok(given: str) -> bool:
    # A fixed-time comparison, so the reply time does not leak how close a guess was.
    want = os.environ.get("LIVE_PASSWORD", "")
    return bool(want) and hmac.compare_digest(given.encode(), want.encode())


def _snapshot() -> str:
    with core.LOCK:
        out = dict(core.STATE)
        out["passages"] = core.PASSAGES
    out["now"] = time.time()
    return json.dumps(out)


def _start_run(question: str, cap: float) -> threading.Thread:
    t = threading.Thread(target=core.run, args=(question, cap), daemon=True)
    t.start()
    return t


def _start_fix(state: dict, i: int, claim: str) -> threading.Thread:
    # Put the run the browser sent back into memory, then re-check one finding.
    with core.LOCK:
        core.STATE.clear()
        core.STATE.update({
            "status": "done", "question": str(state.get("question", ""))[:MAX_QUESTION],
            "steps": list(state.get("steps") or [])[-200:],
            "claims": list(state.get("claims") or []),
            "papers": list(state.get("papers") or []),
            "lit_stats": dict(state.get("lit_stats") or {}),
            "spend": float(state.get("spend") or 0), "calls": int(state.get("calls") or 0),
            "error": "", "trace": state.get("trace") or {"shared": [], "lanes": []},
            "started": float(state.get("started") or 0), "fixing": i, "tick": 0,
        })
        core.PASSAGES = str(state.get("passages") or "")[:MAX_PASSAGES]
        cap = core.STATE["spend"] + FIX_CAP
    t = threading.Thread(target=core.fix_claim, args=(i, claim, cap), daemon=True)
    t.start()
    return t


async def _read_body(receive) -> dict:
    body = b""
    while True:
        msg = await receive()
        body += msg.get("body", b"")
        if not msg.get("more_body"):
            break
    try:
        return json.loads(body or b"{}")
    except json.JSONDecodeError:
        return {}


async def _reply(send, status: int, data: dict) -> None:
    body = json.dumps(data).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"cache-control", b"no-store")]})
    await send({"type": "http.response.body", "body": body})


async def _stream(send, thread: threading.Thread) -> None:
    """Send one JSON line each time the run changes, until the work is done."""
    await send({"type": "http.response.start", "status": 200,
                "headers": [(b"content-type", b"application/x-ndjson"),
                            (b"cache-control", b"no-store"),
                            (b"x-accel-buffering", b"no")]})
    last, quiet = "", 0
    try:
        while True:
            snap = _snapshot()
            if snap != last or quiet >= 20:   # a heartbeat every ~10s keeps the line open
                await send({"type": "http.response.body", "body": (snap + "\n").encode(),
                            "more_body": True})
                last, quiet = snap, 0
            else:
                quiet += 1
            if not thread.is_alive():
                break
            await asyncio.sleep(0.5)
        await send({"type": "http.response.body", "body": (_snapshot() + "\n").encode()})
    finally:
        if BUSY.locked():
            BUSY.release()


async def app(scope, receive, send):
    if scope["type"] != "http":
        return
    path, method = scope.get("path", ""), scope.get("method", "")
    if method != "POST" or not path.startswith("/api/live/"):
        return await _reply(send, 404, {"error": "Not found."})
    headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
    body = await _read_body(receive)

    problem = _setup_problem()
    if problem:
        return await _reply(send, 503, {"error": problem})
    if not _password_ok(headers.get("x-live-password", "")):
        return await _reply(send, 401, {"error": "The password is wrong."})

    route = path.rsplit("/", 1)[-1]
    if route == "check":
        return await _reply(send, 200, {"ok": True})
    if route not in ("run", "fix"):
        return await _reply(send, 404, {"error": "Not found."})

    if not BUSY.acquire(blocking=False):
        return await _reply(send, 409, {"error": "A run is in progress. Wait until it stops."})
    try:
        if route == "run":
            q = str(body.get("question") or "").strip()[:MAX_QUESTION]
            if not q:
                BUSY.release()
                return await _reply(send, 400, {"error": "Type a question first."})
            cap = min(RUN_CAP, max(0.1, float(body.get("budget") or RUN_CAP)))
            thread = _start_run(q, cap)
        else:
            state, claim = body.get("state") or {}, str(body.get("claim") or "").strip()[:MAX_QUESTION]
            lanes = (state.get("trace") or {}).get("lanes") or []
            i = int(body.get("i", -1))
            if not 0 <= i < len(lanes) or len(claim) < 10:
                BUSY.release()
                return await _reply(send, 400, {"error": "Type the corrected finding first."})
            thread = _start_fix(state, i, claim)
    except Exception as e:  # noqa: BLE001 - always free the lock and say what failed
        if BUSY.locked():
            BUSY.release()
        return await _reply(send, 500, {"error": f"{e.__class__.__name__}: {e}"})
    await _stream(send, thread)
