"""
The loop with a gate in it.

A research system that runs in cycles has a compounding problem: cycle 2 builds
on whatever cycle 1 produced, so one bad finding early becomes the foundation
for everything after it. Kosmos has this shape and no gate — its world model
takes whatever the agents wrote.

This puts the router *inside* the loop rather than at the end of it. Each cycle:

  1. The orchestrator reads the shared record and proposes several tasks.
  2. Those tasks run side by side, each producing one finding.
  3. Every finding is routed the moment it exists: act on it, check first,
     set aside.
  4. Only findings that survive enter the shared record. The rest are kept in
     a dropped pile, visible but not built upon.
  5. The next cycle reads the filtered record.

The claim worth testing is simple: does gating produce better final findings
than not gating? Run it both ways and compare. `--no-gate` keeps everything,
so the loop behaves like Kosmos, and the two runs are otherwise identical.

Run:  ANTHROPIC_API_KEY=... python3 recursive.py "your question" --cycles 3
      ANTHROPIC_API_KEY=... python3 recursive.py "your question" --no-gate
      ANTHROPIC_API_KEY=... python3 recursive.py "your question" --no-world

The shared record lives in the world model (knowledge.py, Quorum's world model):
every finding with its tier and checks, linked to what papers report about the
same things. Each cycle the orchestrator reads all of it, grouped by tier, not
just the last few sentences. A task searches the papers only when the world
model knows nothing about what its finding names. `--no-world` turns this off
(the old plain list and a fresh search per task), so the two can be compared.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORCHESTRATOR = "claude-opus-5-5"      # decides what to work on next
ANALYST = "claude-sonnet-5-5"          # does the work
LIBRARIAN = "claude-haiku-5-5"         # a different family, so it errs differently
CRITIC = "claude-opus-5-5"             # judges, never writes
PRICES = {"claude-sonnet-5-5": (2, 10), "claude-haiku-5-5": (0.10, 0.50),
          "claude-opus-5-5": (4, 20)}

# The same copy of Quorum's code the app uses (paper search and the world model).
sys.path.insert(0, str(HERE / "vendor"))
try:
    from quorum import literature as LIT
except Exception:  # noqa: BLE001 - papers are optional
    LIT = None
import knowledge  # noqa: E402


class Budget(RuntimeError):
    pass


class Client:
    """Shared across threads, so the spend cap holds across parallel tasks."""

    def __init__(self, cap: float):
        from anthropic import Anthropic
        self.api, self.cap = Anthropic(), cap
        self.spent, self.calls = 0.0, 0
        self.lock = threading.Lock()

    def ask(self, model: str, prompt: str, max_tokens: int = 700) -> str:
        with self.lock:
            if self.spent >= self.cap:
                raise Budget(f"Spent ${self.spent:.2f} of ${self.cap:.2f}.")
        r = self.api.messages.create(model=model, max_tokens=max_tokens,
                                     messages=[{"role": "user", "content": prompt}])
        pin, pout = PRICES.get(model, (4, 20))
        with self.lock:
            self.spent += (r.usage.input_tokens * pin + r.usage.output_tokens * pout) / 1e6
            self.calls += 1
        return "".join(b.text for b in r.content if b.type == "text").strip()


def lines_of(text: str, limit: int) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if len(line) > 25:
            out.append(line)
    return out[:limit]


TESTABLE = re.compile(
    r"\d|\b(increase|decrease|higher|lower|greater|reduce|correlat|predict|"
    r"associat|fold|percent|%|more|less|faster|slower)\w*", re.I)


def route(finding: str, lit_verdict: str, critic_verdict: str) -> dict:
    """
    Plain code, run the moment a finding exists. This is the gate.

    A check can lower trust but never raise it past what the evidence allows,
    and "no evidence either way" counts against a finding rather than being
    treated as neutral — an unverifiable claim is not a safe one to build on.
    """
    votes = []
    lv = lit_verdict.upper()
    if lv.startswith("SUPPORTED"):
        votes.append({"n": "Published work supports it", "v": 1, "d": lit_verdict[:140]})
    elif lv.startswith("UNSUPPORTED"):
        votes.append({"n": "Published work contradicts it", "v": -1, "d": lit_verdict[:140]})
    else:
        votes.append({"n": "No published work either way", "v": -1,
                      "d": lit_verdict[:140] or "The papers do not address this."})

    if critic_verdict.upper().lstrip().startswith("SHAKY"):
        votes.append({"n": "Reviewer calls it shaky", "v": -1, "d": critic_verdict[:140]})
    else:
        votes.append({"n": "Reviewer finds it sound", "v": 1, "d": critic_verdict[:140]})

    if TESTABLE.search(finding):
        votes.append({"n": "States something measurable", "v": 1,
                      "d": "Names a direction or quantity an experiment could settle."})
    else:
        votes.append({"n": "Nothing measurable stated", "v": -1,
                      "d": "No direction or quantity, so no experiment settles it."})

    score = sum(v["v"] for v in votes)
    doubts = sum(1 for v in votes if v["v"] < 0)
    tier = ("ACT_ON" if doubts == 0 and score >= 2
            else "SET_ASIDE" if doubts >= 2 else "CHECK")
    return {"tier": tier, "trust": round(max(0, min(1, .5 + score / 6)), 2),
            "votes": votes}


def one_task(c: Client, question: str, task: str, record: list[dict], world=None,
             run_id: str = "", established: str = "") -> dict:
    """
    One parallel task: do the work, find the evidence, judge it, route it.

    The literature search happens per task rather than once per question,
    because the agents drift onto sub-topics the original search never covered.
    With the world model on, the task first asks what is already known about
    what its finding names, and searches only when nothing is.
    """
    context = established or "\n".join(f"- {r['finding']}" for r in record[-6:]) or "(nothing yet)"

    finding = c.ask(ANALYST, (
        f"Research question: {question}\n\n"
        f"Already established:\n{context}\n\n"
        f"Your task: {task}\n\n"
        "Write ONE specific finding this task supports: a single sentence "
        "stating something concrete enough to be checked. No hedging."
    ), max_tokens=250).strip().split("\n")[0][:400]

    passages, searched, learned = "", False, {}
    known = knowledge.known_about(world, finding) if world is not None else []
    if world is not None and len(known) >= 2:
        passages = knowledge.evidence_for(finding, known, None, LIT)   # already known: no search
    elif LIT is not None:
        try:
            searched = True
            lib = LIT.library(finding, extra=12, shelf=False, general=True)
            if world is not None:
                learned = knowledge.learn(world, lib, c.ask, LIBRARIAN)
                known = knowledge.known_about(world, finding)
                passages = knowledge.evidence_for(finding, known, lib, LIT)
            else:
                passages = LIT.context_for(lib)[:3500]
        except Budget:
            raise
        except Exception:  # noqa: BLE001 - carry on without papers
            pass

    lit = c.ask(LIBRARIAN, (
        "Does published work support this statement? Reply SUPPORTED, "
        "UNSUPPORTED or UNCLEAR, a dash, then under 20 words.\n\n"
        f"Statement: {finding}\n\nPassages:\n{passages or '(none found)'}"
    ), max_tokens=120)

    crit = c.ask(CRITIC, (
        "Judge a proposed research finding. You do not know if it is true. "
        "Judge only whether it is specific enough to test and whether the "
        "passages support it. Reply SOUND or SHAKY, a dash, under 20 words.\n\n"
        f"Finding: {finding}\n\nPassages:\n{passages or '(none found)'}"
    ), max_tokens=120)

    out = {"task": task, "finding": finding, "searched": searched, "known": len(known),
           "learned": learned, **route(finding, lit, crit)}
    if world is not None:
        out["id"] = uuid.uuid4().hex[:12]
        try:
            world.record_routed(run_id, question, out["id"], finding, out["tier"], out["votes"],
                                extra={"task": task})
        except Exception as e:  # noqa: BLE001 - the record helps; it must not stop the loop
            print(f"   (world model: could not store a finding: {e.__class__.__name__})")
    return out


TIER_WORDS = {"ACT_ON": "Act on it", "CHECK": "Check first", "SET_ASIDE": "Set aside"}


def established_from(world, run_id: str, kept_ids: set[str]) -> str:
    """
    What later work may build on, read from the world model in one export
    (prefetch_related: a fixed number of queries however long the run): every
    kept finding of this run, grouped by tier, with how many paper findings back it.
    """
    out = world.export_run(run_id)
    if out is None:
        return ""
    lines = []
    for tier in ("ACT_ON", "CHECK", "SET_ASIDE"):  # with the gate off, set-aside findings are kept too
        rows = [c for c in out.claims if c.claim_id in kept_ids and c.verdict and c.verdict.label == tier]
        if rows:
            lines.append(f"{TIER_WORDS[tier]}:")
            lines += [f"- {c.text}" + (f" (backed by {len(c.findings)} paper findings)" if c.findings else "")
                      for c in rows]
    return "\n".join(lines)


def run(question: str, cycles: int, width: int, cap: float, gate: bool, use_world: bool = True,
        client=None) -> dict:
    c = client or Client(cap)
    record: list[dict] = []     # what later cycles are allowed to build on
    dropped: list[dict] = []    # what the gate refused, kept for the report
    history = []
    world = knowledge.open_world() if use_world else None
    run_id = f"loop-{time.strftime('%Y%m%d-%H%M%S')}-{'gated' if gate else 'ungated'}-{uuid.uuid4().hex[:4]}"
    kept_ids: set[str] = set()
    print(f"World model: {'on, ' + str(knowledge.db_path()) if world is not None else 'off'}")

    for cycle in range(1, cycles + 1):
        established = (established_from(world, run_id, kept_ids) if world is not None
                       else "\n".join(f"- {r['finding']}" for r in record[-8:])) or "(nothing yet)"
        print(f"\n── Cycle {cycle} " + "─" * 46)

        tasks = lines_of(c.ask(ORCHESTRATOR, (
            f"Research question: {question}\n\n"
            f"Established so far:\n{established}\n\n"
            f"Propose {width} next steps that would each add something new. "
            "One per line, no preamble, each a short instruction."
        ), max_tokens=500), width)
        print(f"   orchestrator proposed {len(tasks)} tasks")

        try:
            with ThreadPoolExecutor(max_workers=min(width, 6)) as pool:
                results = list(pool.map(lambda t: one_task(c, question, t, record, world, run_id,
                                                           established if world is not None else ""), tasks))
        except Budget as e:
            print(f"   stopped: {e}")
            break

        kept = 0
        for r in results:
            r["cycle"] = cycle
            if not gate or r["tier"] != "SET_ASIDE":
                record.append(r)
                kept_ids.add(r.get("id", ""))
                kept += 1
            else:
                dropped.append(r)
            mark = {"ACT_ON": "✓", "CHECK": "~", "SET_ASIDE": "✗"}[r["tier"]]
            print(f"   {mark} [{r['trust']:.2f}] {r['finding'][:84]}")

        history.append({"cycle": cycle, "proposed": len(tasks), "kept": kept,
                        "dropped": len(results) - kept,
                        "searches": sum(1 for r in results if r.get("searched")),
                        "answered_from_world": sum(1 for r in results if not r.get("searched") and r.get("known"))})
        print(f"   kept {kept} of {len(results)} "
              f"{'(gate on)' if gate else '(gate OFF, nothing filtered)'}"
              f" · ${c.spent:.2f} spent")

    # The last word: one summary answer, built only from what survived.
    answer = ""
    if record:
        try:
            answer = c.ask(ORCHESTRATOR, (
                f"Research question: {question}\n\n"
                "Findings that survived checking:\n"
                + ((established_from(world, run_id, kept_ids) if world is not None else "")
                   or "\n".join(f"- {r['finding']}" for r in record[-12:]))
                + "\n\nWrite a 3-sentence answer using only these findings. "
                  "Say plainly where the evidence is thin."
            ), max_tokens=400)
        except Budget:
            answer = "(budget reached before the summary)"

    out = {"question": question, "gate": gate, "world": world is not None, "run_id": run_id,
           "cycles": history, "record": record, "dropped": dropped, "answer": answer,
           "calls": c.calls, "spend": round(c.spent, 4),
           "world_stats": world.stats().model_dump() if world is not None else None}
    path = HERE / f"loop_{'gated' if gate else 'ungated'}{'' if world is not None else '_noworld'}.json"
    path.write_text(json.dumps(out, indent=2))

    tiers = {t: sum(1 for r in record if r["tier"] == t)
             for t in ("ACT_ON", "CHECK", "SET_ASIDE")}
    print("\n" + "=" * 56)
    print(f"Gate {'ON' if gate else 'OFF'} · {len(record)} findings kept, "
          f"{len(dropped)} dropped · {c.calls} calls · ${c.spent:.2f}")
    print(f"Of what was kept: {tiers['ACT_ON']} act on it, {tiers['CHECK']} check "
          f"first, {tiers['SET_ASIDE']} set aside")
    if answer:
        print(f"\nAnswer:\n{answer}\n")
    print(f"Wrote {path}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--width", type=int, default=4, help="parallel tasks per cycle")
    ap.add_argument("--budget", type=float, default=4.0)
    ap.add_argument("--no-gate", action="store_true",
                    help="keep every finding, the way a system without a router works")
    ap.add_argument("--no-world", action="store_true",
                    help="the old shared record: a plain list, and a fresh paper search per task")
    a = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")
    print(f"{a.cycles} cycles x {a.width} parallel tasks, cap ${a.budget:.2f}")
    print(f"Papers: {'on, via Quorum' if LIT else 'off'}")
    run(a.question, a.cycles, a.width, a.budget, gate=not a.no_gate, use_world=not a.no_world)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
