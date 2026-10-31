"""
The full system: sub-agents answer the questions, a panel judges the answers,
and plain code sorts them into tiers.

Four agents, deliberately given different information so they fail differently.
Four signals that fail together are one signal wearing four hats.

  Analyst        Answers the question. Run twice at different temperatures;
                 whether the two runs agree is the first signal.
  Librarian      Answers the same question from published knowledge only,
                 without seeing the Analyst's answer. A second opinion from a
                 different angle.
  Critic         Sees the question and the Analyst's answer, never the
                 Librarian's and never the truth. Judges whether the reasoning
                 holds up on its own terms.
  Router         Plain code. Turns the votes into a tier. No model touches it.

What nobody sees: the correct answer. It lives in golden.csv and is opened
only at the end, to score the run.

Cost control, because this spends real money:
  - a hard dollar cap, checked before every call
  - every call logged with its token counts and cost
  - resumable: finished cases are skipped on a re-run
  - --limit for a cheap trial before committing

Run:  export ANTHROPIC_API_KEY=...
      python3 pipeline.py --limit 10 --budget 2      # trial, about $1
      python3 pipeline.py --split dev --budget 25    # the real run
      python3 score.py                               # then score it
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "bixdata"
OUT = HERE / "agent_runs.jsonl"
LOG = HERE / "llm_calls.jsonl"

# Dollars per million tokens. Sonnet handles the answering; Opus judges,
# because judging is where the harder reasoning sits and it is fewer calls.
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
ANSWERER = "claude-sonnet-5-5"
JUDGE = "claude-opus-5-5"


class Budget(RuntimeError):
    pass


class Client:
    """A thin wrapper that will not let the run overspend."""

    def __init__(self, cap_usd: float):
        from anthropic import Anthropic
        self.api = Anthropic()
        self.cap = cap_usd
        self.spent = 0.0
        self.calls = 0

    def ask(self, model: str, prompt: str, role: str, max_tokens: int = 400) -> str:
        if self.spent >= self.cap:
            raise Budget(f"Spent ${self.spent:.2f} of ${self.cap:.2f}.")
        t0 = time.time()
        resp = self.api.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        pin, pout = PRICES.get(model, PRICES[JUDGE])
        cost = (resp.usage.input_tokens * pin + resp.usage.output_tokens * pout) / 1e6
        self.spent += cost
        self.calls += 1
        with LOG.open("a") as f:
            f.write(json.dumps({
                "role": role, "model": model,
                "in": resp.usage.input_tokens, "out": resp.usage.output_tokens,
                "cost_usd": round(cost, 6), "seconds": round(time.time() - t0, 2),
            }) + "\n")
        return "".join(b.text for b in resp.content if b.type == "text").strip()


def options_of(case: dict) -> list[tuple[str, str]]:
    return [(k.split("_")[1], v) for k, v in sorted(case.items())
            if k.startswith("option_") and v]


def question_block(case: dict) -> str:
    opts = "\n".join(f"  ({L}) {t}" for L, t in options_of(case))
    return f"{case['question'].strip()}\n\n{opts}"


def pick_letter(text: str, case: dict) -> str | None:
    """Read a single letter out of a model's reply."""
    valid = {L for L, _ in options_of(case)}
    m = re.search(r"\b([A-H])\b", text.upper())
    return m.group(1) if m and m.group(1) in valid else None


# --- the agents -------------------------------------------------------------

def analyst(c: Client, case: dict, careful: bool) -> dict:
    """
    Answers the question. Run twice: once quickly, once told to weigh the
    options. If two passes over the same question land differently, that is
    a sign the question is not pinned down by what the agent knows.
    """
    style = ("Think through each option before deciding." if careful
             else "Answer directly.")
    text = c.ask(ANSWERER, (
        "You are a computational biologist answering a question about a "
        f"published bioinformatics analysis.\n\n{question_block(case)}\n\n"
        f"{style} Reply with the letter, a dash, and under 20 words of reason."
    ), role="analyst_careful" if careful else "analyst_quick")
    return {"letter": pick_letter(text, case), "text": text[:200]}


def librarian(c: Client, case: dict) -> dict:
    """
    Answers the same question from general published knowledge, with no sight
    of the Analyst. Where it lands on the same option by a different route,
    that is genuine corroboration.
    """
    text = c.ask(ANSWERER, (
        "Using only what is standard in the published bioinformatics "
        "literature, which option is most likely correct? Do not guess at "
        "what a specific unseen analysis found; reason from what is usual.\n\n"
        f"{question_block(case)}\n\n"
        "Reply with the letter, a dash, and under 20 words of reason."
    ), role="librarian")
    return {"letter": pick_letter(text, case), "text": text[:200]}


def critic(c: Client, case: dict, answer: str) -> dict:
    """
    Reads the question and the proposed answer, nothing else. Never told what
    the other agents said, and never told the truth.
    """
    text = c.ask(JUDGE, (
        "Another system answered a bioinformatics question. You do not know "
        "the correct answer and must not try to work it out. Judge only "
        "whether this answer is well-supported by the question as written, or "
        "whether it looks like a guess.\n\n"
        f"{question_block(case)}\n\n"
        f"Its answer: {answer}\n\n"
        "Reply with exactly SOUND or SHAKY, a dash, and under 15 words."
    ), role="critic", max_tokens=120)
    shaky = text.upper().lstrip().startswith("SHAKY")
    return {"shaky": shaky, "text": text[:200]}


# --- the router -------------------------------------------------------------

def route(signals: dict) -> tuple[str, float, list[dict]]:
    """
    Plain code. The rule is frozen before any result is looked at, and a
    signal can lower trust but never raise it above what the evidence allows.
    """
    votes = []
    quick, careful = signals["quick"], signals["careful"]
    lib, shaky = signals["librarian"], signals["critic_shaky"]

    if quick and careful and quick == careful:
        votes.append({"name": "Answered the same way twice", "vote": 1,
                      "detail": "Two passes over the question agreed."})
    else:
        votes.append({"name": "Changed its answer on rerun", "vote": -1,
                      "detail": "Two passes over the same question disagreed."})

    if lib and careful and lib == careful:
        votes.append({"name": "Literature view agrees", "vote": 1,
                      "detail": "An agent reasoning only from published work picked the same option."})
    elif lib:
        votes.append({"name": "Literature view differs", "vote": -1,
                      "detail": "An agent reasoning from published work picked a different option."})

    if shaky:
        votes.append({"name": "Reviewer calls it shaky", "vote": -1,
                      "detail": "An independent reviewer judged the answer poorly supported."})
    else:
        votes.append({"name": "Reviewer finds it sound", "vote": 1,
                      "detail": "An independent reviewer found the answer well supported."})

    score = sum(v["vote"] for v in votes)
    doubts = sum(1 for v in votes if v["vote"] < 0)
    tier = ("ACT_ON" if doubts == 0 and score >= 2
            else "SET_ASIDE" if doubts >= 2
            else "CHECK")
    return tier, max(0.0, min(1.0, 0.5 + score / 6)), votes


def done_ids() -> set[str]:
    if not OUT.exists():
        return set()
    return {json.loads(l)["case_id"] for l in OUT.open() if l.strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=float, default=5.0, help="hard dollar cap")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")
    src = DATA / "agent_input.csv"
    if not src.exists():
        sys.exit("Run: python3 golden.py")

    cases = [r for r in csv.DictReader(src.open())
             if args.split == "all" or r["split"] == args.split]
    already = done_ids()
    todo = [c for c in cases if c["case_id"] not in already]
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(cases)} cases in '{args.split}', {len(already)} already done, "
          f"{len(todo)} to run")
    print(f"4 calls each, cap ${args.budget:.2f}. Answers stay hidden until scoring.\n")

    c = Client(args.budget)
    for n, case in enumerate(todo, 1):
        try:
            quick = analyst(c, case, careful=False)
            careful = analyst(c, case, careful=True)
            lib = librarian(c, case)
            crit = critic(c, case, careful["text"])
        except Budget as e:
            print(f"\nStopped: {e}")
            break
        except Exception as e:  # noqa: BLE001 - one bad case must not end the run
            print(f"[{n}/{len(todo)}] {case['case_id']}: {e.__class__.__name__}")
            continue

        signals = {"quick": quick["letter"], "careful": careful["letter"],
                   "librarian": lib["letter"], "critic_shaky": crit["shaky"]}
        tier, trust, votes = route(signals)
        with OUT.open("a") as f:
            f.write(json.dumps({
                "case_id": case["case_id"], "split": case["split"],
                "question": case["question"][:300],
                "answer_letter": careful["letter"], "answer_text": careful["text"],
                "signals": signals, "tier": tier, "trust": trust, "votes": votes,
            }) + "\n")
        print(f"[{n}/{len(todo)}] {tier:10s} trust {trust:.2f}  "
              f"${c.spent:.2f} spent")

    print(f"\n{c.calls} calls, ${c.spent:.2f} spent. Wrote {OUT}")
    print("Now run: python3 score.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
