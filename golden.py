"""
Build the golden dataset: the questions, and the answers nobody gets to see.

A trust router only means something if the right answers are locked away while
it works. So this splits the benchmark in two:

  golden.csv    the questions, the options, and the correct answer. This is the
                scoring key. The agents never receive it.
  agent_input.csv  what the agents actually get. Same questions, no answers.

Everything that could let an agent look the answer up is stripped from the
agent's copy: the correct option, the grading columns, and anything naming the
source. Option order is also re-shuffled with a fixed seed, so an agent cannot
learn that the answer tends to sit in a particular position.

There is also a held-out split. Half the questions are marked "dev" for tuning
the router, half "test" for the number we report. Tuning on the same questions
you report would mean reporting how well we fitted 296 examples.

Run:  python3 golden.py
"""

from __future__ import annotations

import ast
import csv
import hashlib
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "bixdata"
SOURCE = DATA / "bixbench_llm_baseline_refusal_False_mcq_gpt-4o_1.0.csv"
SEED = 7


def options(row: dict) -> list[tuple[str, str]]:
    """The answer options as (letter, text) pairs."""
    try:
        raw = ast.literal_eval(row["choices"])
    except Exception:  # noqa: BLE001 - a malformed row is skipped, not fatal
        return []
    out = []
    for c in raw:
        m = re.match(r"\s*\((\w)\)\s*(.*)", c)
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(f"Missing {SOURCE}")

    rows = list(csv.DictReader(SOURCE.open()))
    rng = random.Random(SEED)
    golden, agent = [], []
    skipped = 0

    for row in rows:
        opts = options(row)
        correct = next((t for L, t in opts if L == row["target"]), None)
        if not opts or correct is None:
            skipped += 1
            continue

        # Re-letter the options in a fresh random order, so position carries
        # no information and the agent's copy cannot be matched back by shape.
        shuffled = opts[:]
        rng.shuffle(shuffled)
        letters = [chr(ord("A") + i) for i in range(len(shuffled))]
        relettered = [(L, text) for L, (_, text) in zip(letters, shuffled)]
        new_target = next(L for L, text in relettered if text == correct)

        # A stable, meaningless id. The original ids encode the capsule number,
        # which is a thread back to the source material.
        case_id = "case-" + hashlib.sha256(
            (row["short_qid"] + "golden").encode()).hexdigest()[:10]

        shared = {
            "case_id": case_id,
            "question": row["question"].strip(),
            **{f"option_{L}": text for L, text in relettered},
        }
        # Half for tuning, half for the number we report. Split by a hash of the
        # id, so the same question always lands in the same half.
        split = "test" if int(case_id[-4:], 16) % 2 else "dev"

        agent.append({**shared, "split": split})
        golden.append({**shared, "split": split,
                       "correct_option": new_target,
                       "correct_text": correct})

    cols_agent = ["case_id", "split", "question"] + \
        sorted({k for r in agent for k in r if k.startswith("option_")})
    cols_golden = cols_agent + ["correct_option", "correct_text"]

    for path, data, cols in ((DATA / "agent_input.csv", agent, cols_agent),
                             (DATA / "golden.csv", golden, cols_golden)):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(data)

    n_test = sum(1 for r in agent if r["split"] == "test")
    print(f"{len(agent)} questions ({skipped} skipped for malformed options)")
    print(f"  dev  {len(agent) - n_test}  for tuning the router")
    print(f"  test {n_test}  for the number we report")
    print(f"\n  {DATA / 'agent_input.csv'}   <- give this to the agents")
    print(f"  {DATA / 'golden.csv'}        <- scoring key, keep it back")

    # Prove the agent's copy really has no answers in it.
    leaks = [c for c in cols_agent if "correct" in c or "target" in c or "answer" in c]
    print(f"\nColumns in the agent's copy that could leak an answer: "
          f"{leaks if leaks else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
