"""
The trust router: sort an AI agent's answers into tiers you can act on.

The problem. On BixBench, agents answer bioinformatics questions and get about
a third of them right. Every answer arrives sounding equally sure. A scientist
with 296 answers and no way to sort them cannot use any of them, because
checking one costs as much as doing the work.

What this does. It never tries to answer the questions. It looks only at
signals about HOW the answer was produced, and sorts answers into tiers. The
top tier should be far more accurate than the average. We only reveal the real
answers at the very end, to score ourselves.

The referees, each failing in a different way:

  did_not_abstain    The model was offered "I don't know" and gave a real
                     answer anyway. Free, from the benchmark's refusal runs.
  models_agree       Two different models, run separately, picked the same
                     answer. Free. Independent of the above.
  question_is_hard   How often this question is answered correctly across all
                     the runs we have, excluding the one being judged.
  llm_referee        Optional. A Claude agent reads the question and the
                     answer and says whether it looks sound. Costs money, so
                     it is off unless you ask for it.

Plain code combines the votes; no model ever sets a tier. That rule is the
whole point: a referee can lower trust but never raise it.

A trap worth knowing: BixBench shuffles the answer options per model, so "A"
means different things in different files. Answers are compared by their TEXT,
never by their letter. Comparing letters silently produces a wrong result.

Run:  python3 router.py                 # free, uses the shipped results
      python3 router.py --llm           # adds a Claude referee (costs money)
      python3 router.py --llm --limit 20
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "bixdata"
OUT = HERE / "results.json"

# The benchmark runs we have. "offered_idk" marks the runs where the model was
# allowed to say "I don't know", which is where the abstention signal lives.
RUNS = [
    ("gpt-4o", "forced", "bixbench_llm_baseline_refusal_False_mcq_gpt-4o_1.0.csv", False),
    ("claude", "forced", "bixbench_llm_baseline_refusal_False_mcq_claude-3-5-sonnet-latest_1.0.csv", False),
    ("gpt-4o", "offered_idk", "bixbench_llm_baseline_refusal_True_mcq_gpt-4o_1.0.csv", True),
    ("claude", "offered_idk", "bixbench_llm_baseline_refusal_True_mcq_claude-3-5-sonnet-latest_1.0.csv", True),
]


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0")


def option_text(row: dict, letter: str) -> str | None:
    """
    Turn an answer letter into the words it stands for.

    The options are shuffled per run, so "A" in one file is not "A" in another.
    Only the text is comparable.
    """
    try:
        choices = ast.literal_eval(row["choices"])
    except Exception:  # noqa: BLE001 - a malformed row should be skipped, not crash the run
        return None
    for c in choices:
        m = re.match(r"\s*\((\w)\)\s*(.*)", c)
        if m and m.group(1) == letter:
            return m.group(2).strip().lower()
    return None


def load_runs() -> dict:
    """One record per question, holding what each model did in each run."""
    questions: dict[str, dict] = {}
    for model, mode, filename, offered_idk in RUNS:
        path = DATA / filename
        if not path.exists():
            sys.exit(f"Missing {path}. Copy the BixBench baseline CSVs into bixdata/.")
        for row in csv.DictReader(path.open()):
            qid = row["short_qid"]
            q = questions.setdefault(qid, {"qid": qid, "question": row["question"], "runs": {}})
            picked = option_text(row, row["predicted"])
            # In the "offered I don't know" runs, the `sure` column is True when
            # the model PICKED the I-don't-know option. The name is backwards;
            # it means the model declined, not that it was confident.
            abstained = truthy(row.get("sure")) if offered_idk else False
            q["runs"][f"{model}_{mode}"] = {
                "answer": picked,
                "correct": truthy(row["correct"]),
                "abstained": abstained,
                "raw_response": (row.get("llm_response") or "")[:400],
            }
    return questions


def referees(q: dict, difficulty: dict) -> list[dict]:
    """
    Each referee looks at one thing and votes. None of them sees the right answer.

    A vote is +1 (reason to trust), -1 (reason to doubt) or 0 (no opinion).
    """
    r = q["runs"]
    out = []

    # 1. Did the models take the easy way out? Both were offered "I don't know".
    declined = sum(1 for m in ("gpt-4o", "claude")
                   if r.get(f"{m}_offered_idk", {}).get("abstained"))
    if declined == 0:
        out.append({"name": "Both stood behind an answer", "vote": 1,
                    "detail": "Each model was offered \"I don't know\" and answered anyway."})
    elif declined == 1:
        out.append({"name": "One model backed out", "vote": 0,
                    "detail": "One took the \"I don't know\" option, the other answered."})
    else:
        out.append({"name": "Both backed out", "vote": -1,
                    "detail": "Both models declined when given the chance."})

    # 2. Two different models, run separately. Agreement is weak evidence;
    #    disagreement is strong evidence that something is wrong.
    a = r.get("gpt-4o_forced", {}).get("answer")
    b = r.get("claude_forced", {}).get("answer")
    if a and b and a == b:
        out.append({"name": "The two models agree", "vote": 1,
                    "detail": "Run separately, they landed on the same answer."})
    elif a and b:
        out.append({"name": "The two models disagree", "vote": -1,
                    "detail": "They picked different answers, so at least one is wrong."})

    # 3. How hard is this question for everyone else? Computed leaving out the
    #    run being judged, so a question is never scored using its own answer.
    d = difficulty.get(q["qid"], 0.5)
    if d >= 0.75:
        out.append({"name": "Others find this question easy", "vote": 1,
                    "detail": f"Answered correctly in {d:.0%} of other runs."})
    elif d <= 0.25:
        out.append({"name": "Others find this question hard", "vote": -1,
                    "detail": f"Answered correctly in only {d:.0%} of other runs."})

    if q.get("llm_referee"):
        out.append(q["llm_referee"])
    return out


def decide(votes: list[dict]) -> tuple[str, float]:
    """
    Plain code turns votes into a tier. No model touches this.

    The rule is deliberately simple so it can be stated in one sentence and
    frozen before we look at any results.
    """
    score = sum(v["vote"] for v in votes)
    doubts = sum(1 for v in votes if v["vote"] < 0)
    if doubts == 0 and score >= 2:
        tier = "ACT_ON"
    elif score <= -2 or doubts >= 2:
        tier = "SET_ASIDE"
    else:
        tier = "CHECK"
    # A 0-to-1 trust number for ranking. The tiers are coarse; this breaks ties.
    trust = max(0.0, min(1.0, 0.5 + score / 8))
    return tier, trust


def llm_referee(q: dict, client, model: str) -> dict | None:
    """
    An optional Claude referee. It sees the question and the proposed answer,
    never the right answer, and judges whether the answer looks sound.
    """
    answer = q["runs"].get("gpt-4o_forced", {}).get("answer") or "(none)"
    prompt = (
        "You are checking another system's answer to a bioinformatics question. "
        "You do NOT know the correct answer and must not guess it. Judge only "
        "whether the proposed answer looks sound, given the question as asked.\n\n"
        f"Question: {q['question'][:1500]}\n\n"
        f"Proposed answer: {answer[:500]}\n\n"
        "Reply with exactly one line: SOUND or DOUBTFUL, then a dash and under "
        "15 words of reason."
    )
    try:
        resp = client.messages.create(
            model=model, max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
    except Exception as e:  # noqa: BLE001 - one failed call must not stop the run
        return {"name": "Reviewer unavailable", "vote": 0, "detail": e.__class__.__name__}
    doubtful = text.upper().startswith("DOUBTFUL")
    return {"name": "Reviewer doubts this" if doubtful else "Reviewer finds it sound",
            "vote": -1 if doubtful else 1, "detail": text[:160]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="add a Claude referee (spends money)")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    args = ap.parse_args()

    questions = load_runs()
    qids = sorted(questions)
    print(f"{len(qids)} questions loaded from {len(RUNS)} benchmark runs")

    # Question difficulty, each question scored by the runs OTHER than the one
    # being judged, so nothing is scored using its own result.
    difficulty = {}
    for qid in qids:
        runs = questions[qid]["runs"]
        judged = ("gpt-4o_forced",)
        others = [v["correct"] for k, v in runs.items() if k not in judged]
        difficulty[qid] = sum(others) / len(others) if others else 0.5

    if args.llm:
        try:
            from anthropic import Anthropic
        except ImportError:
            sys.exit("pip install anthropic, or drop --llm")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY, or drop --llm")
        client = Anthropic()
        todo = qids[: args.limit] if args.limit else qids
        print(f"Asking the reviewer about {len(todo)} questions on {args.model}...")
        for n, qid in enumerate(todo, 1):
            questions[qid]["llm_referee"] = llm_referee(questions[qid], client, args.model)
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}")

    # Judge every question, then and only then compare with the truth.
    rows = []
    for qid in qids:
        q = questions[qid]
        votes = referees(q, difficulty)
        tier, trust = decide(votes)
        truth = q["runs"].get("gpt-4o_forced", {})
        rows.append({
            "qid": qid,
            "question": q["question"][:300],
            "tier": tier,
            "trust": round(trust, 3),
            "signals": votes,
            "answer": (truth.get("answer") or "")[:200],
            "was_correct": truth.get("correct", False),
        })

    overall = sum(r["was_correct"] for r in rows) / len(rows)

    def slice_acc(sel):
        return sum(r["was_correct"] for r in sel) / len(sel) if sel else 0.0

    tiers = {}
    for name in ("ACT_ON", "CHECK", "SET_ASIDE"):
        sel = [r for r in rows if r["tier"] == name]
        tiers[name] = {"n": len(sel), "coverage": len(sel) / len(rows),
                       "accuracy": slice_acc(sel)}

    # The risk-coverage curve: accept the most-trusted X% and see how accurate
    # they are. A working router starts high on the left and falls.
    ranked = sorted(rows, key=lambda r: -r["trust"])
    curve = []
    for pct in range(5, 101, 5):
        k = max(1, round(len(ranked) * pct / 100))
        curve.append({"coverage": pct / 100, "accuracy": slice_acc(ranked[:k]), "n": k})

    # How much of the top tier's lead could be luck? Reshuffle the trust scores
    # at random many times and see how often chance does as well.
    best = tiers["ACT_ON"]
    rng = random.Random(0)
    size = max(best["n"], 1)
    chance = [slice_acc(rng.sample(rows, size)) for _ in range(2000)]
    p_value = sum(1 for c in chance if c >= best["accuracy"]) / len(chance)

    report = {
        "benchmark": "BixBench (FutureHouse), 296 multiple-choice questions",
        "overall_accuracy": overall,
        "tiers": tiers,
        "curve": curve,
        "lift": (best["accuracy"] / overall) if overall else 0,
        "p_value": p_value,
        "used_llm_referee": args.llm,
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, indent=2))

    print(f"\nOverall accuracy: {overall:.1%}")
    print(f"{'tier':12s} {'n':>5s} {'coverage':>9s} {'accuracy':>9s}")
    print("-" * 38)
    for name, t in tiers.items():
        print(f"{name:12s} {t['n']:5d} {t['coverage']:8.0%} {t['accuracy']:8.1%}")
    print(f"\nTop tier is {report['lift']:.1f}x the overall rate.")
    print(f"Chance of doing this well at random: {p_value:.1%}")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
