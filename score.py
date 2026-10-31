"""
Open the golden answers and score the run.

This is the only place the correct answers are read. Everything before it ran
blind. The question it answers: were the answers the router trusted actually
the ones that turned out to be right?

Headline number: accuracy inside the "act on it" tier, against accuracy across
everything. A router that works makes the first much larger than the second.

Run:  python3 score.py
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "agent_runs.jsonl"
GOLDEN = HERE / "bixdata" / "golden.csv"
OUT = HERE / "results.json"
TIERS = ("ACT_ON", "CHECK", "SET_ASIDE")


def main() -> int:
    if not RUNS.exists():
        raise SystemExit("No agent_runs.jsonl. Run: python3 pipeline.py --limit 10")

    key = {r["case_id"]: r for r in csv.DictReader(GOLDEN.open())}
    rows = []
    for line in RUNS.open():
        if not line.strip():
            continue
        r = json.loads(line)
        g = key.get(r["case_id"])
        if not g:
            continue
        rows.append({**r, "correct_option": g["correct_option"],
                     "was_correct": r.get("answer_letter") == g["correct_option"]})

    if not rows:
        raise SystemExit("No scored rows.")

    acc = lambda s: sum(x["was_correct"] for x in s) / len(s) if s else 0.0
    overall = acc(rows)

    tiers = {}
    for t in TIERS:
        sel = [r for r in rows if r["tier"] == t]
        tiers[t] = {"n": len(sel), "coverage": len(sel) / len(rows), "accuracy": acc(sel)}

    ranked = sorted(rows, key=lambda r: -r["trust"])
    curve = []
    for pct in range(5, 101, 5):
        k = max(1, round(len(ranked) * pct / 100))
        curve.append({"coverage": pct / 100, "accuracy": acc(ranked[:k]), "n": k})

    # Could chance alone have produced a tier this good? Draw groups of the
    # same size at random and see how often they match it.
    best = tiers["ACT_ON"]
    rng = random.Random(0)
    size = max(best["n"], 1)
    draws = [acc(rng.sample(rows, size)) for _ in range(2000)]
    p = sum(1 for d in draws if d >= best["accuracy"]) / len(draws)

    report = {
        "benchmark": f"BixBench, {len(rows)} questions, agents run live",
        "overall_accuracy": overall,
        "tiers": tiers,
        "curve": curve,
        "lift": best["accuracy"] / overall if overall else 0,
        "p_value": p,
        "used_llm_referee": True,
        "rows": [{
            "qid": r["case_id"], "question": r["question"], "tier": r["tier"],
            "trust": r["trust"], "signals": r["votes"],
            "answer": r.get("answer_text", "")[:200], "was_correct": r["was_correct"],
        } for r in rows],
    }
    OUT.write_text(json.dumps(report, indent=2))

    print(f"{len(rows)} questions scored")
    print(f"Agents alone: {overall:.1%}\n")
    print(f"{'tier':12s} {'n':>5s} {'coverage':>9s} {'accuracy':>9s}")
    print("-" * 38)
    for t in TIERS:
        d = tiers[t]
        print(f"{t:12s} {d['n']:5d} {d['coverage']:8.0%} {d['accuracy']:8.1%}")
    print(f"\nTop tier is {report['lift']:.1f}x overall.")
    print(f"Chance of matching it at random: {p:.1%}")
    if best["n"] < 20:
        print(f"Careful: the top tier holds only {best['n']} answers, so that "
              "figure is noisy.")
    print(f"\nWrote {OUT}. Rebuild the page with:\n"
          "  python3 -c \"import json;t=open('ui_template.html').read();"
          "d=json.load(open('results.json'));d['rows']=sorted(d['rows'],"
          "key=lambda r:-r['trust']);open('index.html','w').write("
          "t.replace('__DATA__',json.dumps(d,separators=(',',':'))"
          ".replace('</','<\\\\/')))\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
