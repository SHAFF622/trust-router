"""
Point the trust router at real open-source Kosmos output.

Kosmos (github.com/jimmc414/Kosmos) runs research cycles and writes each
cycle's hypotheses to artifacts/<run>/cycle_N/hypotheses.json, as records of
{id, statement, rationale, domain}. It does not say which of them to believe.
That is the gap this fills.

Two things are being kept apart here, and the difference matters:

  BixBench is where the router is MEASURED. Every question has a known right
  answer, so we can check whether the answers it trusted were the correct ones.

  Kosmos output is where the router is APPLIED. These are fresh hypotheses
  about the world; nobody knows which are true. We can sort them, but we
  cannot score the sorting. Any claim that we validated Kosmos would be false.

The referees here read the hypothesis as written. None of them has data to
check it against, so they judge how a careful reviewer would at this stage:
is the claim specific enough to test, does the stated reasoning actually
support it, and is it hedged into saying nothing.

Run:  python3 kosmos_adapter.py --runs ~/ai-scientist/kosmos-community/artifacts
      python3 kosmos_adapter.py --runs <dir> --llm      # adds a Claude reviewer
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "kosmos_tiered.json"

# Words that turn a testable claim into one that cannot be wrong.
HEDGES = ("may", "might", "could", "possibly", "potentially", "suggests",
          "appears to", "it is likely", "some", "various", "certain")
# Signs that a claim names something you could actually measure.
MEASURABLE = re.compile(
    r"\d|\b(increase|decrease|higher|lower|greater|reduce|correlat|"
    r"fold|percent|%|p-value|significant|threshold|rate|ratio)\w*", re.I)


def load_hypotheses(root: Path) -> list[dict]:
    """Every hypothesis Kosmos wrote, with the run and cycle it came from."""
    out = []
    for path in sorted(root.glob("**/cycle_*/hypotheses.json")):
        try:
            items = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001 - skip an unreadable file, keep going
            print(f"  skipped {path.name}: {e.__class__.__name__}")
            continue
        cycle = path.parent.name
        run = path.parent.parent.name
        for h in items if isinstance(items, list) else []:
            out.append({
                "id": h.get("id", "")[:12],
                "run": run,
                "cycle": cycle,
                "statement": (h.get("statement") or "").strip(),
                "rationale": (h.get("rationale") or "").strip(),
                "domain": h.get("domain", ""),
            })
    return out


def referees(h: dict, seen_before: dict) -> list[dict]:
    """
    Each referee checks one property of the claim as written. None of them can
    tell whether it is true; they judge whether it is worth the bench time to
    find out, which is the decision a scientist is actually making.
    """
    votes = []
    s = h["statement"]

    # 1. Can it be measured? A claim with no quantity, direction or comparison
    #    cannot be shown wrong, so it cannot be shown right either.
    if MEASURABLE.search(s):
        votes.append({"name": "States something measurable", "vote": 1,
                      "detail": "Names a direction, quantity or comparison you could test."})
    else:
        votes.append({"name": "Nothing measurable stated", "vote": -1,
                      "detail": "No quantity, direction or comparison, so no experiment settles it."})

    # 2. Is it hedged into safety? Three hedges in one sentence usually means
    #    the claim has been written so it cannot turn out wrong.
    hedges = [w for w in HEDGES if w in s.lower()]
    if len(hedges) >= 3:
        votes.append({"name": "Hedged until it says nothing", "vote": -1,
                      "detail": f"Hedging words: {', '.join(hedges[:4])}."})
    elif not hedges:
        votes.append({"name": "Stated plainly", "vote": 1,
                      "detail": "Committed to a definite claim rather than hedging."})

    # 3. Does the reasoning do any work? A rationale shorter than the claim is
    #    usually a restatement of it.
    if len(h["rationale"]) < max(len(s), 80):
        votes.append({"name": "Reasoning restates the claim", "vote": -1,
                      "detail": "The rationale is no longer than the claim it supports."})
    else:
        votes.append({"name": "Reasoning adds support", "vote": 1,
                      "detail": f"{len(h['rationale'])} characters of separate argument."})

    # 4. Did independent cycles land on the same idea? Kosmos runs cycles that
    #    do not see each other, so a repeat is weak corroboration.
    key = tuple(sorted(w for w in re.findall(r"[a-z]{5,}", s.lower())))[:8]
    others = seen_before.get(key, 0)
    if others > 1:
        votes.append({"name": "Found in more than one cycle", "vote": 1,
                      "detail": f"A near-identical claim appears in {others} cycles."})

    if h.get("llm_vote"):
        votes.append(h["llm_vote"])
    return votes


def route(votes: list[dict]) -> tuple[str, float]:
    """Plain code. A referee can lower trust, never raise it past the evidence."""
    score = sum(v["vote"] for v in votes)
    doubts = sum(1 for v in votes if v["vote"] < 0)
    tier = ("ACT_ON" if doubts == 0 and score >= 2
            else "SET_ASIDE" if doubts >= 2
            else "CHECK")
    return tier, max(0.0, min(1.0, 0.5 + score / 8))


def llm_review(h: dict, client, model: str) -> dict:
    prompt = (
        "A research system proposed this hypothesis. You cannot check whether "
        "it is true and must not try. Judge only whether it is worth a "
        "scientist's time: is it specific enough to test, and does the stated "
        "reasoning support it?\n\n"
        f"Hypothesis: {h['statement'][:900]}\n\n"
        f"Its reasoning: {h['rationale'][:900]}\n\n"
        "Reply with exactly WORTH_TESTING or TOO_VAGUE, a dash, and under 15 words."
    )
    try:
        r = client.messages.create(model=model, max_tokens=120,
                                   messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in r.content if b.type == "text").strip()
    except Exception as e:  # noqa: BLE001 - a failed call is no opinion, not a crash
        return {"name": "Reviewer unavailable", "vote": 0, "detail": e.__class__.__name__}
    vague = "TOO_VAGUE" in text.upper()
    return {"name": "Reviewer: too vague to test" if vague else "Reviewer: worth testing",
            "vote": -1 if vague else 1, "detail": text[:150]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="Kosmos artifacts directory")
    ap.add_argument("--llm", action="store_true", help="add a Claude reviewer (costs money)")
    ap.add_argument("--model", default="claude-opus-5-5")
    args = ap.parse_args()

    root = Path(os.path.expanduser(args.runs))
    if not root.exists():
        sys.exit(f"No such directory: {root}")

    hyps = load_hypotheses(root)
    if not hyps:
        sys.exit(f"No cycle_*/hypotheses.json under {root}. Has Kosmos run yet?")
    print(f"{len(hyps)} hypotheses from {len({h['run'] for h in hyps})} Kosmos runs\n")

    # How often each idea recurs across cycles that could not see each other.
    seen: dict = {}
    for h in hyps:
        key = tuple(sorted(w for w in re.findall(r"[a-z]{5,}", h["statement"].lower())))[:8]
        seen[key] = seen.get(key, 0) + 1

    if args.llm:
        try:
            from anthropic import Anthropic
        except ImportError:
            sys.exit("pip install anthropic, or drop --llm")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY, or drop --llm")
        client = Anthropic()
        print(f"Reviewing {len(hyps)} hypotheses on {args.model}...")
        for h in hyps:
            h["llm_vote"] = llm_review(h, client, args.model)

    rows = []
    for h in hyps:
        votes = referees(h, seen)
        tier, trust = route(votes)
        rows.append({**{k: h[k] for k in ("id", "run", "cycle", "statement", "domain")},
                     "tier": tier, "trust": round(trust, 3), "signals": votes})

    rows.sort(key=lambda r: -r["trust"])
    counts = {t: sum(1 for r in rows if r["tier"] == t)
              for t in ("ACT_ON", "CHECK", "SET_ASIDE")}
    OUT.write_text(json.dumps({"source": str(root), "n": len(rows),
                               "counts": counts, "rows": rows}, indent=2))

    label = {"ACT_ON": "Worth the bench time", "CHECK": "Look closer first",
             "SET_ASIDE": "Not testable as written"}
    for t, n in counts.items():
        print(f"{label[t]:26s} {n:3d}")
    print(f"\nTop-ranked hypotheses:")
    for r in rows[:5]:
        print(f"  [{r['trust']:.2f}] {r['statement'][:88]}")
    print(f"\nWrote {OUT}")
    print("\nNote: these are fresh hypotheses with no known answer, so this sorts "
          "them but cannot score the sorting. The accuracy numbers come from "
          "BixBench, where the right answers exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
