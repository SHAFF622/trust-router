"""
The cheap baseline: predict binding strength from the drug molecule alone.

This never looks at the protein. It turns each molecule into a long checklist of
yes/no flags about which small chemical patterns it contains, then learns which
patterns go with tight binding. It runs on a laptop in under a minute.

It answers three questions:

1. HOW HARD IS THIS BENCHMARK, REALLY?  If a laptop model does as well as a big
   structure model, the big model has to justify itself. This is the floor.

2. WHAT DOES ZERO-SHOT COST?  Boltz-2 is given no examples for a target. This
   script runs both settings so the comparison is honest:
     - "knows the target": trained on other ligands of the SAME target, which is
       what a chemist actually has mid-project. A strong, fair baseline.
     - "new target": trained only on OTHER targets. The true like-for-like
       comparison with a zero-shot predictor, and usually much worse.

3. WHEN IS A PREDICTION OUT ON A LIMB?  For every molecule it records how similar
   the closest training molecule was. A prediction about a molecule unlike
   anything seen before deserves less trust. That number is the first failure
   signal the referee panel can use.

Every correlation comes with an error bar, because with 10 to 44 ligands per
target a single number is easy to over-read.

Run:  .venv/bin/python baseline_fingerprint.py
Out:  data/baseline_predictions.csv  and a summary on screen
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

try:
    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.DataStructs import BulkTanimotoSimilarity
    from scipy.stats import pearsonr, spearmanr
    from sklearn.ensemble import RandomForestRegressor
except ImportError as e:
    sys.exit(f"Missing {e.name}. Run:\n"
             "  cd ~/scienceclaw-bench && uv venv && "
             "uv pip install rdkit scikit-learn numpy scipy")

RDLogger.DisableLog("rdApp.*")  # quiet the chemistry parser's chatter

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SEED = 0
BOOTSTRAPS = 2000


def load() -> list[dict]:
    with (DATA / "benchmark.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["p_affinity"] = float(r["p_affinity"])
    return rows


def fingerprints(rows: list[dict]):
    """Each molecule becomes a 2048-slot checklist of chemical patterns."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    keep, fps = [], []
    for r in rows:
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            continue  # a molecule the chemistry parser can't read
        keep.append(r)
        fps.append(gen.GetFingerprint(mol))
    X = np.array([list(fp) for fp in fps], dtype=np.uint8)
    return keep, fps, X


def ci(x: np.ndarray, y: np.ndarray, fn) -> tuple[float, float, float]:
    """
    A correlation plus a 95% range, by resampling.

    With this few points one correlation number is fragile: drop two molecules
    and it can swing badly. Resampling the data many times shows how much it
    could plausibly swing, which stops anyone over-reading a lucky result.
    """
    if len(x) < 4:
        return float("nan"), float("nan"), float("nan")
    point = fn(x, y)[0]
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, len(x), len(x))
        if len(set(y[idx])) < 3:
            continue  # a resample with no spread has no correlation to measure
        draws.append(fn(x[idx], y[idx])[0])
    if not draws:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi


def forest() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)


def main() -> int:
    if not (DATA / "benchmark.csv").exists():
        sys.exit("No data/benchmark.csv. Run: python3 fetch_benchmark.py --all")

    rows, fps, X = fingerprints(load())
    y = np.array([r["p_affinity"] for r in rows])
    targets = np.array([r["target"] for r in rows])
    print(f"{len(rows)} molecules, {len(set(targets))} targets\n")

    knows = np.full(len(rows), np.nan)   # trained on the same target
    newt = np.full(len(rows), np.nan)    # trained only on other targets
    nearest = np.zeros(len(rows))        # similarity to the closest training molecule

    for t in sorted(set(targets)):
        here = np.where(targets == t)[0]
        other = np.where(targets != t)[0]

        # Setting A: the model has seen other ligands for this same target.
        # Leave one out: predict each molecule from all the rest.
        if len(here) >= 5:
            for i in here:
                train = here[here != i]
                m = forest().fit(X[train], y[train])
                knows[i] = m.predict(X[i:i + 1])[0]
                sims = BulkTanimotoSimilarity(fps[i], [fps[j] for j in train])
                nearest[i] = max(sims)

        # Setting B: a target the model has never seen. Zero-shot, like Boltz-2.
        m = forest().fit(X[other], y[other])
        newt[here] = m.predict(X[here])

    out = DATA / "baseline_predictions.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "ligand", "measured", "pred_knows_target",
                    "pred_new_target", "nearest_similarity"])
        for i, r in enumerate(rows):
            w.writerow([r["target"], r["ligand"], round(y[i], 3),
                        "" if np.isnan(knows[i]) else round(knows[i], 3),
                        round(newt[i], 3), round(nearest[i], 3)])

    print(f"{'target':10s} {'n':>4s}  {'knows target':>22s}  {'new target':>22s}")
    print("-" * 64)
    for t in sorted(set(targets), key=lambda t: -int((targets == t).sum())):
        m = targets == t
        ok = m & ~np.isnan(knows)
        a = ci(knows[ok], y[ok], spearmanr) if ok.sum() >= 5 else (float("nan"),) * 3
        b = ci(newt[m], y[m], spearmanr)
        fmt = lambda v: ("  n/a" if np.isnan(v[0])
                         else f"{v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}]")
        print(f"{t:10s} {m.sum():4d}  {fmt(a):>22s}  {fmt(b):>22s}")

    ok = ~np.isnan(knows)
    pa = ci(knows[ok], y[ok], pearsonr)
    pb = ci(newt, y, pearsonr)
    print("-" * 64)
    print(f"Pooled Pearson, knows target : {pa[0]:+.3f} [{pa[1]:+.3f}, {pa[2]:+.3f}]")
    print(f"Pooled Pearson, new target   : {pb[0]:+.3f} [{pb[1]:+.3f}, {pb[2]:+.3f}]")
    print(f"\nWrote {out}")

    print("\nHow to read this:")
    print("  'knows target' is the number Boltz-2 has to beat to be worth its cost.")
    print("  'new target' is the fair zero-shot comparison; expect it to be poor.")
    print("  Where the two brackets overlap, you cannot yet tell the setups apart.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
