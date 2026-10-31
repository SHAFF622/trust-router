"""
Pull the OpenFE protein-ligand benchmark into one clean table.

Why this exists: the benchmark keeps each target's measured binding strengths in
its own YAML file, in whatever units the original paper used (micromolar Ki,
nanomolar IC50, kcal/mol). You can't compare a predictor against a mix like that,
so this turns everything into one number on one scale.

The scale is pAffinity = -log10(concentration in molar). Bigger means binds
tighter. A 1 nM binder is 9, a 1 uM binder is 6. Each step of 1 is a tenfold
difference in strength.

Run:  python3 fetch_benchmark.py
Out:  data/benchmark.csv  (one row per ligand) and data/<target>_protein.pdb
"""

from __future__ import annotations

import csv
import json
import math
import sys
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/OpenFreeEnergy/protein-ligand-benchmark/main"
OUT = Path(__file__).resolve().parent / "data"

# The four congeneric series the FEP+ papers report on. JNK1 is named in the
# repo's README table but its data folder isn't in this fork, so it's left out.
CORE = ["tyk2", "cdk2", "p38"]
# More targets, for when 66 ligands turn out to be too few to measure anything.
EXTRA = ["mcl1", "hif2a", "syk", "ptp1b", "thrombin", "tnks2", "shp2",
         "cdk8", "cmet", "eg5", "pde2", "pfkfb3"]

# How many molar units fit in one of theirs, so we can get everything to molar.
TO_MOLAR = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "nm": 1e-9, "pm": 1e-12}

# Gas constant times body-ish temperature, in kcal/mol. Used to turn a binding
# free energy into the same log scale as the concentrations.
RT = 0.001987 * 297.0


def fetch(path: str) -> str:
    with urllib.request.urlopen(f"{RAW}/{path}", timeout=60) as r:
        return r.read().decode("utf-8")


def parse_ligands(text: str) -> list[dict]:
    """
    Read the ligands YAML without needing a YAML library installed.

    The file has a fixed, simple shape: a ligand name at the left margin, then
    indented fields, then an indented "measurement" block. Hand-parsing it keeps
    this script dependency-free, which matters when you're racing a clock.
    """
    out, cur, in_measurement = [], None, False
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()

        if indent == 0 and line.endswith(":"):
            if cur:
                out.append(cur)
            cur = {"name": line[:-1]}
            in_measurement = False
            continue
        if cur is None:
            continue
        if line.rstrip(":") == "measurement" and line.endswith(":"):
            in_measurement = True
            continue
        if indent <= 2:
            in_measurement = False
        if ":" not in line:
            continue

        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip().strip("'\"")
        if in_measurement:
            cur[f"m_{key}"] = val
        elif key in ("smiles", "name"):
            cur[key] = val
    if cur:
        out.append(cur)
    return out


def to_log_scale(value: str, unit: str, kind: str) -> float | None:
    """One number on one scale, or None when we can't honestly convert it."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    kind, unit = (kind or "").lower(), (unit or "").lower()

    if kind in ("ki", "kd", "ic50", "ec50"):
        factor = TO_MOLAR.get(unit)
        if factor is None or v <= 0:
            return None
        return -math.log10(v * factor)
    if kind in ("pic50", "pki", "pkd"):
        return v
    if kind == "dg":
        # A binding free energy in kcal/mol. More negative means tighter, so the
        # sign flips on the way to the log scale.
        if "kj" in unit:
            v = v / 4.184
        return -v / (RT * math.log(10))
    return None


def main() -> int:
    targets = CORE + (EXTRA if "--all" in sys.argv else [])
    rows, skipped = [], 0

    for target in targets:
        try:
            text = fetch(f"data/{target}/00_data/ligands.yml")
        except Exception as e:  # noqa: BLE001 - one bad target shouldn't stop the rest
            print(f"  {target}: could not fetch ({e.__class__.__name__})")
            continue

        n_before = len(rows)
        for lig in parse_ligands(text):
            y = to_log_scale(lig.get("m_value"), lig.get("m_unit"), lig.get("m_type"))
            if y is None or not lig.get("smiles"):
                skipped += 1
                continue
            rows.append({
                "target": target,
                "ligand": lig["name"],
                "smiles": lig["smiles"],
                "p_affinity": round(y, 3),
                "raw_value": lig.get("m_value", ""),
                "raw_unit": lig.get("m_unit", ""),
                "raw_type": lig.get("m_type", ""),
                "error": lig.get("m_error", ""),
                "doi": lig.get("m_doi", ""),
            })
        print(f"  {target}: {len(rows) - n_before} ligands")

        try:
            pdb = fetch(f"data/{target}/01_protein/crd/protein.pdb")
            (OUT / f"{target}_protein.pdb").write_text(pdb)
        except Exception:  # noqa: BLE001 - the structure is optional for a first pass
            print(f"  {target}: no protein.pdb")

    if not rows:
        print("Nothing fetched. Check the network and try again.")
        return 1

    OUT.mkdir(exist_ok=True)
    csv_path = OUT / "benchmark.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # A tiny summary, so you can see at a glance whether a target has enough
    # ligands and enough spread to tell a good predictor from a bad one.
    summary = {}
    for t in {r["target"] for r in rows}:
        ys = [r["p_affinity"] for r in rows if r["target"] == t]
        mean = sum(ys) / len(ys)
        spread = (sum((y - mean) ** 2 for y in ys) / max(len(ys) - 1, 1)) ** 0.5
        summary[t] = {"n": len(ys), "spread": round(spread, 2),
                      "low": round(min(ys), 2), "high": round(max(ys), 2)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"\n{len(rows)} ligands -> {csv_path}  ({skipped} skipped)")
    for t in sorted(summary, key=lambda k: -summary[k]["n"]):
        s = summary[t]
        print(f"  {t:10s} n={s['n']:3d}  spread={s['spread']:.2f}  "
              f"range {s['low']} to {s['high']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
