"""
Run Boltz-2 over the benchmark and record everything the referee panel needs.

The point is not the prediction on its own. It is the prediction PLUS every
signal that might reveal when it is wrong:

  - the predicted binding strength
  - the gap between Boltz-2's two internal ensemble members, which is free
    disagreement: when its own two halves don't agree, that is a warning
  - the spread across several predicted poses of the same complex
  - the model's own confidence scores for the best pose
  - how long it took and whether it failed

Scale note, easy to get wrong: Boltz-2 reports log10 of IC50 in micromolar,
where LOWER means stronger binding. Our measured values use the opposite
convention, where HIGHER means stronger. The conversion is pAffinity = 6 - y.
Getting this backwards would flip every correlation, so it is done in one place
(`to_p_affinity`) and nowhere else.

An MSA is a search for related protein sequences. It depends only on the
protein, so it is computed once per target and reused for all that target's
molecules: 15 searches instead of 370.

Safe to stop and restart. Finished molecules are skipped.

Run:   .venv/bin/python run_boltz.py --limit 5        # small trial first
       .venv/bin/python run_boltz.py --target tyk2    # one target
       .venv/bin/python run_boltz.py                  # everything
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
WORK = HERE / "runs"
RESULTS = DATA / "boltz_predictions.csv"

# Boltz-2 refuses ligands above 128 atoms and warns above ~56. We count the
# atoms we can see in the SMILES string and skip anything clearly too big,
# rather than burning a GPU slot to be told no.
MAX_HEAVY_ATOMS = 120

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "HID": "H", "HIE": "H", "HIP": "H",
    "CYX": "C", "ASH": "D", "GLH": "E", "LYN": "K",
}

FIELDS = [
    "target", "ligand", "measured_p_affinity",
    "pred_p_affinity",            # converted, comparable to measured
    "raw_affinity_pred_value",    # exactly as Boltz-2 reported it
    "ensemble_gap",               # |member 1 - member 2|: free disagreement
    "binder_probability",
    "confidence_score", "complex_plddt", "iptm", "ligand_iptm", "ptm",
    "pose_spread",                # spread of confidence across poses
    "n_poses", "seconds", "status", "note",
]


def sequence_from_pdb(path: Path) -> str:
    """
    Read the protein's chain as one-letter codes.

    Walks the backbone alpha-carbons in order, one per residue, taking only the
    first chain. Boltz-2 wants the sequence, not the coordinates.
    """
    seq, seen, chain = [], set(), None
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        this_chain = line[21]
        if chain is None:
            chain = this_chain
        if this_chain != chain:
            break
        key = (this_chain, line[22:27])  # residue number plus insertion code
        if key in seen:
            continue
        seen.add(key)
        seq.append(THREE_TO_ONE.get(line[17:20].strip().upper(), "X"))
    return "".join(seq)


def rough_atom_count(smiles: str) -> int:
    """Heavy atoms, counted off the SMILES text. Close enough to screen on."""
    n, i = 0, 0
    while i < len(smiles):
        c = smiles[i]
        if c == "[":
            j = smiles.find("]", i)
            inside = smiles[i + 1:j if j > 0 else len(smiles)]
            if not inside.startswith("H"):
                n += 1
            i = j + 1 if j > 0 else len(smiles)
            continue
        if c.isalpha():
            if c in "BCNOPSFIbcnops":
                n += 1
            elif smiles[i:i + 2] in ("Cl", "Br"):
                n += 1
                i += 1
        i += 1
    return n


def to_p_affinity(raw: float) -> float:
    """
    Boltz-2's scale to ours.

    It reports log10(IC50 in micromolar); lower is tighter. We use
    -log10(concentration in molar); higher is tighter. One micromolar is 6 on
    our scale, so the two are related by pAffinity = 6 - raw.
    """
    return 6.0 - raw


def write_input(path: Path, seq: str, smiles: str, msa: Path | None) -> None:
    lines = ["version: 1", "sequences:", "  - protein:", "      id: A",
             f"      sequence: {seq}"]
    if msa is not None:
        lines.append(f"      msa: {msa}")
    lines += ["  - ligand:", "      id: B", f"      smiles: '{smiles}'",
              "properties:", "  - affinity:", "      binder: B"]
    path.write_text("\n".join(lines) + "\n")


def read_outputs(out_dir: Path, stem: str) -> dict:
    """Collect the affinity numbers and every pose's confidence."""
    base = out_dir / "predictions" / stem
    got: dict = {}

    aff_file = base / f"affinity_{stem}.json"
    if aff_file.exists():
        a = json.loads(aff_file.read_text())
        raw = a.get("affinity_pred_value")
        if raw is not None:
            got["raw_affinity_pred_value"] = round(float(raw), 4)
            got["pred_p_affinity"] = round(to_p_affinity(float(raw)), 3)
        if a.get("affinity_probability_binary") is not None:
            got["binder_probability"] = round(float(a["affinity_probability_binary"]), 4)
        # The two ensemble members. Their gap is disagreement we get for free.
        m1, m2 = a.get("affinity_pred_value1"), a.get("affinity_pred_value2")
        if m1 is not None and m2 is not None:
            got["ensemble_gap"] = round(abs(float(m1) - float(m2)), 4)

    scores = []
    for conf_file in sorted(base.glob(f"confidence_{stem}_model_*.json")):
        c = json.loads(conf_file.read_text())
        if not scores:  # the first file is the best-ranked pose
            for k in ("confidence_score", "complex_plddt", "iptm",
                      "ligand_iptm", "ptm"):
                if c.get(k) is not None:
                    got[k] = round(float(c[k]), 4)
        if c.get("confidence_score") is not None:
            scores.append(float(c["confidence_score"]))
    got["n_poses"] = len(scores)
    got["pose_spread"] = round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0
    return got


def load_done() -> set[tuple[str, str]]:
    if not RESULTS.exists():
        return set()
    with RESULTS.open() as f:
        return {(r["target"], r["ligand"]) for r in csv.DictReader(f)
                if r.get("status") == "ok"}


def append(row: dict) -> None:
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="just this one target")
    ap.add_argument("--limit", type=int, default=0, help="stop after N molecules")
    ap.add_argument("--poses", type=int, default=3,
                    help="structures per molecule; more gives a better spread signal")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per molecule")
    ap.add_argument("--boltz", default="boltz", help="path to the boltz command")
    args = ap.parse_args()

    if shutil.which(args.boltz) is None:
        sys.exit(f"'{args.boltz}' not found. Install it with:\n"
                 "  cd ~/scienceclaw-bench && uv pip install boltz\n"
                 "or point --boltz at the executable.")
    if not (DATA / "benchmark.csv").exists():
        sys.exit("No data/benchmark.csv. Run: python3 fetch_benchmark.py --all")

    with (DATA / "benchmark.csv").open() as f:
        rows = list(csv.DictReader(f))
    if args.target:
        rows = [r for r in rows if r["target"] == args.target]

    done = load_done()
    todo = [r for r in rows if (r["target"], r["ligand"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(rows)} molecules, {len(done)} already done, {len(todo)} to run\n")

    sequences: dict[str, str] = {}
    WORK.mkdir(exist_ok=True)
    started = time.time()

    for n, r in enumerate(todo, 1):
        target, ligand = r["target"], r["ligand"]
        row = {"target": target, "ligand": ligand,
               "measured_p_affinity": r["p_affinity"], "status": "error", "note": ""}

        if target not in sequences:
            pdb = DATA / f"{target}_protein.pdb"
            sequences[target] = sequence_from_pdb(pdb) if pdb.exists() else ""
        seq = sequences[target]
        if not seq:
            row["note"] = "no protein sequence"
            append(row)
            print(f"[{n}/{len(todo)}] {target}/{ligand}: skipped, no sequence")
            continue

        atoms = rough_atom_count(r["smiles"])
        if atoms > MAX_HEAVY_ATOMS:
            row["status"], row["note"] = "skipped", f"ligand too big ({atoms} atoms)"
            append(row)
            print(f"[{n}/{len(todo)}] {target}/{ligand}: skipped, {atoms} atoms")
            continue

        stem = f"{target}__{ligand}".replace("/", "_")
        work = WORK / stem
        work.mkdir(exist_ok=True)
        yaml_path = work / f"{stem}.yaml"

        # One MSA per target, reused by every molecule for that target.
        msa = WORK / f"{target}.a3m"
        write_input(yaml_path, seq, r["smiles"], msa if msa.exists() else None)

        cmd = [args.boltz, "predict", str(yaml_path), "--out_dir", str(work),
               "--diffusion_samples", str(args.poses)]
        if not msa.exists():
            cmd.append("--use_msa_server")

        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=args.timeout)
            row["seconds"] = round(time.time() - t0, 1)
            if p.returncode != 0:
                row["note"] = (p.stderr or p.stdout or "")[-300:].replace("\n", " ")
            else:
                row.update(read_outputs(work, stem))
                row["status"] = "ok" if "pred_p_affinity" in row else "error"
                if row["status"] == "error":
                    row["note"] = "ran but produced no affinity value"
                # Keep this target's MSA for the next molecule.
                if not msa.exists():
                    for found in work.rglob("*.a3m"):
                        shutil.copy(found, msa)
                        break
        except subprocess.TimeoutExpired:
            row["seconds"] = args.timeout
            row["note"] = f"timed out after {args.timeout}s"

        append(row)
        mark = "ok " if row["status"] == "ok" else "FAIL"
        extra = (f"pred={row.get('pred_p_affinity')} measured={r['p_affinity']} "
                 f"gap={row.get('ensemble_gap')}" if row["status"] == "ok"
                 else row["note"][:60])
        rate = (time.time() - started) / n
        left = rate * (len(todo) - n) / 60
        print(f"[{n}/{len(todo)}] {mark} {target}/{ligand}  {extra}  "
              f"({row.get('seconds', 0)}s, ~{left:.0f} min left)")

    print(f"\nWrote {RESULTS}")
    print("Next: .venv/bin/python baseline_fingerprint.py, then compare.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
