"""
The world model, as Trust Router uses it.

Before this, the agents shared what they knew as a plain list of sentences, and
every task searched the papers again from nothing (8 to 38 seconds each on a cold
cache). Now there is one store, Quorum's world model, that keeps:

  - the papers and their best passages,
  - findings read out of those passages (subject, outcome, direction, exact quote),
  - every finding the agents wrote, with its tier and the checks behind it.

Two things change for the agents:
  1. A task asks the world model first. It searches the papers only when the
     world model knows nothing about what the finding names.
  2. Each finding is judged against its own evidence: the paper findings about
     the same things, then the passages that match its own words best. Before,
     every finding got the same first 3,000 characters of passages.

Everything here is optional. Without Tortoise installed, or with no papers, the
app works as it did before.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent

EXTRACT_VERSION = "tr-extract-v2"
EXTRACT_SYSTEM = """You read numbered passages from research papers and list the findings they report.
A finding links a subject to an outcome.

- subject: what differs between the groups compared, or what goes up or down. Examples: "BRCA1 carrier status",
  "African ancestry", "TP53 mutation", "tamoxifen treatment". In "TP53 mutations were more frequent in BRCA1
  carriers", the subject is "BRCA1 carrier status", not TP53.
- outcome: what was measured. Examples: "TP53 mutation frequency", "overall survival", "risk of a second cancer".
- direction, for higher values or presence of the subject:
    better / worse  only when the outcome is good or bad for patients (survival, risk, response to treatment);
    higher / lower  for plain measurements (a mutation rate, expression, a count, a proportion);
    none            no association or no difference;
    mixed           it differs by group or the passage is unclear.
- cohort: who was studied, and what they were compared with, if the passage says.

Rules:
- Only list findings the passage states. Do not use outside knowledge.
- "quote" must be copied exactly from the passage, word for word. Findings without an exact quote are discarded.
- Use -1 for an effect size or p-value the passage does not give.
- A plain description with no comparison and no outcome (such as "PIK3CA accounts for 10% of records") is not a
  finding. An empty list is a fine answer."""


def db_path() -> Path:
    """On Vercel only /tmp can be written, and it is wiped between cold starts; on a laptop it stays."""
    if os.environ.get("TR_WORLD_DB"):
        return Path(os.environ["TR_WORLD_DB"])
    if os.environ.get("VERCEL"):
        return Path("/tmp/trust-router-world.sqlite")
    return HERE / "data" / "world.sqlite"


def open_world():
    """The world model, or None when its libraries are missing (the app then works as before)."""
    try:
        from quorum.world import WorldModel
    except ImportError:
        return None
    return WorldModel.open(db_path())


def known_about(world, text: str, limit: int = 5) -> list:
    """Paper findings about the things this sentence names."""
    if world is None:
        return []
    from quorum.world import queries
    return world.run(queries.findings_for(text, None, limit))


def learn(world, lib: dict, ask, model: str) -> dict:
    """Store a paper search and have the AI read its new passages into findings (each passage once)."""
    if world is None or not lib:
        return {"passages": 0, "calls": 0, "kept": 0, "dropped": 0, "skipped": 0}
    from quorum.world.store import AskJSON
    ids = world.ingest(lib)
    counts = world.extract(ids, AskJSON(ask, model), system=EXTRACT_SYSTEM, version=EXTRACT_VERSION)
    return {**counts, "ids": ids}


def findings_from(world, passage_ids: list[int], limit: int = 16) -> list[dict]:
    """The findings the world model holds from these passages, as plain dicts for the page."""
    if world is None or not passage_ids:
        return []
    from quorum.world import queries
    return [f.model_dump() for f in world.run(queries.findings_in(passage_ids, limit))]


def evidence_for(text: str, known: list, lib: Optional[dict], LIT, chars: int = 3500) -> str:
    """
    The evidence one finding is judged against: paper findings about the same things
    first (short, with exact quotes), then the passages that best match this finding's
    own words, up to `chars` characters in all.
    """
    parts = [f.line() for f in known]
    if lib and LIT is not None and lib.get("passages"):
        ranked = LIT.rank(text, lib["passages"])
        for p in ranked:
            if p.get("score", 0) <= 0:
                break
            parts.append(f"[PMID {p['pmid']}] {p['title']} ({p['section']}): {p['text'][:700]}")
    out, used = [], 0
    for line in parts:
        if used + len(line) > chars:
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)
