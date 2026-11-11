"""
The world model for ordinary (synchronous) code: the pipeline and the referees.

Tortoise is async, and the pipeline runs referees on several threads at once. So
every database call here is handed to one private event loop on one background
thread, and the caller waits for the answer. One loop means one SQLite writer, so
threads never trip over each other.

    world = WorldModel.open()                 # data_cache/world.sqlite
    ids = world.ingest(lib)                   # papers + best passages from literature.library()
    world.extract(ids, llm)                   # the AI reads new passages into findings (each passage once)
    world.brief(question)                     # findings as plain lines for the AI scientist
    world.findings_for(claim)                 # what papers say about this claim, for the skeptic
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import os
import threading
from pathlib import Path
from typing import Optional

from quorum.world import queries
from quorum.world.schemas import ExtractOut, FindingOut, RunExport

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data_cache" / "world.sqlite"

EXTRACT_VERSION = "world-extract-v1"
EXTRACT_SYSTEM = """You read numbered passages from research papers and list the findings they report about
breast cancer outcomes. A finding links one subject (a gene, marker, subtype, treatment or patient feature)
to one outcome (such as overall survival or breast cancer-specific survival), with a direction for higher
values or presence of the subject: "worse", "better", "none" (no association) or "mixed".

Rules:
- Only list findings the passage states. Do not use outside knowledge.
- "quote" must be copied exactly from the passage, word for word. Findings without an exact quote are discarded.
- Use -1 for an effect size or p-value the passage does not give.
- Background statements with no outcome are not findings. An empty list is a fine answer."""


class WorldModel:
    _lock = threading.Lock()
    _shared: dict[str, "WorldModel"] = {}

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="world-model", daemon=True).start()
        self.ctx = self._start(self._init())

    @classmethod
    def open(cls, path: Optional[str | Path] = None) -> "WorldModel":
        """One world model per file per process (Tortoise's connections are process-wide)."""
        key = str(Path(path or os.environ.get("QUORUM_WORLD_DB") or DEFAULT_PATH).resolve())
        with cls._lock:
            for k, w in list(cls._shared.items()):
                if k != key:  # Tortoise holds one database at a time; switching files closes the old one
                    w.close()
            if key not in cls._shared:
                cls._shared[key] = cls(Path(key))
            return cls._shared[key]

    async def _init(self):
        await queries.init(str(self.path))
        # One call at a time. Tasks on one loop still interleave at every await, and two
        # transactions interleaving on one SQLite connection fail ("cannot start a transaction
        # within a transaction"). Each call takes milliseconds, so queueing them costs nothing.
        self._one_at_a_time = asyncio.Lock()
        return contextvars.copy_context()  # later calls run in the context that holds the connection

    async def _queued(self, coro):
        async with self._one_at_a_time:
            return await coro

    def _start(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(60)

    def run(self, coro, timeout: float = 120):
        """Run one async query on the world model's loop and wait for its result."""
        done: concurrent.futures.Future = concurrent.futures.Future()

        def start():
            task = self.loop.create_task(self._queued(coro), context=self.ctx)
            task.add_done_callback(lambda t: done.cancel() if t.cancelled() else
                                   done.set_exception(t.exception()) if t.exception() else done.set_result(t.result()))
        self.loop.call_soon_threadsafe(start)
        return done.result(timeout)

    def close(self) -> None:
        if self.loop.is_closed():
            return
        try:
            self._start(queries.close())
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            WorldModel._shared.pop(str(self.path.resolve()), None)

    # -- filling it -------------------------------------------------------------

    def ingest(self, lib: dict, top: int = 12) -> list[int]:
        return self.run(queries.ingest_library(lib, top))

    def extract(self, passage_ids: list[int], llm, model: Optional[str] = None, batch_size: int = 6,
                on_batch=None, system: str = EXTRACT_SYSTEM, version: str = EXTRACT_VERSION) -> dict:
        """
        Have the AI read the passages it has not read before, into findings. A passage
        read once is never sent again, so a second run on the same papers costs nothing.
        Returns counts: passages read, AI calls, findings kept, findings dropped (quote not in the passage).
        """
        todo = self.run(queries.unread(passage_ids, version))
        totals = {"passages": len(todo), "calls": 0, "kept": 0, "dropped": 0, "skipped": len(passage_ids) - len(todo)}
        if llm is None or not todo:
            return totals
        try:
            from quorum.llm import SONNET, Refused
        except ImportError:  # a copy of the world model without the rest of Quorum (Trust Router)
            SONNET, Refused = None, _NeverRaised
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            user = "\n\n".join(f"Passage {n} [PMID {b['pmid']}, {b['section']}]:\n{b['text']}"
                               for n, b in enumerate(batch, 1))
            totals["calls"] += 1
            try:
                out = llm.json(ExtractOut, model=model or SONNET, system=system, user=user,
                               prompt_version=version, effort="low")
            except (Refused, ValueError):  # a refusal, or a reply that is not the JSON asked for
                continue
            counts = self.run(queries.store_findings(batch, out, version))
            totals["kept"] += counts["kept"]
            totals["dropped"] += counts["dropped"]
            if on_batch:
                on_batch(counts)
        return totals

    def record_routed(self, run_id: str, question: str, claim_id: str, text: str, tier: str,
                      checks: list[dict], extra: Optional[dict] = None) -> list[FindingOut]:
        """
        Store one finding an agent wrote and how the router sorted it, for apps that do not use
        Quorum's Claim and Verdict (Trust Router). `checks` are the votes: {"n": name, "v": -1/0/1, "d": detail}.
        Links it to the paper findings about the same things. Returns those paper findings.
        """
        linked = self.run(queries.findings_for(text, None, 5))
        body = {"claim_id": claim_id, "text": text, "type": "agent", **(extra or {})}
        evidence = [{"referee": c.get("n", "check"), "data": {"vote": c.get("v")}, "detail": c.get("d", ""),
                     "outcome": {1: "supports", -1: "refutes"}.get(c.get("v"), "no_evidence")} for c in checks]
        self.run(queries.record_claims(run_id, question, [body]))
        self.run(queries.record_judgement(claim_id, {"label": tier, "triage": tier, "reasons": [],
                                                     "rule_version": "router"}, evidence,
                                          [(f.id, "related") for f in linked]))
        return linked

    def record_claims(self, run_id: str, question: str, claims: list) -> None:
        self.run(queries.record_claims(run_id, question, [c.model_dump(mode="json") for c in claims]))

    def record_judgement(self, claim, verdict) -> list[FindingOut]:
        """Store the verdict and its evidence, and link the claim to the findings it touches."""
        spec, hr = _spec_and_hr(claim, verdict)
        linked = self.findings_for(claim, spec=spec)
        links = [(f.id, queries.relation_of(f, spec, hr)) for f in linked]
        self.run(queries.record_judgement(
            claim.claim_id, {**verdict.model_dump(mode="json"), "triage": verdict.triage},
            [e.model_dump(mode="json") for e in verdict.evidence], links))
        return linked

    # -- reading it -------------------------------------------------------------

    def findings_for(self, claim, spec: Optional[dict] = None, limit: int = 5) -> list[FindingOut]:
        if spec is None and getattr(claim, "spec", None) is not None:
            spec = claim.spec.model_dump(mode="json")
        return self.run(queries.findings_for(claim.text, spec, limit))

    def brief(self, question: str, passage_ids: Optional[list[int]] = None, limit: int = 12) -> str:
        """
        What the world model knows that bears on the question, as plain lines for an AI prompt:
        findings from this question's best passages, then findings about anything the question names.
        """
        found = self.run(queries.findings_in(passage_ids or [], limit))
        seen = {f.id for f in found}
        found += [f for f in self.run(queries.findings_for(question, None, limit)) if f.id not in seen]
        return "\n".join(f.line() for f in found[:limit])

    def export_run(self, run_id: str) -> Optional[RunExport]:
        return self.run(queries.export_run(run_id))

    def stats(self):
        return self.run(queries.stats())


class _NeverRaised(Exception):
    pass


class AskJSON:
    """
    Lets an app whose client only has ask(model, prompt, max_tokens) -> text fill the world model.
    It asks for JSON matching the schema and checks the reply with Pydantic.
    """

    def __init__(self, ask, model: str, max_tokens: int = 1500):
        self.ask, self.model, self.max_tokens = ask, model, max_tokens

    def json(self, schema, *, model=None, system: str, user: str, prompt_version: str = "", effort: str = ""):
        import json as _json
        import re as _re
        shape = _json.dumps(schema.model_json_schema())
        reply = self.ask(self.model, f"{system}\n\n{user}\n\nReply with only one JSON object matching this JSON "
                                     f"schema. No prose, no code fences.\nSchema: {shape}", self.max_tokens)
        m = _re.search(r"\{.*\}", reply, _re.S)
        if not m:
            raise ValueError("The reply had no JSON object.")
        return schema.model_validate_json(m.group(0))


def _spec_and_hr(claim, verdict) -> tuple[Optional[dict], Optional[float]]:
    """The analysis the referees actually ran and its hazard ratio, from the re-analysis evidence."""
    for e in verdict.evidence:
        if e.referee == "reanalyst" and e.data.get("planned_spec"):
            return e.data["planned_spec"], e.data.get("hr")
    spec = claim.spec.model_dump(mode="json") if getattr(claim, "spec", None) is not None else None
    return spec, None
