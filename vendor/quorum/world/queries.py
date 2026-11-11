"""
Everything that reads or writes the world model, as async functions (Tortoise is async).

The pipeline is ordinary synchronous code, so it calls these through store.WorldModel,
which runs them on one background event loop. The FastAPI app (api.py) awaits them directly.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Optional

from tortoise import Tortoise
from tortoise.expressions import Q
from tortoise.functions import Count
from tortoise.transactions import in_transaction

from quorum.world.models import (ClaimFinding, ClaimRecord, Entity, EvidenceRecord, Finding, Paper, Passage,
                                 Run, VerdictRecord)
from quorum.world.schemas import (ClaimOut, EntityOut, EvidenceOut, ExtractOut, FindingOut, Graph, GraphEdge,
                                  GraphNode, LinkOut, RunExport, Stats, VerdictOut)

MODULES = {"world": ["quorum.world.models"]}


async def init(path: str) -> None:
    await Tortoise.init(db_url=f"sqlite://{path}", modules=MODULES)
    await Tortoise.generate_schemas(safe=True)


async def close() -> None:
    await Tortoise.close_connections()


# ---------------------------------------------------------------------------
# Names: one entity per real thing
# ---------------------------------------------------------------------------

# Words that describe how a thing was measured, not what it is: "ESR1 expression" is ESR1.
_FILLER = re.compile(r"\b(expression|expressions|levels?|gene|genes|status|mrna|protein|high|low|"
                     r"higher|lower|increased|decreased|elevated|positivity|amplification)\b")
# The same outcome under different names, mapped to the codes the data uses (OS, DSS, ...).
OUTCOMES = {
    "OS": ["overall survival", "all-cause mortality", "all cause mortality", "death from any cause", "os"],
    "DSS": ["breast cancer-specific survival", "breast cancer specific survival", "cancer-specific survival",
            "cancer specific survival", "disease-specific survival", "disease specific survival",
            "breast cancer death", "breast cancer mortality", "dss", "bcss"],
    "RFS": ["relapse-free survival", "recurrence-free survival", "relapse free survival",
            "recurrence free survival", "disease-free survival", "disease free survival", "rfs", "dfs"],
    "DMFS": ["distant metastasis-free survival", "distant metastasis free survival", "dmfs"],
    "PFS": ["progression-free survival", "progression free survival", "pfs"],
}
_OUTCOME_OF = {alias: code for code, aliases in OUTCOMES.items() for alias in aliases}


def subject_key(name: str) -> str:
    k = _FILLER.sub(" ", name.lower())
    k = re.sub(r"[^a-z0-9+\-/ ]", " ", k)
    return re.sub(r"\s+", " ", k).strip() or name.lower().strip()


def outcome_key(name: str) -> str:
    n = re.sub(r"\s+", " ", name.lower()).strip()
    if n in _OUTCOME_OF:
        return _OUTCOME_OF[n]
    for alias, code in sorted(_OUTCOME_OF.items(), key=lambda kv: -len(kv[0])):
        if len(alias) > 4 and alias in n:
            return code
    return n


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(" ", " ").replace(" ", " ")).strip().lower()


def quote_found(quote: str, passage: str) -> bool:
    """The AI must copy its evidence word for word. A quote that is not in the passage is thrown away."""
    q = _norm(quote).strip(" .\"'")
    return len(q) >= 12 and q in _norm(passage)


def _sha(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Papers and passages (from the literature agent's library)
# ---------------------------------------------------------------------------

def _spread(lib: dict, top: int) -> list[dict]:
    """
    The passages to read: the best few overall, then the best passage of each other relevant
    paper. Taking only the top passages overall let one paper fill most of the list, so most
    findings ended up citing that one paper.
    """
    chosen, seen = [], set()

    def take(r):
        key = (str(r["pmid"]), r["text"][:200])
        if key not in seen and len(chosen) < top:
            seen.add(key)
            chosen.append(r)

    for r in lib.get("passages", [])[: max(1, top // 2)]:
        take(r)
    for p in lib.get("papers", []):
        bp = p.get("best_passage")
        if bp and p.get("relevance", 0) > 0:
            take({"pmid": p["pmid"], "title": p.get("title", ""), "section": bp.get("section", ""), "text": bp["text"]})
    for r in lib.get("passages", []):
        take(r)
    return chosen


async def ingest_library(lib: dict, top: int = 12) -> list[int]:
    """Store the papers and the passages to read (spread across papers). Returns the passage ids, best first."""
    ids = []
    async with in_transaction():
        for p in lib.get("papers", []):
            paper, made = await Paper.get_or_create(pmid=str(p["pmid"]),
                                                    defaults={"title": p.get("title", ""), "pmcid": p.get("pmcid"),
                                                              "shelf": p.get("shelf", "")})
            if not made and p.get("title") and not paper.title:
                paper.title = p["title"]
                await paper.save(update_fields=["title"])
        for r in _spread(lib, top):
            pmid = str(r["pmid"])
            await Paper.get_or_create(pmid=pmid, defaults={"title": r.get("title", "")})
            psg, _ = await Passage.get_or_create(sha=_sha(pmid, r.get("section", ""), r["text"]),
                                                 defaults={"paper_id": pmid, "section": r.get("section", "")[:200],
                                                           "text": r["text"]})
            ids.append(psg.id)
    return ids


async def unread(passage_ids: list[int], version: str) -> list[dict]:
    """The passages this extraction prompt has not read yet, in the order given."""
    # "Not read with this version" must include never-read passages: in SQL, NULL != 'v' is not true.
    rows = await Passage.filter(Q(read_with__isnull=True) | ~Q(read_with=version), id__in=passage_ids) \
        .select_related("paper")
    by_id = {r.id: r for r in rows}
    return [{"id": i, "pmid": by_id[i].paper.pmid, "title": by_id[i].paper.title,
             "section": by_id[i].section, "text": by_id[i].text} for i in passage_ids if i in by_id]


async def store_findings(batch: list[dict], out: ExtractOut, version: str) -> dict:
    """Save what the AI found in one batch of passages, after checking every quote. Returns counts."""
    kept = dropped = 0
    async with in_transaction():
        # A passage read again with a newer prompt: its older findings are replaced, not kept beside the new ones.
        await Finding.filter(passage_id__in=[b["id"] for b in batch]).exclude(prompt_version=version).delete()
        for f in out.findings:
            if not 1 <= f.passage <= len(batch) or not quote_found(f.quote, batch[f.passage - 1]["text"]):
                dropped += 1
                continue
            src = batch[f.passage - 1]
            subj, _ = await Entity.get_or_create(kind=f.subject_kind, key=subject_key(f.subject),
                                                 defaults={"name": f.subject.strip()[:200]})
            okey = outcome_key(f.outcome)
            outc, _ = await Entity.get_or_create(kind="outcome", key=okey,
                                                 defaults={"name": okey if okey in OUTCOMES else f.outcome.strip()[:200]})
            sig = _sha(src["pmid"], subj.key, outc.key, f.direction, _norm(f.quote))
            if await Finding.exists(sig=sig):
                kept += 1
                continue
            await Finding.create(
                sig=sig, paper_id=src["pmid"], passage_id=src["id"], subject=subj, outcome=outc,
                cohort=f.cohort.strip()[:120], direction=f.direction, effect_kind=f.effect_kind,
                effect=f.effect if f.effect > 0 else None, p_value=f.p_value if f.p_value >= 0 else None,
                adjusted=f.adjusted, quote=f.quote.strip(), prompt_version=version)
            kept += 1
        await Passage.filter(id__in=[b["id"] for b in batch]).update(read_with=version)
    return {"kept": kept, "dropped": dropped}


# ---------------------------------------------------------------------------
# Reading findings back
# ---------------------------------------------------------------------------

def _finding_out(f: Finding) -> FindingOut:
    return FindingOut(id=f.id, pmid=f.paper.pmid, title=f.paper.title, subject=f.subject.name,
                      outcome=f.outcome.name, cohort=f.cohort, direction=f.direction, effect_kind=f.effect_kind,
                      effect=f.effect, p_value=f.p_value, adjusted=f.adjusted, quote=f.quote)


def _claim_terms(text: str, spec: Optional[dict]) -> tuple[str, set[str]]:
    """The claim's words, and the exact names its analysis uses (variable, compared levels)."""
    names: set[str] = set()
    if spec:
        var = str(spec.get("variable") or "").lower()
        names |= {var, *[p for p in re.split(r"[_\s]+", var) if len(p) > 1]}
        names |= {str(spec.get(k) or "").lower() for k in ("exposed", "reference")}
    names.discard("")
    return " " + _norm(text) + " ", {subject_key(n) for n in names} | names


def _mentions(key: str, text: str, names: set[str]) -> bool:
    return key in names or re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", text) is not None


# Gene-like names: capitals with a digit (TP53, BRCA1, PIK3CA, ESR1, HER2). A claim about "TP53 mutation
# rates" must find a finding about "TP53 mutation frequency", so these are matched on their own.
_GENE = re.compile(r"\b[A-Z][A-Z0-9-]*\d[A-Z0-9-]*\b")
_WORD = re.compile(r"[a-z0-9]{3,}")


def _anchors(name: str) -> set[str]:
    return {g.lower() for g in _GENE.findall(name)}


async def findings_for(text: str, spec: Optional[dict] = None, limit: int = 5) -> list[FindingOut]:
    """
    Findings about the things a claim mentions: by the subject's name, or by a gene name in the
    subject or the outcome. Best match first: most shared words, then the same outcome, then adjusted.
    """
    words, names = _claim_terms(text, spec)
    genes = _anchors(text) | {a for n in names for a in _anchors(n.upper())}
    entities = await Entity.all()
    subjects = [e.id for e in entities if e.kind != "outcome" and len(e.key) >= 2
                and (_mentions(e.key, words, names) or _anchors(e.name) & genes)]
    outcomes = [e.id for e in entities if e.kind == "outcome" and _anchors(e.name) & genes]
    if not subjects and not outcomes:
        return []
    rows = await Finding.filter(Q(subject_id__in=subjects) | Q(outcome_id__in=outcomes)) \
        .select_related("paper", "subject", "outcome")
    endpoint = (spec or {}).get("endpoint")
    claim_words = set(_WORD.findall(words))

    def shared(f):
        return len(claim_words & set(_WORD.findall(f"{f.subject.name} {f.outcome.name} {f.cohort}".lower())))

    rows.sort(key=lambda f: (-shared(f), f.outcome.key != endpoint, not f.adjusted, f.effect is None, f.id))
    return [_finding_out(f) for f in rows[:limit]]


async def findings_in(passage_ids: list[int], limit: int = 12) -> list[FindingOut]:
    """Findings read from these passages (the question's best passages), adjusted results first."""
    rows = await Finding.filter(passage_id__in=passage_ids).select_related("paper", "subject", "outcome")
    rank = {pid: i for i, pid in enumerate(passage_ids)}
    rows.sort(key=lambda f: (rank.get(f.passage_id, 99), not f.adjusted, f.id))
    return [_finding_out(f) for f in rows[:limit]]


def relation_of(finding: FindingOut, spec: Optional[dict], hr: Optional[float]) -> str:
    """
    Does the claim's own re-analysed number point the same way as the paper?
    Only said when both are about the same outcome and the same thing (the variable,
    or the exposed level of a comparison); otherwise the claim is just "related".
    """
    if not spec or hr is None or hr == 1 or finding.direction not in ("worse", "better"):
        return "related"
    if outcome_key(finding.outcome) != spec.get("endpoint"):
        return "related"
    target = subject_key(str(spec.get("exposed") or spec.get("variable") or ""))
    sk = subject_key(finding.subject)
    if not target or not (sk == target or sk in set(re.split(r"[_\s]+", target))):
        return "related"
    claim_dir = "worse" if hr > 1 else "better"
    return "agrees" if claim_dir == finding.direction else "disagrees"


# ---------------------------------------------------------------------------
# Claims, evidence, verdicts
# ---------------------------------------------------------------------------

async def record_claims(run_id: str, question: str, claims: Iterable[dict]) -> None:
    async with in_transaction():
        await Run.get_or_create(id=run_id, defaults={"question": question})
        for c in claims:
            await ClaimRecord.update_or_create(
                claim_id=c["claim_id"],
                defaults={"run_id": run_id, "type": c.get("type", "data"), "text": c["text"], "body": c,
                          "created_at": c.get("created_at", "")})


async def record_judgement(claim_id: str, verdict: dict, evidence: list[dict], links: list[tuple[int, str]]) -> None:
    async with in_transaction():
        await EvidenceRecord.filter(claim_id=claim_id).delete()  # a re-judged claim keeps only its latest evidence
        await EvidenceRecord.bulk_create([
            EvidenceRecord(claim_id=claim_id, position=i, referee=e["referee"], outcome=e["outcome"],
                           detail=e.get("detail", ""), data=e.get("data", {}))
            for i, e in enumerate(evidence)])
        await VerdictRecord.create(claim_id=claim_id, label=verdict["label"], triage=verdict.get("triage", ""),
                                   reasons=verdict.get("reasons", []), rule_version=verdict.get("rule_version", ""),
                                   created_at=verdict.get("created_at", ""))
        for finding_id, relation in links:
            await ClaimFinding.update_or_create(claim_id=claim_id, finding_id=finding_id,
                                                defaults={"relation": relation})


def _claim_out(c: ClaimRecord, run_id: str) -> ClaimOut:
    verdicts = list(c.verdicts)
    v = verdicts[-1] if verdicts else None
    return ClaimOut(
        claim_id=c.claim_id, run_id=run_id, type=c.type, text=c.text,
        evidence=[EvidenceOut(referee=e.referee, outcome=e.outcome, detail=e.detail, data=e.data or {})
                  for e in c.evidence],
        verdict=VerdictOut(label=v.label, triage=v.triage, reasons=v.reasons or [], rule_version=v.rule_version)
        if v else None,
        findings=[LinkOut(relation=lk.relation, finding=_finding_out(lk.finding)) for lk in c.links],
    )


# One query per table, however many claims: claims, evidence, verdicts, links, findings, papers, subjects, outcomes.
PREFETCH = ("evidence", "verdicts", "links__finding__paper", "links__finding__subject", "links__finding__outcome")


async def export_run(run_id: str) -> Optional[RunExport]:
    """A whole run, with every claim's evidence, verdict and linked findings, in a fixed number of queries."""
    run = await Run.get_or_none(id=run_id)
    if run is None:
        return None
    claims = await ClaimRecord.filter(run_id=run_id).order_by("created_at", "claim_id").prefetch_related(*PREFETCH)
    return RunExport(run_id=run_id, question=run.question, claims=[_claim_out(c, run_id) for c in claims])


async def claim(claim_id: str) -> Optional[ClaimOut]:
    c = await ClaimRecord.get_or_none(claim_id=claim_id).prefetch_related(*PREFETCH)
    return _claim_out(c, c.run_id) if c else None


# ---------------------------------------------------------------------------
# Browsing
# ---------------------------------------------------------------------------

async def stats() -> Stats:
    return Stats(papers=await Paper.all().count(), passages=await Passage.all().count(),
                 passages_read=await Passage.filter(read_with__isnull=False).count(), entities=await Entity.all().count(),
                 findings=await Finding.all().count(), runs=await Run.all().count(),
                 claims=await ClaimRecord.all().count(), links=await ClaimFinding.all().count())


async def entities(kind: Optional[str] = None) -> list[EntityOut]:
    q = Entity.all() if kind is None else Entity.filter(kind=kind)
    rows = await q.annotate(n=Count("as_subject")).order_by("-n", "name")
    return [EntityOut(id=e.id, kind=e.kind, name=e.name, findings=e.n) for e in rows]


async def findings(subject: Optional[str] = None, outcome: Optional[str] = None, limit: int = 100) -> list[FindingOut]:
    q = Finding.all()
    if subject:
        q = q.filter(subject__key=subject_key(subject))
    if outcome:
        q = q.filter(outcome__key=outcome_key(outcome))
    rows = await q.order_by("-id").limit(limit).select_related("paper", "subject", "outcome")
    return [_finding_out(f) for f in rows]


async def graph(subject: Optional[str] = None, limit: int = 300) -> Graph:
    """Entities, papers and claims as nodes; findings and claim links as edges. For drawing."""
    q = Finding.all() if not subject else Finding.filter(subject__key=subject_key(subject))
    rows = await q.order_by("-id").limit(limit).select_related("paper", "subject", "outcome") \
        .prefetch_related("claim_links__claim")
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    for f in rows:
        s, o, p = f"e{f.subject.id}", f"e{f.outcome.id}", f"p{f.paper.pmid}"
        nodes.setdefault(s, GraphNode(id=s, kind=f.subject.kind, label=f.subject.name))
        nodes.setdefault(o, GraphNode(id=o, kind="outcome", label=f.outcome.name))
        nodes.setdefault(p, GraphNode(id=p, kind="paper", label=f"PMID {f.paper.pmid}"))
        edges.append(GraphEdge(source=s, target=o, kind=f.direction, finding_id=f.id))
        edges.append(GraphEdge(source=p, target=s, kind="reports", finding_id=f.id))
        for lk in f.claim_links:
            c = f"c{lk.claim.claim_id}"
            nodes.setdefault(c, GraphNode(id=c, kind="claim", label=lk.claim.text[:80]))
            edges.append(GraphEdge(source=c, target=s, kind=lk.relation, finding_id=f.id))
    return Graph(nodes=list(nodes.values()), edges=edges)
