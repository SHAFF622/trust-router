"""
The world model's tables (Tortoise ORM, stored in SQLite).

Think of the world model as the team's shared lab notebook. The literature agent
writes down what papers report, as structured findings ("higher ESR1 -> better
breast-cancer survival, HR 0.83, adjusted, METABRIC, PMID ..."). The AI
scientist and the referees read from it instead of re-reading whole papers. Every
claim, the evidence about it and its verdict go in too, linked to the findings
they touch. The notebook stays between runs, so nothing learned is thrown away.

How the tables connect:

    Paper --< Passage --< Finding >-- Entity (subject: a gene, marker, subtype...)
                            |    >-- Entity (outcome: OS, DSS, ...)
                            |
    Run --< ClaimRecord --< ClaimFinding >-- Finding
                 |--< EvidenceRecord   (one row per referee)
                 |--< VerdictRecord    (append-only: the last one is current)

("--<" means "one to many".) Reading a whole run back means following these
links. `prefetch_related()` does that in a fixed number of queries, one per
table, however many claims there are (see queries.py).
"""

from __future__ import annotations

from tortoise import fields
from tortoise.models import Model


class Paper(Model):
    pmid = fields.CharField(max_length=16, primary_key=True)
    title = fields.TextField(default="")
    pmcid = fields.CharField(max_length=20, null=True)
    shelf = fields.CharField(max_length=60, default="")  # where it came from, e.g. "Pan-Cancer Atlas"
    created_at = fields.DatetimeField(auto_now_add=True)

    passages: fields.ReverseRelation["Passage"]
    findings: fields.ReverseRelation["Finding"]

    class Meta:
        table = "papers"


class Passage(Model):
    """A short piece of a paper. `read_with` records which extraction prompt has read it, so it is read once."""

    id = fields.IntField(primary_key=True)
    paper = fields.ForeignKeyField("world.Paper", related_name="passages")
    sha = fields.CharField(max_length=64, unique=True)
    section = fields.CharField(max_length=200, default="")
    text = fields.TextField()
    read_with = fields.CharField(max_length=40, null=True)

    class Meta:
        table = "passages"


class Entity(Model):
    """A thing findings are about. `key` is the matching form: "ESR1 expression" and "ESR1" share key "esr1"."""

    id = fields.IntField(primary_key=True)
    kind = fields.CharField(max_length=20)  # gene, marker, subtype, treatment, clinical, other, outcome
    key = fields.CharField(max_length=120)
    name = fields.CharField(max_length=200)

    as_subject: fields.ReverseRelation["Finding"]
    as_outcome: fields.ReverseRelation["Finding"]

    class Meta:
        table = "entities"
        unique_together = (("kind", "key"),)


class Finding(Model):
    """One result a paper reports, with the exact words that say it."""

    id = fields.IntField(primary_key=True)
    sig = fields.CharField(max_length=64, unique=True)  # fingerprint, so the same finding is stored once
    paper = fields.ForeignKeyField("world.Paper", related_name="findings")
    passage = fields.ForeignKeyField("world.Passage", related_name="findings", null=True)
    subject = fields.ForeignKeyField("world.Entity", related_name="as_subject")
    outcome = fields.ForeignKeyField("world.Entity", related_name="as_outcome")
    cohort = fields.CharField(max_length=120, default="")
    direction = fields.CharField(max_length=10)  # worse, better, higher, lower, none, mixed (for higher values / presence)
    effect_kind = fields.CharField(max_length=10, default="none")  # HR, OR, RR, other, none
    effect = fields.FloatField(null=True)
    p_value = fields.FloatField(null=True)
    adjusted = fields.BooleanField(default=False)
    quote = fields.TextField()
    prompt_version = fields.CharField(max_length=40)
    created_at = fields.DatetimeField(auto_now_add=True)

    claim_links: fields.ReverseRelation["ClaimFinding"]

    class Meta:
        table = "findings"


class Run(Model):
    id = fields.CharField(max_length=80, primary_key=True)  # the run folder name, e.g. web-20261024-194038
    question = fields.TextField(default="")
    created_at = fields.DatetimeField(auto_now_add=True)

    claims: fields.ReverseRelation["ClaimRecord"]

    class Meta:
        table = "runs"


class ClaimRecord(Model):
    claim_id = fields.CharField(max_length=40, primary_key=True)
    run = fields.ForeignKeyField("world.Run", related_name="claims")
    type = fields.CharField(max_length=20)
    text = fields.TextField()
    body = fields.JSONField()  # the full Claim, as the pipeline saw it
    created_at = fields.CharField(max_length=40)

    evidence: fields.ReverseRelation["EvidenceRecord"]
    verdicts: fields.ReverseRelation["VerdictRecord"]
    links: fields.ReverseRelation["ClaimFinding"]

    class Meta:
        table = "claims"


class EvidenceRecord(Model):
    id = fields.IntField(primary_key=True)
    claim = fields.ForeignKeyField("world.ClaimRecord", related_name="evidence")
    position = fields.IntField(default=0)
    referee = fields.CharField(max_length=40)
    outcome = fields.CharField(max_length=20)
    detail = fields.TextField(default="")
    data = fields.JSONField(default=dict)

    class Meta:
        table = "evidence"
        ordering = ["position"]


class VerdictRecord(Model):
    id = fields.IntField(primary_key=True)
    claim = fields.ForeignKeyField("world.ClaimRecord", related_name="verdicts")
    label = fields.CharField(max_length=20)
    triage = fields.CharField(max_length=20)
    reasons = fields.JSONField(default=list)
    rule_version = fields.CharField(max_length=40)
    created_at = fields.CharField(max_length=40)

    class Meta:
        table = "verdicts"
        ordering = ["id"]


class ClaimFinding(Model):
    """A claim touches a finding. `relation` compares directions on the same outcome; see queries.relation_of."""

    id = fields.IntField(primary_key=True)
    claim = fields.ForeignKeyField("world.ClaimRecord", related_name="links")
    finding = fields.ForeignKeyField("world.Finding", related_name="claim_links")
    relation = fields.CharField(max_length=12)  # agrees, disagrees, related

    class Meta:
        table = "claim_findings"
        unique_together = (("claim", "finding"),)
