"""
The gated loop with the world model, offline: a fake model client and a fake paper search.

What must hold:
  - A task searches the papers only when the world model knows nothing about its
    finding; a later cycle on the same subjects searches nothing.
  - --no-world behaves as before: one search per task.
  - The orchestrator sees every kept finding of the run, not just the last 8.
  - With the gate off, set-aside findings are kept and shown too.
  - Each finding in the live app gets the passages that match its own words.

Run:  ~/ai-scientist/quorum/.venv/bin/python -m pytest tests -q
"""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))
pytest.importorskip("tortoise")

import knowledge  # noqa: E402
import recursive  # noqa: E402
from quorum import literature as REAL_LIT  # noqa: E402

PAPERS = {
    "esr1": ("111", "Higher ESR1 expression was associated with better survival in ER-positive breast cancer "
                    "(HR 0.83). ESR1 expression also predicted longer relapse-free survival."),
    "tp53": ("222", "TP53 mutation was associated with worse overall survival in basal-like breast cancer. "
                    "TP53 mutation also predicted shorter relapse-free survival."),
}


class FakeLIT:
    """Paper search without the internet; counts searches."""

    def __init__(self):
        self.searches = 0

    def library(self, text, **options):
        assert options.get("shelf") is False and options.get("general") is True  # Trust Router's search settings
        self.searches += 1
        key = "esr1" if "esr1" in text.lower() else "tp53"
        pmid, body = PAPERS[key]
        return {"papers": [{"pmid": pmid, "title": key.upper()}],
                "passages": [{"pmid": pmid, "title": key.upper(), "section": "Results", "text": body}]}

    rank = staticmethod(REAL_LIT.rank)
    context_for = staticmethod(REAL_LIT.context_for)


def extraction(prompt):
    """The world model asking the model to read a passage: return two findings with exact quotes."""
    passages = prompt.split("Passage 1 [", 1)[-1]  # only the passages: the instructions name both genes as examples
    if "TP53" not in passages.split("Reply with only one JSON object")[0]:
        f = [("ESR1 expression", "survival", "better", "Higher ESR1 expression was associated with better survival"),
             ("ESR1 expression", "relapse-free survival", "better",
              "ESR1 expression also predicted longer relapse-free survival")]
    else:
        f = [("TP53 mutation", "overall survival", "worse", "TP53 mutation was associated with worse overall survival"),
             ("TP53 mutation", "relapse-free survival", "worse", "TP53 mutation also predicted shorter relapse-free survival")]
    return json.dumps({"findings": [
        {"passage": 1, "subject": s, "subject_kind": "gene", "outcome": o, "cohort": "", "direction": d,
         "effect_kind": "none", "effect": -1, "p_value": -1, "adjusted": False, "quote": q} for s, o, d, q in f]})


class FakeClient:
    """Stands in for Claude. Keeps every prompt, so tests can read what each agent was shown."""

    def __init__(self, shaky=False):
        self.prompts, self.calls, self.spent, self.n, self.shaky = [], 0, 0.0, 0, shaky
        import threading
        self.lock = threading.Lock()

    def ask(self, model, prompt, max_tokens=700):
        with self.lock:
            self.prompts.append(prompt)
            self.calls += 1
            self.n += 1
            n = self.n
        if "Reply with only one JSON object" in prompt:
            return extraction(prompt)
        if "Propose" in prompt and "next steps" in prompt:
            return ("Check whether ESR1 expression predicts survival in ER-positive tumours\n"
                    "Check whether TP53 mutation predicts worse survival in basal tumours")
        if "Write ONE specific finding" in prompt:
            gene = "ESR1 expression" if "ESR1" in prompt.split("Your task:")[1] else "TP53 mutation"
            return f"Higher {gene} predicts a 20% change in survival (finding {n})."
        if "Does published work support" in prompt:
            return "UNCLEAR - not addressed" if self.shaky else "SUPPORTED - the passages say so"
        if "Judge a proposed research finding" in prompt:
            return "SHAKY - vague" if self.shaky else "SOUND - specific and backed"
        return "A short answer."


@pytest.fixture
def lit(tmp_path, monkeypatch):
    monkeypatch.setenv("TR_WORLD_DB", str(tmp_path / "world.sqlite"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(recursive, "HERE", tmp_path)  # loop_*.json goes to the temp folder
    fake = FakeLIT()
    monkeypatch.setattr(recursive, "LIT", fake)
    yield fake
    w = knowledge.open_world()
    if w is not None:
        w.close()


def test_a_later_cycle_answers_from_the_world_model_without_searching(lit):
    out = recursive.run("What predicts breast cancer survival?", cycles=3, width=2, cap=9, gate=True,
                        client=FakeClient())
    first, second, third = out["cycles"]
    assert first["searches"] == 2 and first["answered_from_world"] == 0
    assert second["searches"] == 0 and second["answered_from_world"] == 2
    assert third["searches"] == 0
    assert lit.searches == 2
    assert out["world_stats"]["findings"] == 4 and out["world_stats"]["claims"] == 6


def test_no_world_searches_every_task_like_before(lit):
    out = recursive.run("What predicts breast cancer survival?", cycles=3, width=2, cap=9, gate=True,
                        use_world=False, client=FakeClient())
    assert lit.searches == 6 and out["world_stats"] is None


def test_the_orchestrator_sees_every_kept_finding_not_just_the_last_eight(lit):
    c = FakeClient()
    recursive.run("What predicts breast cancer survival?", cycles=6, width=2, cap=9, gate=True, client=c)
    first_finding = next(re.search(r"Higher .*?\(finding \d+\)\.", p).group(0) for p in c.prompts
                         if "Write ONE specific finding" not in p and re.search(r"\(finding \d+\)", p))
    last_plan = [p for p in c.prompts if "next steps" in p][-1]
    assert first_finding in last_plan             # cycle 1's finding still visible in cycle 6
    assert last_plan.count("(finding ") == 10      # all 10 kept findings from cycles 1 to 5
    assert "paper findings" in last_plan           # each with how much the papers back it


def test_without_the_gate_set_aside_findings_are_kept_and_shown(lit):
    c = FakeClient(shaky=True)  # every finding gets two doubts, so the router sets it aside
    out = recursive.run("q", cycles=2, width=2, cap=9, gate=False, client=c)
    assert len(out["record"]) == 4 and not out["dropped"]
    assert "Set aside:" in [p for p in c.prompts if "next steps" in p][-1]
    gated = recursive.run("q", cycles=2, width=2, cap=9, gate=True, client=FakeClient(shaky=True))
    assert not gated["record"] and len(gated["dropped"]) == 4


def test_each_finding_gets_its_own_best_passages():
    lib = {"passages": [{"pmid": "111", "title": "ESR1", "section": "R", "text": PAPERS["esr1"][1]},
                        {"pmid": "222", "title": "TP53", "section": "R", "text": PAPERS["tp53"][1]}]}
    ev = knowledge.evidence_for("TP53 mutation predicts worse survival", [], lib, REAL_LIT)
    assert ev.startswith("[PMID 222]")
    ev = knowledge.evidence_for("ESR1 expression predicts better survival", [], lib, REAL_LIT)
    assert ev.startswith("[PMID 111]")


def test_live_app_checks_findings_side_by_side_with_their_own_evidence(lit, monkeypatch):
    import app
    world = knowledge.open_world()
    monkeypatch.setattr(app, "WORLD", world)
    monkeypatch.setattr(app, "LIT", FakeLIT())
    shown = {}

    class AppClient(FakeClient):
        def __init__(self, cap=2.0):
            super().__init__()

        def ask(self, model, prompt, max_tokens=900):
            if "write 4 to 6" in prompt:
                return ("Higher ESR1 expression predicts 20% better survival in ER-positive tumours.\n"
                        "TP53 mutation predicts 30% worse overall survival in basal tumours.")
            if "Judge whether published work supports" in prompt:
                stmt = prompt.split("Statement: ")[1].split("\n")[0]
                shown[stmt[:4]] = prompt.split("Passages from papers:\n")[1]
                return "SUPPORTED - yes"
            if "reviewing a proposed research finding" in prompt:
                return "SOUND - fine"
            return super().ask(model, prompt, max_tokens)

    monkeypatch.setattr(app, "Client", AppClient)
    app.run("Does ESR1 or TP53 predict survival?", cap=1.0)
    assert app.STATE["status"] == "done", app.STATE["error"]
    assert [c["tier"] for c in app.STATE["claims"]] == ["ACT_ON", "ACT_ON"]
    # The ESR1 finding leads with the world model's ESR1 findings. The TP53 finding has no TP53
    # findings to use (this search only found the ESR1 paper), so it gets passages, never ESR1's findings.
    assert shown["High"].startswith("[PMID 111] higher/present ESR1")
    assert "higher/present ESR1" not in shown["TP53"]
    world_node = next(n for n in app.STATE["trace"]["shared"] if n["key"] == "world")
    assert world_node["state"] == "done" and "new findings" in world_node["detail"]
    assert world.stats().claims == 2
