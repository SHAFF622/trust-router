"""
A live view of the architecture, at http://localhost:8800

Kosmos ships no interface at all, so this draws the shape of the system and
fills each part with what is actually happening: the shared record in the
middle, the agents that feed it on either side, and underneath it the layer
Kosmos does not have, which decides what to believe.

Everything is read from disk on each request, so refreshing shows the current
state of a run in progress.

Run:  python3 serve.py        then open http://localhost:8800
"""

from __future__ import annotations

import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS, CALLS = HERE / "agent_runs.jsonl", HERE / "llm_calls.jsonl"
GOLDEN = HERE / "bixdata" / "golden.csv"
KOSMOS = Path.home() / "ai-scientist" / "kosmos-community" / "artifacts"
PORT = 8800

LABEL = {"ACT_ON": "Act on it", "CHECK": "Check first", "SET_ASIDE": "Set aside"}
# Which of our agents stands in for each part of the Kosmos architecture.
ANALYSIS_ROLES = ("analyst_quick", "analyst_careful")
LIT_ROLES = ("librarian",)
CHECK_ROLES = ("critic",)


def jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # the last line may be half-written mid-run
    return out


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def kosmos_hyps() -> list[dict]:
    out = []
    for p in sorted(KOSMOS.glob("**/cycle_*/hypotheses.json")) if KOSMOS.exists() else []:
        try:
            for h in json.loads(p.read_text()):
                out.append({"run": p.parent.parent.name, "cycle": p.parent.name,
                            "statement": (h.get("statement") or "").strip()})
        except Exception:  # noqa: BLE001 - a file being written is skipped
            pass
    return out


def agent_stats(calls: list[dict], roles) -> dict:
    sel = [c for c in calls if c.get("role") in roles]
    if not sel:
        return {"n": 0, "cost": 0.0, "secs": 0.0, "model": "—"}
    return {"n": len(sel),
            "cost": sum(c.get("cost_usd", 0) for c in sel),
            "secs": sum(c.get("seconds", 0) for c in sel) / len(sel),
            "model": sel[0].get("model", "—")}


def build() -> str:
    rows, calls = jsonl(RUNS), jsonl(CALLS)
    key = ({r["case_id"]: r["correct_option"] for r in csv.DictReader(GOLDEN.open())}
           if GOLDEN.exists() else {})
    for r in rows:
        r["ok"] = r.get("answer_letter") == key.get(r["case_id"]) if r["case_id"] in key else None

    scored = [r for r in rows if r["ok"] is not None]
    acc = lambda s: sum(x["ok"] for x in s) / len(s) if s else 0.0
    overall = acc(scored)
    tiers = {t: [r for r in scored if r["tier"] == t] for t in LABEL}
    spend = sum(c.get("cost_usd", 0) for c in calls)

    an, lit, cr = (agent_stats(calls, ANALYSIS_ROLES), agent_stats(calls, LIT_ROLES),
                   agent_stats(calls, CHECK_ROLES))
    hyps = kosmos_hyps()

    def box(title, lines, cls=""):
        body = "".join(f"<div class='l'>{l}</div>" for l in lines)
        return f"<div class='box {cls}'><div class='bt'>{title}</div>{body}</div>"

    analysis = box("Data analysis agents", [
        f"<b>{an['n']}</b> agent calls", f"{an['model']}",
        f"{an['secs']:.1f}s average, ${an['cost']:.2f}"])
    literature = box("Literature agents", [
        f"<b>{lit['n']}</b> agent calls", f"{lit['model']}",
        f"{lit['secs']:.1f}s average, ${lit['cost']:.2f}"])
    world = box("Shared record", [
        f"<b>{len(rows)}</b> findings posted",
        f"every one carries the signals behind it",
        f"{len(calls)} calls, ${spend:.2f} spent"], "accent")

    verify = box("Verification panel &mdash; our addition", [
        f"<b>{cr['n']}</b> independent reviews on {cr['model']}",
        "plus two checks in plain code: did it answer the same way twice,",
        "and did the literature view agree"], "add")

    tier_cells = "".join(
        f"<div class='tier {t}'><div class='tn'>{LABEL[t]}</div>"
        f"<div class='tv'>{acc(v)*100:.0f}%</div>"
        f"<div class='tc'>{len(v)} findings</div></div>"
        for t, v in tiers.items())

    verdict = ("<p class='warn'>The tiers are not in order yet, so the sorting is "
               "not working on this run.</p>"
               if scored and acc(tiers["ACT_ON"]) <= acc(tiers["CHECK"]) else "")

    recent = "".join(
        f"<tr><td><span class='pill {r['tier']}'>{LABEL[r['tier']]}</span></td>"
        f"<td class='n'>{r['trust']:.2f}</td><td class='q'>{esc(r['question'][:92])}</td>"
        f"<td class='n'>{'correct' if r['ok'] else 'wrong' if r['ok'] is not None else '—'}</td></tr>"
        for r in rows[::-1][:20]) or "<tr><td colspan=4 class='q'>No findings yet.</td></tr>"

    hyp_rows = "".join(
        f"<tr><td class='q'>{esc(h['statement'][:140])}</td>"
        f"<td class='n'>{esc(h['cycle'])}</td></tr>" for h in hyps[:10]) \
        or "<tr><td colspan=2 class='q'>Kosmos has produced no hypotheses yet.</td></tr>"

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Research run</title><meta http-equiv="refresh" content="5">
<style>
:root{{--bg:#0d1117;--panel:#161b23;--line:#283140;--fg:#e8edf4;--muted:#94a0b0;
--accent:#56b6d8;--add:#9b8cf0;--good:#5cc98c;--warn:#e3a94b;--stop:#e07a7a;
--mono:ui-monospace,SFMono-Regular,Menlo,monospace}}
*{{box-sizing:border-box}}
body{{margin:0;padding:26px 18px 60px;background:var(--bg);color:var(--fg);
font:14px/1.5 system-ui,-apple-system,sans-serif}}
.wrap{{max-width:960px;margin:0 auto;display:flex;flex-direction:column;gap:20px}}
h1{{font-size:1.3rem;margin:0}} .sub{{color:var(--muted);margin:4px 0 0;font-size:.86rem}}
h2{{font-size:.92rem;margin:0 0 10px}}
section{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}}
.arch{{display:flex;flex-direction:column;gap:10px;align-items:center}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:10px;width:100%}}
@media(max-width:620px){{.pair{{grid-template-columns:1fr}}}}
.box{{border:1px solid var(--line);border-radius:9px;padding:12px 14px;width:100%;background:#121821}}
.box.accent{{border-color:var(--accent);background:rgba(86,182,216,.07)}}
.box.add{{border-color:var(--add);background:rgba(155,140,240,.07)}}
.bt{{font-weight:600;font-size:.9rem;margin-bottom:5px}}
.l{{color:var(--muted);font-size:.82rem}} .l b{{color:var(--fg);font-family:var(--mono)}}
.arrow{{color:var(--muted);font-size:1.1rem;line-height:1}}
.tiers{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}
@media(max-width:620px){{.tiers{{grid-template-columns:1fr}}}}
.tier{{border:1px solid var(--line);border-radius:9px;padding:12px 14px}}
.tier.ACT_ON{{border-color:var(--good)}} .tier.CHECK{{border-color:var(--warn)}}
.tier.SET_ASIDE{{border-color:var(--stop)}}
.tn{{font-size:.78rem;color:var(--muted)}}
.tv{{font-family:var(--mono);font-size:1.7rem;font-weight:600;margin-top:2px}}
.tier.ACT_ON .tv{{color:var(--good)}} .tier.CHECK .tv{{color:var(--warn)}}
.tier.SET_ASIDE .tv{{color:var(--stop)}}
.tc{{font-size:.76rem;color:var(--muted)}}
.warn{{color:var(--warn);font-size:.83rem;margin:12px 0 0}}
table{{width:100%;border-collapse:collapse;font-size:.84rem}}
th{{text-align:left;font-family:var(--mono);font-size:.63rem;letter-spacing:.07em;
text-transform:uppercase;color:var(--muted);font-weight:500;padding:0 8px 7px 0;
border-bottom:1px solid var(--line)}}
td{{padding:7px 8px 7px 0;border-bottom:1px solid var(--line);vertical-align:top}}
td.n{{font-family:var(--mono);text-align:right;white-space:nowrap}} td.q{{color:var(--muted)}}
.pill{{font-family:var(--mono);font-size:.62rem;letter-spacing:.05em;text-transform:uppercase;
padding:2px 8px;border-radius:999px;white-space:nowrap}}
.ACT_ON.pill{{color:var(--good);background:#14301f}}
.CHECK.pill{{color:var(--warn);background:#33270f}}
.SET_ASIDE.pill{{color:var(--stop);background:#331818}}
footer{{color:var(--muted);font-size:.77rem}}
</style></head><body><div class="wrap">
<header><h1>Research run</h1>
<p class="sub">The Kosmos architecture with a verification layer added. Reads
what is on disk and refreshes every 5 seconds.</p></header>

<section><h2>The system, as it runs</h2><div class="arch">
  {box("Inputs", [f"A question and a dataset &middot; {len(rows)} findings produced so far"])}
  <div class="arrow">&darr;</div>
  <div class="pair">{analysis}{literature}</div>
  <div class="arrow">&darr;</div>
  {world}
  <div class="arrow">&darr;</div>
  {verify}
</div></section>

<section><h2>What the panel decided</h2>
<div class="tiers">{tier_cells}</div>
<p class="sub" style="margin-top:12px">Agents without the panel: <b>{overall*100:.0f}%</b>
across {len(scored)} scored findings.</p>{verdict}</section>

<section><h2>Latest findings</h2><table>
<thead><tr><th>Tier</th><th class="n">Trust</th><th>Question</th><th class="n">Truth</th></tr></thead>
<tbody>{recent}</tbody></table></section>

<section><h2>Kosmos hypotheses on disk</h2><table>
<thead><tr><th>Statement</th><th class="n">Cycle</th></tr></thead>
<tbody>{hyp_rows}</tbody></table></section>

<footer>Local only. The shareable version is the published artifact.</footer>
</div></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = build().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"Open http://localhost:{PORT}  (ctrl-C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
