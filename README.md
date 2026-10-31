# Trust router for AI scientists

AI agents doing science are right somewhere between a third and two thirds of
the time, and they sound equally sure either way. A scientist handed 300
answers with no way to sort them cannot use any of them, because checking one
costs about as much as doing the work.

This does not try to make the agents smarter. It sorts their answers into
three tiers, using signals about *how* each answer was produced and never the
answer itself:

- **Act on it** — the signals agree; worth the bench time
- **Check first** — something is off; look before committing
- **Set aside** — the model is out of its depth here

## The measured result

On [BixBench](https://arxiv.org/abs/2503.00096) (FutureHouse), 296
bioinformatics questions:

| Tier | Coverage | Accuracy |
| --- | --- | --- |
| Act on it | 8% | **69.6%** |
| Check first | 30% | 47.7% |
| Set aside | 62% | 20.0% |
| **Agent alone** | 100% | **32.1%** |

2.2x lift on the top tier. Random sorting never matched it in 2,000 shuffles.
The most-trusted 5% runs at 93%.

## The finding that cost us a result

A second run used four agents that were all Claude Sonnet. The router scored
1.1x, and "check first" came out *more* accurate than "act on it" — no signal
at all. Random sorting matched it 58% of the time.

The working version's agreement signal came from GPT-4o versus Claude, two
different model families. Same model in three roles makes the same mistakes,
so agreement means nothing.

**Independence between agents matters more than the number of agents.** The
live app now runs its three agents on Sonnet, Haiku and Opus for that reason.

## What is here

| File | What it does |
| --- | --- |
| `app.py`, `app.html` | Ask a question, watch three agents work, get sorted findings. Reuses Quorum's literature code for real papers and PDFs |
| `serve.py` | Live view of the architecture during a benchmark run |
| `router.py` | The router scored on BixBench's shipped results. Free, no API calls |
| `pipeline.py` | Four agents answering the benchmark live, with a hard dollar cap |
| `score.py` | Opens the golden answers and scores a run. The only place answers are read |
| `golden.py` | Builds the held-out dataset: options re-shuffled, ids hashed, no answer columns in the agents' copy |
| `kosmos_adapter.py` | Routes open-source Kosmos hypotheses into tiers |
| `fetch_benchmark.py` | Pulls the OpenFE protein-ligand benchmark into one clean table |
| `run_boltz.py` | Runs Boltz-2 and records every signal a referee could use |
| `baseline_fingerprint.py` | The cheap laptop baseline a structure model has to beat |
| `BENCHMARK_PROMPT.md` | The per-case scoring task, word for word, for every system tested |
| `OPEN_PROBLEM_PROMPT.md` | The open task, to compare how different systems approach it |

## Running it

```bash
# The live app: ask a question, get sorted findings
ANTHROPIC_API_KEY=sk-ant-... python3 app.py      # http://localhost:8800

# The free result, no API calls
python3 router.py

# A live benchmark run, hard-capped
ANTHROPIC_API_KEY=sk-ant-... python3 pipeline.py --split dev --limit 40 --budget 8
python3 score.py
```

`app.py` imports `literature.py` from [Quorum](https://github.com/SHAFF622/quorum)
for Europe PMC search, PubMed Central full text and PDF reading. Without it the
app still runs; it just has no papers.

## Honest limits

- The top tier on the working run holds 23 answers, so its true accuracy is
  roughly 50–89%.
- Those are zero-shot baselines, not full agentic runs. The agentic benchmark
  needs Docker and hours.
- Applied to Kosmos hypotheses the router sorts but cannot be scored, because
  nobody knows which fresh hypotheses are true. Every accuracy figure here
  comes from BixBench, where the answers exist.
- Nobody on this project is a biologist. We can tell a wrong answer from a
  right one because the benchmark says so; we cannot tell a catastrophically
  wrong answer from a trivially wrong one.

## Credits

Benchmarks: [BixBench](https://arxiv.org/abs/2503.00096),
[OpenFE protein-ligand benchmark](https://github.com/OpenFreeEnergy/protein-ligand-benchmark).
Architecture borrowed from [Kosmos](https://arxiv.org/abs/2511.02824) and its
[open-source implementation](https://github.com/jimmc414/Kosmos).
