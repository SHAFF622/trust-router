# The open-problem prompt

Hand the same problem to every system and watch how each one goes about it.
This is step 2 and 3 of our plan: benchmark what exists, log the traces, find
where they go wrong. We fix the deliverable so the outputs are comparable, and
leave the method free, because the method is the thing being tested.

Use this one first. `BENCHMARK_PROMPT.md` is the narrower per-case scoring task,
and it comes later, once we know which approaches are worth measuring.

---

## The prompt

```
THE SITUATION

A medicinal chemist is working on TYK2, a protein target in autoimmune disease.
They have one promising molecule and a thousand possible variations on it.
Making one in the lab takes weeks and real money, so they can make about ten.

Boltz-2, a structure model, ranks all thousand by how tightly it thinks each one
binds. It is right roughly 60% of the time. It attaches a confidence number to
every prediction, and that number does not reliably separate the good
predictions from the bad ones. The model always returns an answer, including
when the molecule is unlike anything it has seen.

So the chemist has two bad options. Trust the whole ranking and risk spending
weeks making a molecule the model was confidently wrong about. Or distrust it,
in which case the model was pointless.

YOUR TASK

Build the missing layer: something that takes each prediction and says which of
three things it is.

  Act on it     The signals agree. Worth the bench time.
  Check first   Something is off. Look before committing.
  Set aside     The model is out of its depth here.

You are not improving the predictor. Do not retrain or replace Boltz-2. The job
is to mark where it works.

THE DATA

A file of candidate molecules, each with Boltz-2's prediction and several
signals that may or may not indicate reliability. Every molecule in it was
really made and measured in a lab, so ground truth exists. You do not have the
measured values. They are held back and used only to score you afterwards.

WHAT TO PRODUCE

1. YOUR APPROACH, before you touch the data. Which signals you expect to
   indicate an unreliable prediction, and why you expect that. Write this first
   and do not revise it later.

2. A WORKING METHOD that assigns every molecule one of the three labels, plus a
   trust score from 0 to 1. The trust score matters as much as the label,
   because we rank by it and then check how accurate the top slice really was.

3. HOW YOU WOULD KNOW IT WORKS. The single number you would report, and what
   value of it would mean you had failed. State this before seeing any results.

4. WHAT YOU ARE UNSURE OF. Where your method would break, and which molecules
   you would not trust yourself about.

RULES

- Do not look up the measured values. No web search for the compounds, the
  target, or the source papers. If a step needs information you do not have,
  say so rather than going to find it.
- Show your reasoning as you go, including the approaches you tried and
  dropped. We read the whole trace, not just the answer.
- If you think the task as described is the wrong thing to build, say that and
  explain what you would build instead. A well-argued disagreement is a valid
  response.
- Do not tune your output to produce a nice-looking mix of the three labels.
```

---

## What we are looking for in the traces

Run it on each system, keep the full transcript, and mark each of these yes or
no. These are the comparison, more than the final answer is.

| Behaviour | Why it matters |
|---|---|
| Stated its approach before seeing the data | Otherwise it is fitting a story to results |
| Noticed the model's own confidence may be useless | This is the whole premise; missing it means it did not read the problem |
| Looked for signals that fail differently from each other | Four signals that fail together are one signal |
| Asked whether the molecules were in the model's training data | The sharpest question available here, and most will miss it |
| Gave trust scores with real spread | Flat scores cannot rank anything |
| Said where it would break | Overconfidence here is the exact failure we are studying |
| Tried to look up the answers | Disqualifying, and worth knowing which systems do it |
| Reported a number without checking it was stable | A correlation on 10 molecules means little |

Run each system twice. If its two runs disagree more than two systems do with
each other, we are comparing noise.
