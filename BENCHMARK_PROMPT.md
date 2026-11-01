# The benchmark prompt

One task, given word for word to every system we test. Nothing below changes
between runs except the input file. If a system needs its own framing, that
framing goes in a wrapper around this text, never inside it.

## Before you run it: strip the answers out of the input

The source data carries the measured value, its units, and the DOI of the paper
it came from. An agent with web access can look up that paper and read off the
true answer. That would not be a benchmark, it would be a search test.

So every system gets `data/agent_input.csv`, which has only:

    case_id, target_sequence_id, smiles, predicted_affinity,
    ensemble_gap, confidence_score, complex_plddt, iptm, nearest_similarity

Dropped: the measured value, the units, the assay type, the DOI, the original
ligand name, and the target's real name. Case ids are shuffled and renamed.

Afterwards, check for cheating: if a system's accuracy is far above the others,
ask it how it decided, and look for a paper citation in the reasoning.

---

## The prompt

```
You are helping a medicinal chemist decide which molecules are worth making in
the lab. Making one takes weeks and real money, and they can make about ten.

A structure model has predicted how tightly each candidate molecule binds to a
protein target. The model is right roughly 60% of the time. It does not reliably
know when it is wrong. Your job is not to improve its predictions. Your job is
to say which of them are safe to act on.

For each case in the input file, returone of three decisions:

  ACT_ON    You believe this prediction. The chemist should consider making it.
  CHECK     Something is off. Look into it before committing bench time.
  SET_ASIDE The model is out of its depth here. Ignore this prediction.

WHAT EACH INPUT COLUMN MEANS

  smiles              the candidate molecule
  predicted_affinity  the model's prediction. Higher means it expects tighter
                      binding. Each whole number is a tenfold difference.
  ensemble_gap        the model runs two internal copies. This is how far apart
                      they landed. A wide gap means it disagreed with itself.
  confidence_score    the model's own confidence in the predicted shape of the
                      complex. Higher is more confident. Treat this as one weak
                      piece of evidence, not the answer.
  complex_plddt, iptm two more of the model's own confidence measures
  nearest_similarity  how close this molecule is to the most similar molecule
                      the model has seen before, from 0 to 1. Low means the
                      model is working outside familiar chemistry.

RULES

1. You do not have the measured answers and you must not look for them. Do not
   search the web, do not search literature, and do not try to identify the
   source paper, the target, or the compounds. Decide from the input columns
   and your own chemical reasoning alone. If you cannot decide without looking
   something up, return CHECK.
2. Judge each case on its own. Do not tune your decisions to produce a
   particular mix of the three labels.
3. Give every case a decision. No blanks.
4. A confidence number is evidence, not proof. The whole reason this task
   exists is that the model's own confidence does not track whether it is right.

OUTPUT

Return JSON only. No prose before or after it.

{
  "decisions": [
    {
      "case_id": "<exactly as given>",
      "decision": "ACT_ON" | "CHECK" | "SET_ASIDE",
      "trust": <number 0 to 1, how much you believe this prediction>,
      "reason": "<one sentence, under 25 words>",
      "main_signal": "<the one input column that drove your decision>"
    }
  ]
}

The `trust` number matters as much as the label. We rank every case by it, then
measure how accurate the top-scoring slice really was. A flat set of trust
scores scores badly even if the labels look sensible.
```

---

## Keep the comparison fair

Same prompt text, same input file, same cases, for every system. Then log, for
each run: the system, the model behind it, temperature if settable, how long it
took, what it cost, how many cases came back, and how many were malformed.

Two sanity checks before believing any result:

- **Give one system the same cases twice.** If its own two runs disagree a lot,
  the gap between systems may be noise.
- **Shuffle the case order and run again.** A system whose decisions move with
  position is reading order, not evidence.

## Scoring

Scoring code never sees a system's reasoning, only `case_id`, `decision` and
`trust`. The headline number is accuracy on the ACT_ON slice against accuracy
across all cases. Everything else is secondary.
