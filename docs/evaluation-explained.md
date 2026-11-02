# Evaluation explained

A plain-language guide to what this project does, how it is measured, and what the results mean. No background in search or machine learning is assumed. Every number comes from the last full run on the current index (`data/eval/20260925-091624_…`: 40 questions, dense retrieval, `gpt-4o-mini` answers, Gemini judge) unless stated otherwise. The complete history is in [results.md](results.md).

## Contents

**Overview**
- [The short version](#the-short-version)
- [The problem](#the-problem)
- [Who cares about what](#who-cares-about-what)
- [Which models were used](#which-models-were-used)
- [How it is tested](#how-it-is-tested)

**Results**
- [Results at a glance](#results-at-a-glance)
- [Why the system failed](#why-the-system-failed): every failure, by exact reason
  - [Why 80% answered but only 47% correct](#why-80-answered-but-only-47-correct)
  - [How a partial answer is classified](#how-a-partial-answer-is-classified)
  - [The 16 failures by reason](#the-16-failures-by-reason)
  - [Why the model leaves things out](#why-the-model-leaves-things-out)
  - [The 10 answered but not correct](#the-10-answered-but-not-correct)
  - [Is "incomplete" the right label?](#is-incomplete-the-right-label)
  - [The judge is inconsistent](#the-judge-is-inconsistent)
  - [The wrong-fault answers](#the-wrong-fault-answers)
  - [What this means](#what-this-means)

**Metrics explained**
- [Answer metrics](#answer-metrics): TP, FP, FN, TN, accuracy, precision and recall
  - [Two levels, two different TPs](#two-levels-two-different-tps)
  - [Faithful and the other checks](#faithful-and-the-other-checks)
- [Retrieval metrics](#retrieval-metrics)
  - [How the 4 chunks are chosen](#how-the-4-chunks-are-chosen)
  - [What @k means](#what-k-means)
  - [MRR step by step](#mrr-step-by-step)
  - [Which retrieval metrics suit this project](#which-retrieval-metrics-suit-this-project)
  - [Comparing search methods](#comparing-search-methods)
- [Refusing: the central tradeoff](#refusing-the-central-tradeoff)
- [Cost and speed](#cost-and-speed)

**Reliability and next steps**
- [How far to trust these numbers](#how-far-to-trust-these-numbers)
- [Common questions](#common-questions)
- [Glossary](#glossary)
- [What could be done differently](#what-could-be-done-differently)

## The short version

**What it is.** An assistant that answers questions about equipment manuals, demonstrated on three public sensor manuals. It finds the relevant passages, writes an answer from them, attaches a checkable source to every statement, and says so when the manuals do not contain the answer.

**Who it is for.** Engineers, maintenance and service staff who need an answer from a long manual, often while something is broken.

**Why it was built this way.** A wrong instruction for industrial equipment is worse than no instruction. So the design favours refusing over guessing, and everything is measured rather than assumed.

**The result in one paragraph.** Of 30 questions the manuals can answer, the right page was among the 4 retrieved chunks for 28 (93%), an answer was given for 24 (80%), and 14 were judged correct (47%). None of the 10 questions the manuals cannot answer got an invented answer. Search is not the main problem: only 2 of the 16 failures came from search. Most came from the answer step, where the model left out details it had been given, or refused vague questions.

## The problem

Technical knowledge sits in long PDF manuals. Keyword search fails when the user's words differ from the manual's, and a general chatbot may invent a plausible but wrong procedure, register number or voltage.

The approach is **RAG** (retrieval-augmented generation): search the manuals first, then let a language model (LLM) answer using only what was found.

```
question + device  ->  find 4 passages in that manual  ->  LLM writes an answer  ->  quotes are checked  ->  answer or refusal
```

A **chunk** is one passage of up to 800 characters. The manuals are split into 562 chunks: 264 for HPP271, 254 for HMW90 and 44 for WAA151. An **embedding** turns text into a list of numbers so that similar meanings land close together, which lets a search match meaning and not only exact words.

## Who cares about what

| Person | What they care about | Measured by |
|---|---|---|
| Technician using it | Is the answer right and complete? Can I check it? | answer precision and recall, citation on a relevant page |
| Safety or quality owner | Does it ever make things up, or mislead? | unanswerable refused, faithful, wrong answers |
| Product owner | How often does it actually help, or does it refuse too much? | answered, false refusals |
| Engineer improving it | Which stage is failing: search or answer writing? | failures by reason, retrieval metrics |
| Whoever pays for it | What does it cost, and how long does a user wait? | cost, p50 and p95 latency |

## Which models were used

| Job | Model | Notes |
|---|---|---|
| Embeddings (search) | `text-embedding-3-small` | OpenAI. Turns chunks and questions into numbers |
| Writing answers | `gpt-4o-mini`, temperature 0 | OpenAI's low-cost "mini" tier. A normal production model, but not its most capable one. Used for all 40 questions |
| Judging answers | `gemini-3.1-flash-lite` (23 answers), `gemini-3.5-flash-lite` (1) | Google's light tier, on the free tier. A different family from the answer model |
| Reranking (optional) | `gpt-4o-mini` | Not used in the main run |
| Local-class stand-in | `gemma-4-26b-a4b-it` | Tried on 3 questions only, through an API |
| Writing the test questions | Claude Sonnet 5 | Through Claude Code, the same assistant that helped build the system. |

**Are these models strong enough?** This is not known, because no stronger answer model was ever run. The evidence points to the answer step, not the search step, but that does not separate the model from the prompt. The prompt says "Answer only what was asked" and "Do not add background ... or facts from neighbouring topics", which pushes any model towards short answers. A fair test changes one thing at a time with one pinned judge: same model with a new prompt, then a stronger model with the same prompt.

## How it is tested

**The golden set** (`eval/golden.jsonl`) is the answer key: 40 questions written in advance, mostly by Claude, 30 the manuals can answer and 10 they cannot. The model was Claude Sonnet 5, through Claude Code. Each answerable question has the pages holding the answer and a reference answer. Without an answer key, "it looks good" is the only measure, and that does not survive changing anything.

The questions are deliberately mixed: clean ones ("How do I reset the HMW90 to factory settings?"), vague ones ("sensor info is wrong"), and bare codes ("Internal 1"). The 10 unanswerable ones sound reasonable but are not in the manuals ("Can the HMW90 be connected to a KNX building automation bus?").

**Why measure in layers.** One overall score cannot tell you what to fix. If an answer is wrong, either search did not find the page, or it did and the model mishandled it. The fixes are completely different, so the two are measured separately.

## Results at a glance

Every one of the 40 questions lands in exactly one of these outcomes:

| Outcome | Count | Share of 40 |
|---|---|---|
| Answerable, answered, judged correct | 14 | 35% |
| Answerable, answered, judged not correct (partial or wrong) | 10 | 25% |
| Answerable, refused | 6 | 15% |
| Unanswerable, correctly refused | 10 | 25% |
| Unanswerable, answered (invented) | 0 | 0% |

The metrics below are all calculated from these counts. See [two levels, two different TPs](#two-levels-two-different-tps) for how.

| Level | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Decision:** should it answer or refuse? | 85% (34 of 40) | 100% (24 of 24) | 80% (24 of 30) | 89% |
| **Answer:** was the answer right? | 60% (24 of 40) | 58% (14 of 24) | 47% (14 of 30) | 52% |

| Retrieval (dense search) | Result |
|---|---|
| hit@4: a right page was among the 4 chunks | 93% (28 of 30) |
| page recall@4: share of the right pages found | 89% |
| MRR: how high the first right chunk was | 0.81 |

## Why the system failed

This is the most important section. 16 of the 30 answerable questions did not end up with a correct answer, according to the judge. Every explanation of why an answer failed or was judged not correct is gathered here. Some points repeat on purpose; duplicates will be trimmed later.

### Why 80% answered but only 47% correct

**Answered and correct measure different things.**

- **Answered** only means the system did not refuse. That takes two things: the model said the text was sufficient, and at least one of its quotes passed the check. Partial and wrong answers are included.
- **Correct** means the judge found all the essential facts in the answer, with no contradiction. Partial answers are **not** correct.

**While answering, nothing checks whether an answer is complete.** The system's own checks only decide answered or refused. One true sentence with a real quote is enough for the system to show the answer, so a partial answer counts as **answered**. If the manual lists three causes and the model gives one, every check passes and the user sees a confident answer. Only the judge notices that it is partial, and then it marks it **not correct**. The judge needs a reference answer, so it runs only during evaluation, never for a real user. That is the gap between 24 answered and 14 correct: 10 questions.

### How a partial answer is classified

Take `hmw90-fail1`, which gave 1 of the 3 causes in the manual:

| Where | Classified as |
|---|---|
| The system, at answer time | **Answered**: shown to the user, not refused |
| The judge, during evaluation | **Not correct**: a key fact is missing |
| Level 1 metrics (answer or refuse?) | **TP**: answering was the right choice |
| Level 2 metrics (was the answer right?) | **FP and FN**: not right, and the right answer was not given |
| Hand review | **Partial, usable** |

So a partial answer counts in "answered" (80%) but not in "correct" (47%). That is true everywhere in this page.

```
30  answerable questions
28  right page in the 4 chunks       search lost 2
24  answered                         23 of these had the right page, 1 did not
14  judged correct                   10 answered but not correct
```

### The 16 failures by reason

Each failure is given its one main reason. The grouping comes from reading the saved answers, reference answers and refusal reasons.

| Reason | Stage | Model or code responsible | Count | Questions |
|---|---|---|---|---|
| Left out details that were in the retrieved chunks | Answer writing | `gpt-4o-mini` | 5 | `hmw90-fail1`, `hmw90-relay`, `hmw90-factory-reset`, `hpp271-dont-work`, `waa151-heater-ice` |
| Refused a vague question although the right page was there | Answer writing | `gpt-4o-mini` | 3 | `hpp271-reading-incorrect`, `hpp271-wont-power-on`, `hpp271-error-persists` |
| Answered about the wrong fault, right page but wrong table row | Answer writing | `gpt-4o-mini`, reading tables flattened by `pypdf` | 2 | `hmw90-internal-3-sentence`, `waa151-bent-connector` |
| **Correct answer thrown away** by the quote check | Quote check | Code, no model (`docrag/rag/verify.py`) | 2 | `hpp271-filtering-factor`, `hpp271-insight-reset-loop` |
| Search did not find the right page | Search | `text-embedding-3-small` | 2 | `hmw90-internal-1`, `hpp271-chemicals-slow` |
| Answer was right for the narrow question asked, but the reference answer also listed steps the user did not ask about, so the judge marked it not correct | Evaluation | Judge `gemini-3.1-flash-lite`, comparing against reference answers written by Claude Sonnet 5 | 2 | `hmw90-htm10-05`, `hpp271-no-change-5min` |
| **Total** | | | **16** | |

By stage and model:

| Stage | Model or code | Failures |
|---|---|---|
| Answer writing | `gpt-4o-mini` | **10** |
| Quote check | Code, no model | **2** |
| Search | `text-embedding-3-small` | **2** |
| Evaluation | Judge `gemini-3.1-flash-lite`, reference answers by Claude Sonnet 5 | **2** |

What each reason means:

- **Left out details.** The model stopped early. The likely causes are in [why the model leaves things out](#why-the-model-leaves-things-out).
- **Refused a vague question.** "error does not disappear" does not clearly match any passage, so the model said the text was not enough. This is the price of refusing over guessing.
- **Wrong fault.** PDF tables become running text, so it is hard to tell which remedy belongs to which fault. This likely explains these two, but it was not tested.
- **Correct answer thrown away.** The model wrote the right answer, for example register `030Ahex`, value 1. The quote did not match any single chunk word for word, so the statement was dropped and the user saw a refusal. For `hpp271-insight-reset-loop` the likely cause is a quote that straddles two chunks. These two are the cheapest to fix.
- **Search failure.** "Internal 1" is a bare code, and "acetone" never appears in the manual.
- **Evaluation.** The system did nothing clearly wrong here. Example: the user asked "Reading is frozen for a few minutes, is that normal?" and got "Purge is being performed. No actions required", which answers the question. The reference answer, written in advance, also said what to do if it stays frozen over 5 minutes. The user did not ask that, but the judge compares with the reference, so it marked the answer not correct. The fault is in the answer key and the grading, not the system. See [the judge is inconsistent](#the-judge-is-inconsistent).

### Why the model leaves things out

These are the likely reasons. None has been tested separately.

- **The prompt asks for it.** The rules say "Answer only what was asked" and not to add "facts from neighbouring topics". That removed padding, but it can also remove the second and third causes. An earlier prompt without this rule listed all three causes for `hmw90-fail1`.
- **Vague questions.** "my relay isn't switching like I set it up" gives no hint of how much to list.
- **Flattened tables.** It is hard to tell which rows belong together.
- **The model itself.** A stronger model may do better. This was never tried.

### The 10 answered but not correct

"In the chunks?" is a rough keyword search of the 4 retrieved chunks for the missing detail. It shows where to look, but it is not proof. "Review grade" is explained in the next section.

| Question | What it answered | What was missing | In the chunks? | Review grade |
|---|---|---|---|---|
| `hpp271-no-change-5min` | Normal during a purge, no action | What to do after 5 minutes | Yes | Acceptable |
| `hmw90-htm10-05` | A temperature measurement error | Restart, replace the module. Only the meaning was asked | Yes | Acceptable |
| `hmw90-fail1` | 1 of 3 causes | Echo collision, intermittent connection | Yes | Partial, usable |
| `hpp271-dont-work` | 3 causes | Operating voltage, communication settings, pin 5 and more | Yes, except pin 5 | Partial, usable |
| `waa151-heater-ice` | Send for repair | First check the heating element connection | Yes | Partial, usable |
| `hpp271-chemicals-slow` | Sensor purge | Change the filter | No: search miss | Partial, usable |
| `hmw90-relay` | Repeats the symptom, "check DIP switch settings" | Rotary switch, hysteresis, STATUS, RTEST | Yes, except RTEST | Not usable |
| `hmw90-factory-reset` | FRESTORE, then RESET | `PASS 9000` first, and the DIP switch method | Yes | Not usable |
| `hmw90-internal-3-sentence` | Restart, replace the HTM10 module | The Internal 3 remedy: factory settings | Code yes, remedy not found | Wrong |
| `waa151-bent-connector` | Heating element connection | Connector damage from the hexagon nut | Yes | Wrong |

### Is "incomplete" the right label?

**No.** An earlier version of this page said "8 incomplete, 2 wrong". That was too coarse, and it made the 8 sound almost fine. They are not all alike. The judge gives only yes or no, so the saved answers were reviewed by hand using four grades:

| Grade | Rule | Count |
|---|---|---|
| **Acceptable** | Answers what was asked, and acting on it would not mislead | 2 |
| **Partial, usable** | True and helpful, but the manual lists more causes or fixes that the answer does not mention | 4 |
| **Not usable** | Too vague to act on, or missing a required step | 2 |
| **Wrong** | Addresses the wrong fault | 2 |

- **`hpp271-no-change-5min` is acceptable.** The user asked "is that normal?" and the answer is yes, a purge is running. The reference added what to do after 5 minutes, which was not asked.
- **`hmw90-factory-reset` is not almost OK.** The manual's example shows `pass 9000` before the command, and this answer omits it, so following it may produce the very error `FAIL 1`. The judge only noticed that the DIP switch method was missing.
- **`hmw90-relay` is not almost OK.** Its "cause" just repeats the user's complaint.

**Incomplete answers are not counted as correct.** All 10 are "not correct" in the official numbers. The review does not change those. It shows what happens to the metrics under fairer grading:

| What counts as a good answer | Good answers | Precision (of 24) | Recall (of 30) | F1 | Accuracy (of 40) |
|---|---|---|---|---|---|
| **Judge, official** | 14 | 58% | 47% | 52% | 60% |
| Judge + acceptable | 16 | 67% | 53% | 59% | 65% |
| Judge + acceptable + partial usable | 20 | 83% | 67% | 74% | 75% |

So the honest summary is a range: **between 14 and 20 of 30 answered usefully, and 4 of 24 answers should not be followed.** This review was done by Claude. Several of the 14 correct answers also leave out smaller details, and none was downgraded.

### The judge is inconsistent

It treated the same behaviour differently, which is further evidence that a yes-or-no verdict is too blunt:

| Passed as correct | Failed as not correct | Same behaviour |
|---|---|---|
| `hpp271-zero-ma-startup`: wait 5 minutes, no mention of what to do next | `hpp271-no-change-5min`: normal, no mention of what to do after 5 minutes | Both skip the follow-up step |
| `waa151-dont-work`: "shaft is covered with ice and snow ... send to Vaisala for repair" | `waa151-heater-ice`: the same statement | Both skip the connection check |

### The wrong-fault answers

Both answers passed every check and quoted real text, but for a different fault. The first case is also described under [faithful and the other checks](#faithful-and-the-other-checks).

- **`hmw90-internal-3-sentence`.** The user asked what to do about error Internal 3. The system answered "Restart the transmitter. Replace the module if unable to remove the problem". The quote is real, but it is the remedy for a problem with the HTM10 module, not Internal 3. It passed the quote check, it cited page 74 (a right page), and the judge marked it not correct, yet still faithful.
- **`waa151-bent-connector`.** The user removed the connector's hexagon nut. The system answered about the heating element connection, a neighbouring row in the same troubleshooting table, instead of saying the nut can bend the connector.
- **Why both pass the system's checks.** The quote check only proves the words exist on the page, and the right page holds the whole table, so a wrong row on the right page looks fine. Only the judge, comparing with the reference answer, saw it.

### What this means

Only 2 of 16 failures were search failures, so better search will not move 47% much. The effort belongs in the answer step and the quote check. See [what could be done differently](#what-could-be-done-differently).

## Answer metrics

Here is what each number in the run means:

| Metric | Plain meaning | Result |
|---|---|---|
| **answered** | Of answerable questions, how many got an answer, right or not? Includes partial and wrong answers. Same as decision recall | 80% (24 of 30) |
| **false refusal** | Answerable, but the system said "insufficient" | 20% (6 of 30) |
| **judged correct** | Contains the essential facts of the reference, with no contradiction. Partial answers do not count. Same as answer recall | 47% (14 of 30) |
| **faithful** | The judge found every claim supported by the retrieved text. See the warning below | 100% (24 of 24) |
| **citation on a relevant page** | Does an answer cite a page listed in the golden set? | 88% (21 of 24) |
| **unanswerable refused** | The system correctly refused | 100% (10 of 10) |
| **hallucination rate** | Answered when it should have refused | 0% (0 of 10) |

### Two levels, two different TPs

**There are two separate questions, so there are two different TPs.** Mixing them is what makes it confusing. An incomplete answer is a TP at level 1, because answering was the right choice, but not a TP at level 2, because the answer was not right.

**Level 1: should the system answer or refuse?** A positive means "the system gave an answer". This level ignores whether the answer was right.

| | System answered | System refused |
|---|---|---|
| **Answerable** (30) | TP = 24 | FN = 6 |
| **Unanswerable** (10) | FP = 0 | TN = 10 |

- **Accuracy** = (TP + TN) / all = (24 + 10) / 40 = **85%**
- **Precision** = TP / (TP + FP) = 24 / 24 = **100%**. Every time it answered, answering was the right call.
- **Recall** = TP / (TP + FN) = 24 / 30 = **80%**. This is the "answered" metric.

**The 10 correct refusals are the TN at this level.** The label depends on what you call positive: if "positive" meant "refused", they would be TP.

**Level 2: was the answer right?** A positive means "the system gave a right answer".

| Label | Meaning | Count |
|---|---|---|
| **TP** | Answered and judged correct | 14 |
| **FP** | Answered but not correct: incomplete and wrong answers | 10 |
| **FN** | Answerable, but no right answer came out: 6 refused + the same 10 not correct | 16 |
| **TN** | Unanswerable and refused | 10 |

- **Precision** = TP / (TP + FP) = 14 / 24 = **58%**. When it answers, how often is the answer right?
- **Recall** = TP / (TP + FN) = 14 / 30 = **47%**. Of all answerable questions, how many got a right answer? **This is the "judged correct" metric.**
- **F1** = 2 x precision x recall / (precision + recall) = **52%**. One number that balances the two.
- **Accuracy** = questions handled correctly / all = (14 + 10) / 40 = **60%**

**Why the counts add up to 50, not 40.** The 10 not-correct answers are both an FP and an FN: the system said something not right, and it also failed to give the right answer. This is the usual way of scoring systems that can decline to answer. It is also why accuracy is calculated from the 40 questions and not by adding up the table.

**Why precision and recall matter here.** Precision is about trust: can I act on what it says? Recall is about usefulness: does it help when help exists? A system can raise one by giving up the other, by refusing more or answering more.

### Faithful and the other checks

| Check | Who does it | Question it answers | What it does not prove |
|---|---|---|---|
| **Quote check** | Code, on every answer | Is this quote word for word in a retrieved chunk of the cited page? | That the quote supports the statement |
| **Citation on a relevant page** | Code, during evaluation | Does the answer cite a page listed in the golden set? | That the right part of the page was used |
| **Faithful** | LLM judge, during evaluation | Is every claim in the answer supported by the retrieved text? | That the answer is complete or correct |
| **Correct** | LLM judge, during evaluation | Does the answer contain the essential facts of the reference? | Nothing about whether it is safe to follow |

**Faithful is not based on pages.** The judge is told: "every factual claim in the system's answer is supported by the excerpts. One unsupported claim means not faithful, even if it happens to be true."

**A correct page is not a good measure of faithfulness.** A page can hold a whole error table, so the right page can still give the wrong row. That is what happened with `hmw90-internal-3-sentence`:

- The user asked what to do about error Internal 3.
- The system answered "Restart the transmitter. Replace the module if unable to remove the problem".
- The quote is real, but it is the remedy for a problem with the HTM10 module, not Internal 3.
- It passed the quote check. It cited page 74, a right page. The judge marked it not correct, yet still faithful.

**The 100% faithful score is too generous.** The text is in the manual, but the claim "do this for Internal 3" is not. On a strict reading that answer is not faithful. Two more are doubtful: `waa151-bent-connector` applied a heating-element remedy to a connector fault, and `hpp271-chemicals-slow` named acetone as the cause when the quote only says "chemicals". So the fair statement is **no answer contained made-up text, and on a strict reading about 21 of 24 were faithful.** This count is a reading of the saved answers, not a rerun.

**Watch the denominators.** Faithful and citation are out of the 24 answered, since a refusal cannot be unfaithful.

**The judge** is a second LLM that compares each answer with the reference answer and the retrieved text. It comes from a different model family (Gemini judging OpenAI) to avoid a model favouring its own style. It is needed because a sentence cannot be compared to a reference by exact match. It is imperfect, which is why the numbers are read as direction only.

## Retrieval metrics

These need no answer model, so they are nearly free to run.

### How the 4 chunks are chosen

Search scores every chunk in the named device's manual by similarity to the question, sorts them best first, and keeps the first 4. That is the only rule.

- **They are not consecutive.** They can come from anywhere in the manual. For `hmw90-fail1` they came from pages 47, 42, 73 and 42. Ranked means ordered by score: first is the best match, fourth the weakest of the four.
- **They are not known to be relevant.** The system does not know which chunks are right. It takes the 4 that look most similar, even if none of them helps. Whether they really were relevant is what these metrics check, using the pages listed in the golden set.
- **The 4 is a fixed setting** (`top_k`). Every question gets exactly 4 chunks, whether it needs 1 or 10. The system does not decide how many to fetch based on the question.

A chunk counts as relevant if it comes from a page listed in the golden set for that question.

### What @k means

The results form a ranked list, and **k is where you stop reading it**. **Top 4** means the first 4 results in that list, so it is the same as k = 4.

- **@1** looks only at the first chunk.
- **@3** looks at the first 3 chunks.
- **@4** looks at all 4 chunks the model receives.

**Why measure @4 if there are always 4 chunks?** Because the question is not how many chunks came back, but which ones. There are hundreds of chunks in a manual and only 4 are chosen. hit@4 asks whether a right page made it into those 4, which is the only way the model can see it. A different search method picks a different 4, so hit@4 compares methods. @1 and @3 add how good the ordering is.

| Metric | Plain meaning | Result (dense) |
|---|---|---|
| **hit@1** | Was the first chunk on a right page? | 70% (21 of 30) |
| **hit@3** | Was at least one of the first 3 chunks on a right page? | 93% (28 of 30) |
| **hit@4** | Was at least one of the 4 chunks on a right page? | 93% (28 of 30) |
| **page recall@4** | Of the right pages for a question, what share appeared in the 4 chunks? Average over the 30 questions | 89% |
| **MRR** | How high up the list was the first right chunk? 1.0 means always first. Looks at all 4 chunks | 0.81 |
| **precision@4** | Of the 4 chunks, what share was on a right page? Average over the 30 questions. Not a good measure here, see below | 42% |
| **capped precision@4** | The same, but divided by the most that were possible | 61% |

hit@3 equals hit@4, so the fourth slot never supplied the first right page in this run.

Real examples from the run:

| Question | Right pages | Pages retrieved | Result |
|---|---|---|---|
| `hmw90-htm10-03` | 74 | 74, 74, 75, 75 | hit@1 yes, recall 1/1, reciprocal rank 1, precision 2/4 |
| `hmw90-fail1` | 73 | 47, 42, **73**, 42 | hit@1 no, hit@3 yes, recall 1/1, reciprocal rank 1/3, precision 1/4 |
| `hmw90-factory-reset` | 77, 78, 79 | 77, 79, 79, 77 | recall 2/3 with one page missing, precision 4/4 |
| `hmw90-internal-1` | 74 | 95, 96, 85, 87 | total miss, reciprocal rank 0 |

### MRR step by step

MRR (mean reciprocal rank) rewards putting a right chunk near the top. hit@k only says yes or no, while MRR gives partial credit by position.

For each question, find the first chunk on a right page and give points by its position. Later right chunks are ignored.

| First right chunk is at | Points |
|---|---|
| position 1 | 1 |
| position 2 | 1/2 = 0.5 |
| position 3 | 1/3 = 0.33 |
| position 4 | 1/4 = 0.25 |
| not in the list | 0 |

Then average the points over all questions. This run:

| Position | Questions | Points |
|---|---|---|
| 1 | 21 | 21 x 1 = 21 |
| 2 | 5 | 5 x 0.5 = 2.5 |
| 3 | 2 | 2 x 0.33 = 0.67 |
| not found | 2 | 0 |
| **Total** | **30** | **24.17** |

MRR = 24.17 / 30 = **0.81**. The first right chunk is usually first or second.

### Which retrieval metrics suit this project

**Retrieval is a ranking task, not a yes-or-no classification,** so accuracy, precision and recall do not fit it directly. Precision and recall can be defined for it:

| Label | Meaning here |
|---|---|
| **TP** | Retrieved chunk on a right page |
| **FP** | Retrieved chunk not on a right page |
| **FN** | Relevant chunk or page not retrieved |
| **TN** | Chunk not retrieved and not relevant: hundreds per question |

- **Accuracy is useless here.** Because TN is huge, it averages about 97% and is at least 91% on every question, even when search misses completely. That is why it is not reported.
- **Precision@4 is a poor measure here.** The formula is right, and the irrelevant chunks do count as FP: with 2 of 4 relevant, precision = 2 / (2 + 2) = 50%. But we always fetch 4, so the score depends on a number we picked. Questions have between 2 and 9 relevant chunks. With only 2, precision cannot exceed 50%, which is the case for 13 of 30 questions. With k = 10, it would be flooded even more. Its average maximum here is 72%, so 42% says little on its own.
- **It is also understated.** `hpp271-condensation-off` was answered correctly from page 31, but only page 73 is listed as right, so the chunk that gave the answer counts as FP.
- **Page recall@4 suits this project better.** Irrelevant chunks are not in its formula, so it is not flooded, and it answers the real question: did the model get what it needed? It has one limit: raising k can only raise it, so it must be read with the fixed k.

| Metric | Verdict for this project |
|---|---|
| **hit@4** | Keep as the main number: could the model see the answer at all? |
| **page recall@4** | Keep as a main number: matters when the answer spans pages |
| **MRR** | Keep as secondary: order matters little, since the model reads all 4 |
| **precision@4** | Drop as a headline. Only useful to compare methods on the same questions |
| **capped precision@4** | Better replacement: TP / min(4, relevant chunks), which removes the ceiling |
| **Context recall** | **The best fit, not measured yet.** What share of the facts in the reference answer appears in the 4 chunks? It measures facts, not pages |
| **nDCG** | A standard ranking metric for graded relevance, such as "contains the answer" against "same page". Not worth it here, since order matters little |

**The deeper limit is page-level labels.** A chunk on the right page counts as TP even if the answer is in a different row. That is how `hmw90-internal-3-sentence` scored a perfect hit and still got the wrong fault.

### Comparing search methods

| Method | How it works | hit@1 | hit@3 | hit@4 | page recall@4 | MRR | precision@4 |
|---|---|---|---|---|---|---|---|
| **dense** (default) | Match by meaning, using embeddings | 70% | 93% | 93% | 89% | 0.81 | 42% |
| **hybrid** | Combine meaning search with keyword search (BM25) | 67% | 83% | 87% | 81% | 0.75 | 38% |
| **hybrid + rerank** | An LLM re-scores hybrid's top 8 and keeps the best 4 | 83% | 93% | 93% | 87% | 0.88 | 42% |

Keyword search was added because embeddings miss exact codes such as `Internal 1`. In practice its noise on chatty questions pushed good results down, so hybrid did worse overall.

**Is reranking really better?** Only at putting a right chunk first. It did not find more, and it has not been shown to give better answers:

- **The gain is ordering.** hit@1 rose from 21 to 25 questions, and MRR from 0.81 to 0.88. The numbers are correct, from saved runs.
- **It found no more overall.** Both find 28 of 30, and page recall is slightly lower. Reranking rescued 2 questions dense missed (`hmw90-internal-1`, `hpp271-chemicals-slow`) and lost 2 dense found (`hmw90-relay`, `waa151-dont-work`).
- **Order matters little here.** The model reads all 4 chunks, so moving a right chunk from third to first does not give the model anything new.
- **Answers did not improve.** On the 10-question sample, correct answers went from 5 of 8 to 4 of 8, one answer became a refusal, and OpenAI cost rose 2.2 times. Eight questions cannot prove it is worse, but nothing shows it is better.
- **No full test exists.** Reranked answers were never run on all 40 questions.

So reranking looks good on hit@1 and MRR, and that gain has not reached the answers. Dense stays the default and reranking stays optional.

**Device filter.** Limiting search to the named device's manual raised hit@4 from 80% to 93% (on an earlier index), because manuals share wording such as "restart the transmitter". That is a large gain from a simple rule, not a clever model.

## Refusing: the central tradeoff

The system can be wrong in two ways:

- **Answering when it should not.** Dangerous, because a technician might act on it.
- **Refusing when it could have answered.** Annoying, and the user has to look it up.

Making the system stricter reduces the first and raises the second, so in practice you cannot get both to zero. This project deliberately accepts the second kind: 0 of 10 invented answers, at the price of 20% false refusals. That is why refusals are a stated design decision and not a bug.

All 10 correct refusals came from the model itself, not from the quote check or a score threshold.

**A limit of this claim.** Refusing protects against questions the manual cannot answer. It does not protect against a wrong or partial answer to a question it can answer, and 4 of the 24 answers should not be followed.

**Why not refuse based on the search score?** It seems natural: a low match means no answer. The saved run shows it would not work. The best match for unanswerable questions (median 0.60, range 0.46 to 0.69) is as high as or higher than for answerable ones (median 0.53, range 0.29 to 0.79). A question about KNX still resembles the manual's text about connections. Search scores measure "similar topic", not "contains the answer". That is why the `min_score` gate stays off.

## Cost and speed

| Metric | Result | Meaning |
|---|---|---|
| Tokens | 45,943 in, 3,376 out | **The amount of text**, for all 40 questions. A token is a word fragment. In is what the model reads (rules, 4 chunks, question). Out is what it writes |
| Answering cost | $0.0089 for 40 questions | **The money**: tokens multiplied by price. About $0.0002 per question |
| p50 latency | 1.74 s | Half of questions finish faster than this: the typical wait |
| p95 latency | 3.58 s | 95% finish faster than this: the slow case users notice |

**Tokens and cost are the same thing in different units.** Tokens are the quantity and cost is the bill: 45,943 x $0.15 per million + 3,376 x $0.60 per million = $0.0089. Tokens are the stable measure, because prices change. This covers answering calls only: the judge used a free tier and is not priced. The prices in `docrag/pricing.py` are unverified.

Percentiles are used because an average hides slow cases. Cost is tiny here, so it does not limit the design. Moving to a stronger model is affordable, which makes that fix worth testing.

## How far to trust these numbers

- **The sample is small.** One question moves a rate by about 3 points. The 95% range for 14 of 30 correct is roughly 30% to 64%, and for 28 of 30 at hit@4 roughly 79% to 98%.
- **Zero failures is not proof of zero.** 0 of 10 invented answers only supports a true rate below about 26% at 95% confidence. "All 10 refused" is good evidence, not a guarantee. A real claim of safety would need far more unanswerable questions.
- **Runs vary.** Two full runs on the same earlier index scored 47% and 57%. The later 57% to 47% change between indexes is inside that spread, so it cannot be called a regression.
- **The judge varies.** It flipped verdicts between runs, it graded the same behaviour differently within this run, and its fallback chain can switch models: this run used one model for 23 answers and another for 1. Pinning a single model fixes only the last of these.
- **The test is not independent.** The questions were mostly written by Claude, the same assistant that helped build the system, and not by real users. They may be easier or differently worded than real questions, so the scores may not carry over.
- **The answer key can be wrong.** Early on, printed page numbers were used instead of PDF page numbers, which understated scores until it was found.
- **Tuning on these 40 questions would overfit.** Settings adjusted to these questions would look better here without being better. Several tempting tweaks were rejected for this reason.

Because of this, small differences in the tables are not treated as real. Large gaps are: 93% found against 47% correct is a real signal. So is the device filter.

## Common questions

**Why is recall only 47%? Is the system bad?**
It is a strict measure, with refusals counted as wrong and any missing key fact counted as wrong, and the judge is inconsistent. Under fairer grading the figure is between 47% and 67%. The honest reading is that the system is cautious but not yet complete: no text was made up, the page is almost always found, and the answers are often partial. A user can be misled by a partial answer, because nothing warns them it is partial. For a demo the goal was to know exactly why, and the failure table shows that.

**Why does the system fail?**
Of 16 failures: 10 in answer writing (5 left out details, 3 refused vague questions, 2 wrong fault), 2 correct answers thrown away by the quote check, 2 search misses, and 2 where the answer was fine but the grading was strict.

**What would you do next?**
Fix the quote check, since it throws away correct answers. Then compare prompts on the incomplete answers, because it is cheap and targets the largest group. Then try a stronger answer model, with the judge pinned so the comparison is fair. The full list is in [what could be done differently](#what-could-be-done-differently).

**Why not hybrid search? Isn't that best practice?**
It was built and measured, and it was worse on these manuals as a final top 4. Best practice is a starting hypothesis; the measurement decides.

**Why a separate judge model?**
A model grading its own family may share its blind spots. The evidence that this matters is limited, but the cost of separating them is close to zero.

**Does retrieval quality guarantee answer quality?**
No. Better search ordering from the reranker did not produce better answers in the 10-question sample, and the first time the index was called ready on retrieval numbers alone, answer quality turned out not to have been checked ([decision log, entry 12](decision-log.md)).

**What went wrong during the project?**
The [decision log](decision-log.md) keeps these on purpose. The best example is skipping the merge of tiny chunks: a one-word chunk, `Display`, outranked the real answer, hit@1 fell from 80% to 67%, and merging recovered only part of it (70%).

**What was not built?**
Web UI, document upload, handling of figures and scans, a truly local model, and a clarifying-question flow. Gemma through an API stood in for a local-class model, and it took 1 to 3 minutes per question and was unreliable in testing. So the local route is untested, not proven.

**Would this work at scale?**
Not shown. Search is exact and in memory, which suits a few thousand chunks. Three manuals say little about thousands of documents, where finding the right manual and device gets harder. See [limitation 11](limitations.md#11-scale).

## Glossary

| Term | Meaning |
|---|---|
| RAG | Search the documents first, then have an LLM answer from what was found |
| LLM | Large language model, such as GPT or Gemini |
| Chunk | One passage of a document, the unit that search returns |
| Embedding | Numbers representing the meaning of text, so similar text can be found |
| Cosine similarity | The score for how close two embeddings are; higher is closer |
| Dense retrieval | Search by embedding similarity |
| BM25 | A classic keyword-search scoring formula |
| RRF | Reciprocal rank fusion: merges two ranked lists using positions, not scores |
| Reranker | A second step that re-scores search results |
| Golden set | Questions and answers prepared in advance and used as an answer key |
| Top k, @k | Only the first k results of the ranked list are considered |
| TP, FP, FN, TN | True positive, false positive, false negative, true negative: right and wrong decisions, by type |
| Accuracy | Share of all cases handled correctly |
| Precision | TP / (TP + FP): how much of what was given is right |
| Recall | TP / (TP + FN): how much of what should be given was given |
| F1 | A single score that balances precision and recall |
| Context recall | Share of the facts in the reference answer that appear in the retrieved text |
| nDCG | A ranking metric that gives more credit for better items placed higher |
| Hallucination | A confident answer not based on the source |
| False refusal | Saying "insufficient" when the documents do contain the answer |
| Ablation | Switching one part off to measure what it contributes |
| p50, p95 | The time that 50% or 95% of requests finish within |
| Overfitting | Tuning to the test questions so scores rise without real improvement |

## What could be done differently

**In the evaluation:**

| Change | Why | Effort |
|---|---|---|
| **Grade answers in four levels**, not yes or no: acceptable, partial and usable, not usable, wrong | Yes or no treats a harmless omission like a wrong fault, and the judge applies it inconsistently. This is the most important change | Change the judge prompt and rerun it; tokens needed |
| **Score facts.** Split each reference answer into facts, marked required or optional, and report the share covered | `hmw90-fail1` becomes 1 of 3 facts, not just "not correct". It gives a real completeness number | Rewrite 30 reference answers |
| **Match reference answers to the question** | `hpp271-no-change-5min` failed for something the user did not ask | Review 30 reference answers |
| **Ask the judge whether each claim applies to the question** | Would catch the wrong-fault answers now passed as faithful | Change the judge prompt |
| **Pin one judge model** | Removes one source of noise from comparisons | Set `GEMINI_MODELS` to one model |
| **Report two levels**, decision and answer, with TP, FP, FN and TN | One rate hides whether the system refused or answered badly | Small change in `docrag/eval/runner.py` |
| **Record the failure reason per question** | The table above was built by hand and should come out of each run | Small, after grading exists |
| **Replace precision@4** with capped precision and context recall | Precision@4 depends on a number we chose, and pages are too coarse | Capped precision is cheap; context recall needs fact labels |
| **Label relevant chunks, not only pages**, and allow several right places | Page 31 counts as FP although it gave the answer, and the right page can have the wrong row | Slow, by hand |
| **Use questions from real users**, and more unanswerable ones | The current set is small and mostly written by Claude | Slow, needs users |

**In the system:**

| Change | Fixes | Tradeoff |
|---|---|---|
| **Check quotes against the whole cited page** | 1 or 2 correct answers thrown away; not tested | Weakens the check slightly |
| **Compare prompts** that ask for every listed cause | 5 answers that left out details | May bring back padding |
| **Test a stronger answer model** | Possibly left-out details, wrong rows and refusals | Higher cost, which is tiny now |
| **Warn the user** that the manual may list more causes, and link the section | Partial answers that look complete | Slightly longer answers |
| **Ask a clarifying question** for vague input | 3 refused vague questions | New behaviour to build and test |
| **Extract tables with their structure** | Possibly the 2 wrong-fault answers | Needs a new parser and a re-embed |
| **Improve search for codes and synonyms** | 2 search misses | Reranking fixes these but loses 2 others |
