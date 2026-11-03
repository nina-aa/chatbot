# Evaluation explained

How the system is measured, what the results are, and why it fails where it does. No background in search or machine learning is assumed. Every number comes from the last full run on the current index (`data/eval/20260925-091624_…`: 40 questions, dense retrieval, `gpt-4o-mini` answers, Gemini judge) unless stated otherwise. The complete history is in [results.md](results.md).

## Contents

1. [The short version](#the-short-version)
2. [How it works](#how-it-works)
3. [How it is tested](#how-it-is-tested)
4. [Results](#results)
5. [Why the system failed](#why-the-system-failed)
6. [Hand review of the answers](#hand-review-of-the-answers)
7. [Retrieval](#retrieval)
8. [Refusing: the central tradeoff](#refusing-the-central-tradeoff)
9. [Cost and speed](#cost-and-speed)
10. [How far to trust these numbers](#how-far-to-trust-these-numbers)
11. [What could be done differently](#what-could-be-done-differently)

## The short version

**What it is.** An assistant that answers questions about equipment manuals, demonstrated on three public sensor manuals. It finds the relevant passages, writes an answer from them, attaches a checkable source to every statement, and says so when the manuals do not contain the answer.

**Who it is for.** Engineers, maintenance and service staff who need an answer from a long manual, often while something is broken.

**Why it was built this way.** A wrong instruction for industrial equipment is worse than no instruction. So the design favours refusing over guessing, and everything is measured rather than assumed.

**The result in one paragraph.** Of 30 questions the manuals can answer, the right page was among the 4 retrieved chunks for 28 (93%), an answer was given for 24 (80%), and 14 were judged correct (47%). None of the 10 questions the manuals cannot answer got an invented answer. Search is not the main problem: only 2 of the 16 failures came from search. Most came from the answer step, where the model left out details it had been given, or refused vague questions.

## How it works

Technical knowledge sits in long PDF manuals. Keyword search fails when the user's words differ from the manual's, and a general chatbot may invent a plausible but wrong procedure, register number or voltage. The approach is **RAG** (retrieval-augmented generation): search the manuals first, then let a language model (LLM) answer using only what was found.

```
question + device  ->  find 4 passages in that manual  ->  LLM writes an answer  ->  quotes are checked  ->  answer or refusal
```

A **chunk** is one passage of up to 800 characters. The manuals are split into 562 chunks: 264 for HPP271, 254 for HMW90 and 44 for WAA151. An **embedding** turns text into a list of numbers so that similar meanings land close together, which lets a search match meaning and not only exact words.

| Job | Model | Notes |
|---|---|---|
| Embeddings (search) | `text-embedding-3-small` | OpenAI. Turns chunks and questions into numbers |
| Writing answers | `gpt-4o-mini`, temperature 0 | OpenAI's low-cost "mini" tier. A normal production model, but not its most capable one. Used for all 40 questions |
| Judging answers | `gemini-3.1-flash-lite` (23 answers), `gemini-3.5-flash-lite` (1) | Google's light tier, on the free tier. A different family from the answer model |
| Reranking (optional) | `gpt-4o-mini` | Not used in the main run |
| Local-class stand-in | `gemma-4-26b-a4b-it` | Tried on 3 questions only, through an API |
| Writing the test questions | Claude Sonnet 5 | Through Claude Code, the same assistant that helped build the system. |

## How it is tested

**The golden set** (`eval/golden.jsonl`) is the answer key: 40 questions written in advance, mostly by Claude, 30 the manuals can answer and 10 they cannot. The model was Claude Sonnet 5, through Claude Code. Each answerable question has the pages holding the answer and a reference answer. Without an answer key, "it looks good" is the only measure, and that does not survive changing anything.

The questions are deliberately mixed: clean ones ("How do I reset the HMW90 to factory settings?"), vague ones ("sensor info is wrong"), and bare codes ("Internal 1"). The 10 unanswerable ones sound reasonable but are not in the manuals ("Can the HMW90 be connected to a KNX building automation bus?").

**Why measure in layers.** One overall score cannot tell you what to fix. If an answer is wrong, either search did not find the page, or it did and the model mishandled it. The fixes are completely different, so the two are measured separately.

## Results

Every one of the 40 questions lands in exactly one of these outcomes:

| Outcome | Count | Share of 40 |
|---|---|---|
| Answerable, answered, judged correct | 14 | 35% |
| Answerable, answered, judged not correct (partial or wrong) | 10 | 25% |
| Answerable, refused | 6 | 15% |
| Unanswerable, correctly refused | 10 | 25% |
| Unanswerable, answered (invented) | 0 | 0% |

The metrics are scored at two levels. **Decision** asks whether answering or refusing was the right call. **Answer** asks whether the answer was right. Precision is about trust (when it answers, is it right?), recall about usefulness (when an answer exists, does it give it?).

| Level | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Decision:** should it answer or refuse? | 85% (34 of 40) | 100% (24 of 24) | 80% (24 of 30) | 89% |
| **Answer:** was the answer right? | 60% (24 of 40) | 58% (14 of 24) | 47% (14 of 30) | 52% |

| Retrieval (dense search) | Result |
|---|---|
| hit@4: a right page was among the 4 chunks | 93% (28 of 30) |
| page recall@4: share of the right pages found | 89% |
| MRR: how high the first right chunk was | 0.81 |

**Why 80% answered but only 47% correct.** An answer is shown when the model says the text is sufficient and at least one quote passes the check. Nothing checks completeness at answer time, so a partial answer counts as answered. Only the judge, which needs a reference answer and therefore runs only during evaluation, marks it not correct. That is the gap of 10 questions.

## Why the system failed

16 of the 30 answerable questions did not end up with a correct answer. Each failure is given its one main reason, from reading the saved answers, reference answers and refusal reasons.

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

**What this means.** Only 2 of 16 failures were search failures, so better search will not move 47% much. The effort belongs in the answer step and the quote check. The two thrown-away correct answers are the cheapest to fix: the quote did not match any single chunk word for word, likely because it straddles two chunks. The wrong-fault answers pass every check because the quote check only proves the words exist on the cited page, and the right page holds the whole troubleshooting table.

## Hand review of the answers

The judge gives only yes or no, and it graded the same behaviour differently in different questions. So the 10 answered-but-not-correct answers were also reviewed by hand, using four grades:

| Grade | Rule | Count |
|---|---|---|
| **Acceptable** | Answers what was asked, and acting on it would not mislead | 2 |
| **Partial, usable** | True and helpful, but the manual lists more causes or fixes that the answer does not mention | 4 |
| **Not usable** | Too vague to act on, or missing a required step | 2 |
| **Wrong** | Addresses the wrong fault | 2 |

The official numbers do not change. The review shows what happens under fairer grading:

| What counts as a good answer | Good answers | Precision (of 24) | Recall (of 30) | F1 | Accuracy (of 40) |
|---|---|---|---|---|---|
| **Judge, official** | 14 | 58% | 47% | 52% | 60% |
| Judge + acceptable | 16 | 67% | 53% | 59% | 65% |
| Judge + acceptable + partial usable | 20 | 83% | 67% | 74% | 75% |

So the honest summary is a range: **between 14 and 20 of 30 answered usefully, and 4 of 24 answers should not be followed.** This review was done by Claude, and none of the 14 correct answers was downgraded.

The judge also rated all 24 answers faithful. That is too generous: no answer contained made-up text, but on a strict reading about 21 of 24 were faithful, because a real quote applied to the wrong fault still makes a false claim.

## Retrieval

Search scores every chunk in the named device's manual by similarity to the question and keeps the best 4. **hit@4** (was a right page among them?) is the main number, since it decides whether the model could see the answer at all. **Page recall@4** matters when an answer spans pages. **MRR** rewards a right chunk near the top, but order matters little here, because the model reads all 4. Precision@4 is not a headline number: with a fixed 4 chunks and as few as 2 relevant ones, its ceiling is often 50%.

| Method | How it works | hit@1 | hit@3 | hit@4 | page recall@4 | MRR | precision@4 |
|---|---|---|---|---|---|---|---|
| **dense** (default) | Match by meaning, using embeddings | 70% | 93% | 93% | 89% | 0.81 | 42% |
| **hybrid** | Combine meaning search with keyword search (BM25) | 67% | 83% | 87% | 81% | 0.75 | 38% |
| **hybrid + rerank** | An LLM re-scores hybrid's top 8 and keeps the best 4 | 83% | 93% | 93% | 87% | 0.88 | 42% |

- **Hybrid did worse.** Keyword search was added for exact codes such as `Internal 1`, but its noise on chatty questions pushed good results down.
- **Reranking improves ordering only.** It finds the same 28 of 30, and on a 10-question sample answers did not improve while cost rose 2.2 times. Dense stays the default.
- **The device filter mattered most.** Limiting search to the named device's manual raised hit@4 from 80% to 93% (on an earlier index), because manuals share wording such as "restart the transmitter".

## Refusing: the central tradeoff

The system can be wrong in two ways:

- **Answering when it should not.** Dangerous, because a technician might act on it.
- **Refusing when it could have answered.** Annoying, and the user has to look it up.

Making the system stricter reduces the first and raises the second, so in practice you cannot get both to zero. This project deliberately accepts the second kind: 0 of 10 invented answers, at the price of 20% false refusals. That is why refusals are a stated design decision and not a bug.

All 10 correct refusals came from the model itself, not from the quote check or a score threshold.

**A limit of this claim.** Refusing protects against questions the manual cannot answer. It does not protect against a wrong or partial answer to a question it can answer, and 4 of the 24 answers should not be followed.

**Why not refuse based on the search score?** It seems natural: a low match means no answer. The saved run shows it would not work. The best match for unanswerable questions (median 0.60, range 0.46 to 0.69) is as high as or higher than for answerable ones (median 0.53, range 0.29 to 0.79). A question about KNX still resembles the manual's text about connections. Search scores measure "similar topic", not "contains the answer". That is why the `min_score` gate stays off.

## Cost and speed

| Metric | Result |
|---|---|
| Tokens, 40 questions | 45,943 in, 3,376 out |
| Answering cost | $0.0089 for 40 questions, about $0.0002 per question |
| p50 latency (typical wait) | 1.74 s |
| p95 latency (slow case) | 3.58 s |

Cost covers answering calls only; the judge ran on a free tier. Cost is tiny, so it does not limit the design, and testing a stronger answer model is affordable.

## How far to trust these numbers

- **The sample is small.** One question moves a rate by about 3 points. The 95% range for 14 of 30 correct is roughly 30% to 64%.
- **Zero is not proof of zero.** 0 of 10 invented answers only supports a true rate below about 26% at 95% confidence.
- **Runs and the judge vary.** Two full runs on the same earlier index scored 47% and 57%, and the judge flipped verdicts between runs.
- **The test is not independent.** The questions were mostly written by Claude, not by real users.

So small differences in the tables are not treated as real. Large gaps are: 93% found against 47% correct is a real signal, and so is the device filter.

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

5 of the 16 failures are left-out details. The obvious next experiment is a prompt that says "answer only what was asked, but include every cause or step the manual lists for it".
