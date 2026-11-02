# Run: `gpt-4.1-mini` as the answer model

A full 40-question run on 2026-11-02 that changes one thing from the run published in [evaluation-explained.md](evaluation-explained.md): the answer model, `gpt-4o-mini` to `gpt-4.1-mini`. Everything else is the same: the 562-chunk index, dense retrieval, top 4, device filter, the same prompt and the same Gemini judge chain.

- New run: `data/eval/20261102-101629_openai_judge-gemini`
- Baseline: `data/eval/20260925-091624_openai_judge-gemini`
- How it was run: `OPENAI_CHAT_MODEL=gpt-4.1-mini` set in the environment for this one command, then `python -m docrag eval eval/golden.jsonl --llm openai`. No code or config file was changed.

## Which Gemini model judged

Each saved result records the model that actually answered the judge call (`judge_model` in `results.jsonl`). The judge only runs on answered questions, since a refusal has nothing to grade.

| Run | Judge calls | Model used | Judge errors |
|---|---|---|---|
| Baseline, 2026-09-25 (the published numbers) | 24 | `gemini-3.1-flash-lite` 23, `gemini-3.5-flash-lite` 1 (`hmw90-htm10-01-02`) | 0 |
| This run | 27 | `gemini-3.1-flash-lite` 27 | 0 |

**The free tier did not block either run.** No judge call failed. In the baseline, one call fell back to the second model in the chain. Why it fell back (quota, overload or timeout) is not recorded, because `GeminiLLM` keeps the reasons only when every model fails. Since both runs used `gemini-3.1-flash-lite` for all but one call, the judge was in effect pinned, and the comparison below is fair on that point.

## Results side by side

| Metric | `gpt-4o-mini` | `gpt-4.1-mini` |
|---|---|---|
| Answered (of 30 answerable) | 80% (24) | 90% (27) |
| False refusals | 20% (6) | 10% (3) |
| **Judged correct** (of 30) | **47% (14)** | **60% (18)** |
| Faithful (judge, of answered) | 100% | 100% |
| Cited a relevant page (of answered) | 88% (21 of 24) | 93% (25 of 27) |
| Unanswerable correctly refused | 100% (10) | 90% (9) |
| Answered an unanswerable question | 0 | 1 (`unans-wifi`, see below) |
| Answers with a statement dropped by the quote check | 0% | 14% |
| Retrieval: hit@4, MRR | 93%, 0.81 | 93%, 0.81 (same index, same search) |
| Answer tokens, in / out | 45,943 / 3,376 | 45,873 / 5,833 |
| Answer cost | $0.0089 | about $0.028 |
| Latency p50 / p95 | 1.74 s / 3.58 s | 1.62 s / 2.77 s |

The cost for `gpt-4.1-mini` is not in `docrag/pricing.py`, so the run printed "n/a". It was worked out by hand at $0.40 in and $1.60 out per million tokens (OpenAI list price as last known, not verified): 45,873 x 0.40 + 5,833 x 1.60, divided by a million, is $0.028. That is about 3 times the baseline, and still well under a cent per question.

## What changed, question by question

28 of the 40 questions came out the same. 12 changed:

| Question | `gpt-4o-mini` | `gpt-4.1-mini` | What happened |
|---|---|---|---|
| `hmw90-fail1` | not correct | correct | Now lists the causes it used to leave out |
| `hmw90-relay` | not correct | correct | No longer just repeats the symptom |
| `hmw90-htm10-05` | not correct | correct | Same kind of answer; the judge passed it this time |
| `hpp271-no-change-5min` | not correct | correct | Same kind of answer; the judge passed it this time |
| `hmw90-internal-3-sentence` | not correct | correct | **Still the wrong fault** (see below) |
| `hpp271-filtering-factor` | refused | correct | Its quote now passes the quote check |
| `hpp271-error-persists` | refused | correct | Vague question, now answered |
| `hpp271-reading-incorrect` | refused | not correct | Vague question, now answered with 1 of many causes |
| `hmw90-temp-high` | correct | not correct | Dropped the second cause (mounting orientation) |
| `hpp271-ppm-no-h2o2` | correct | not correct | Dropped what to do if the reading persists |
| `hpp271-no-signal` | correct | not correct | Gave 1 of 3 causes; the old answer gave all 3 |
| `unans-wifi` | refused | answered | See below |

Unchanged failures: `hmw90-internal-1`, `hpp271-wont-power-on` and `hpp271-insight-reset-loop` are still refused, and `hmw90-factory-reset`, `hpp271-dont-work`, `waa151-bent-connector`, `waa151-heater-ice` and `hpp271-chemicals-slow` are still not correct.

Net: 7 more judged correct, 3 fewer, so 14 to 18.

## Caveats

- **One of the 18 is a judge error.** `hmw90-internal-3-sentence` gives the same wrong-fault answer as before ("restart, replace the module", which is the remedy for an HTM10 module problem, not Internal 3). The judge failed it last time and passed it this time, with the reason "correctly identified the remedy for Internal 3". On the hand review used in [evaluation-explained.md](evaluation-explained.md) it is still **wrong**, so a fair count is 17 of 30 (57%).
- **Two of the gains are the judge, not the model.** `hmw90-htm10-05` and `hpp271-no-change-5min` were already graded "acceptable" by hand in the baseline. The judge passing them now says more about the judge than about `gpt-4.1-mini`.
- **The new model also leaves things out.** The three regressions are all the same failure as before: details that were in the retrieved chunks were left out. The prompt rule "Answer only what was asked" still applies, and `gpt-4.1-mini` follows it too, just on different questions.
- **The difference is inside the noise.** Two baseline runs on an earlier index scored 47% and 57%. 18 of 30 has a 95% range of roughly 42% to 75%. So "better" is a reasonable reading, not a proven one. Running it a second time would show how stable it is.

## The `unans-wifi` answer

Question: "Can the WAA151 anemometer send data over Wi-Fi?" The manual does not mention Wi-Fi. The answer was:

> The WAA151 anemometer outputs a pulse signal proportional to wind speed via a phototransistor, not Wi-Fi.

It quotes page 7 correctly about the pulse output. The words "not Wi-Fi" are the model's own inference, not in the quote. The scoring counts any answer to an unanswerable question as a hallucination, so the rate went from 0% to 10%. In substance the answer is probably right and not harmful: the sensor's only output is a pulse signal. But it is exactly the behaviour the design wants to avoid, which is going beyond what the text says. `gpt-4o-mini` refused this question. On 10 unanswerable questions, one case cannot show whether `gpt-4.1-mini` is less cautious in general.

## Reading of the result

- The stronger model helps with the problems the baseline blamed on the answer step. It refused fewer vague questions (6 to 3 refused), and it completed some of the partial answers.
- It does not fix leaving things out. That points at the prompt as much as the model, so the prompt comparison in [evaluation-explained.md](evaluation-explained.md#what-could-be-done-differently) is still worth doing.
- It traded a little caution for more answers: one unanswerable question answered, and more statements dropped by the quote check (14% of answers against 0%), so it writes more claims that cannot be matched word for word.
- Cost rose about 3 times, from under a cent to about 3 cents for 40 questions. Cost is still not what limits the design.
