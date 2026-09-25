# Results (full history)

Every measurement made during development, newest index first for the current numbers and older ones kept for comparison. The summary is in the [README](../README.md#results).

Golden set of 40 questions, `text-embedding-3-small`, answers by `gpt-4o-mini`, judge Gemini. **Small sample: one question moves a rate by about 3 points**, so read these as direction, not benchmark. The judge is an LLM and is not perfectly stable (see limitation 9).

| Run (device filter on unless noted) | hit@1 | hit@4 | MRR | answered | judged correct | faithful | unanswerable refused |
|---|---|---|---|---|---|---|---|
| dense, **no** device filter | 63% | 80% | 0.70 | 70% | 14/30 | 95% | 10/10 |
| dense, device filter | 80% | 93% | 0.86 | 77% | 17/30 | 100% | 10/10 |

**Current index** is variant A of [limitation 14](limitations.md#14-the-dense-regression-after-the-re-embed-tiny-chunks-outrank-content) (562 chunks: headers stripped, chunks under 120 characters merged). Retrieval only, 30 answerable questions, device filter, top 4 (runs `data/eval/20260925-0715…`, `…0827…`, `…0828…`, not yet copied to `eval/runs/`):

| retriever | hit@1 | hit@3 | hit@4 | precision@4 | page recall@4 | MRR |
|---|---|---|---|---|---|---|
| dense | 70% | 93% | 93% | 42% | 89% | 0.81 |
| hybrid (RRF) | 67% | 83% | 87% | 38% | 81% | 0.75 |
| hybrid + rerank | 83% | 93% | 93% | 42% | 87% | 0.88 |

Reranking cost $0.0112 for the 40 questions (61,266 in / 3,280 out tokens). Reranking still gives the best hit@1 and MRR (+13 points and +0.07 over dense), but it no longer beats dense at hit@4 (93% both) and page recall@4 is now lower (87% vs 89%). It also loses two answerable questions that dense finds; see [limitation 6](limitations.md#6-hybrid-and-the-reranker-have-not-beaten-dense-as-a-final-top-4). The table below is the same comparison one step earlier, on the 637-chunk index before the merge (rerank hit@4 97%, MRR 0.89): the difference is two questions and one run each, so read the two as the same result within noise.

**After the re-embed, before the merge** (637 chunks, running headers stripped, DKP gone; retrieval only, 30 answerable questions, device filter, top 4; runs `data/eval/20260925-0658…` to `…0659…`, not yet copied to `eval/runs/`):

| retriever | hit@1 | hit@3 | hit@4 | precision@4 | page recall@4 | MRR |
|---|---|---|---|---|---|---|
| dense | 67% | 90% | 93% | 42% | 89% | 0.78 |
| hybrid (RRF) | 63% | 83% | 87% | 38% | 81% | 0.74 |
| hybrid + rerank | 83% | 97% | 97% | 42% | 91% | 0.89 |

The reranking cost $0.0109 for the 40 questions (58,931 in / 3,496 out tokens). Dense fell from the table below (hit@1 80% to 67%, MRR 0.86 to 0.78) even though nothing about dense retrieval changed except the index; 5 of 30 questions moved (`hmw90-internal-1` from rank 7 to not retrieved, and four HMW90/HPP271 questions from rank 1 down to 2 to 4). The likely cause is tiny chunks, see [limitation 14](limitations.md#14-the-dense-regression-after-the-re-embed-tiny-chunks-outrank-content); not yet confirmed by a counterfactual. Retrieval-only says nothing about answer quality: that needs a full answer run.

Before the re-embed (old index, kept for comparison). Retrieval-only ablation, device filter on, top 4 (and top 8 for the last column):

| retriever | hit@1 | hit@4 | precision@4 | page recall@4 | MRR | hit@8 |
|---|---|---|---|---|---|---|
| dense | 80% | 93% | 43% | 91% | 0.86 | 97% |
| keyword (BM25) | 63% | 83% | 37% | 79% | 0.72 | 83% |
| hybrid (RRF) | 70% | 87% | 39% | 81% | 0.78 | 100% |

**Answer quality on the current index (sample of 10, `eval/golden_sample10.jsonl`, dense, `gpt-4o-mini` answers at temperature 0, Gemini judge; run `data/eval/20260925-090103_…`).** 8 answerable, 2 unanswerable. Answered 7 of 8, **judged correct 2 of 8**, faithful 100%, cited a relevant page 86%, both unanswerable questions correctly refused, hit@4 100%, MRR 0.81, no dropped statements, no errors. Answering cost $0.0022, p50 latency 1.98 s, p95 4.26 s (judge tokens are free-tier and not priced). The last full dense run on the old index got 5 of these same 8 correct, so this is worse, and the cause is not one thing:
- Five answers were judged *incomplete but faithful* (`hmw90-relay`, `hmw90-factory-reset`, `hpp271-no-change-5min`, `hpp271-dont-work`, `waa151-heater-ice`) and one question was refused (`hpp271-wont-power-on`, it was answered before). This is limitation 1 again. In a rough substring check of the four retrieved chunks, the detail the judge said was missing was **present** in the context for `relay` (rotary switch, hysteresis, STATUS), `factory-reset` (DIP switch, PASS 9000), `no-change-5min` (5 minutes, scaling, error level) and `heater-ice` (heating element, connector), so the model left it out; it was absent for `relay`'s RTEST command, `factory-reset`'s firmware version and `dont-work`'s pin 5, so retrieval also lost some detail.
- Two answers got much shorter than in the old run: `hmw90-relay` 4 statements to 1, `hmw90-factory-reset` 7 to 2. The answer model is at temperature 0, so this is not sampling noise: the retrieved chunks changed with the new chunk boundaries (limitation 14) and the model's answer changed with them. Which part is retrieval and which is the model's own tendency to stop early was **not separated**.
- The judge is also part of the difference: `hpp271-no-change-5min` had the same one-statement answer as before and was judged correct then and incomplete now. Eight questions cannot tell a real regression from this noise; a full 40-question run on the current index (about $0.01) has not been done, and the numbers in the first table above are still from the old index.

**Full answer run on the current index (40 questions, dense, numpy store, `gpt-4o-mini` at temperature 0, Gemini judge; run `data/eval/20260925-091624_…`, 2026-09-25).** This replaces nothing above: the first table of this section is still the Sep 24 run on the old index, and the two are shown side by side.

| | Sep 24 full run, old index | now, 562-chunk index |
|---|---|---|
| hit@1 / hit@4 / MRR | 80% / 93% / 0.86 | 70% / 93% / 0.81 |
| answered | 77% (23 of 30) | 80% (24 of 30) |
| false refusals | 23% | 20% |
| **judged correct** | **57% (17 of 30)** | **47% (14 of 30)** |
| faithful | 100% | 100% |
| cited a relevant page | 100% | 88% |
| unanswerable questions refused | 10 of 10 | 10 of 10 |
| answering cost / p50 / p95 latency | $0.0095 / 1.58 s / 3.31 s | $0.0089 / 1.74 s / 3.58 s |

What changed, question by question (the 12 questions that answered correctly both times stayed correct):
- Went down: `hmw90-fail1`, `hmw90-relay`, `hmw90-factory-reset`, `hpp271-no-change-5min` (correct to incomplete but faithful), `hpp271-filtering-factor` (correct to refused), `hpp271-wont-power-on` (incomplete to refused).
- Went up: `hpp271-ppm-no-h2o2` and `hpp271-manual-purge-modbus` (incomplete to correct); `hmw90-internal-3-sentence`, `hmw90-htm10-05` and `hpp271-chemicals-slow` went from refused to answered but incomplete.
- Refused now: `hmw90-internal-1`, `hpp271-filtering-factor`, `hpp271-reading-incorrect`, `hpp271-wont-power-on`, `hpp271-insight-reset-loop`, `hpp271-error-persists` (four of them were also refused on Sep 24).

**How to read it.** Correct fell by 3 questions, but the two Sep 24 full runs on the same index scored 47% and 57% (the first table used the second), so a 3-question move is inside the run-to-run spread and is not evidence of a regression. What agrees with the retrieval numbers (limitation 14): hit@1 and MRR are lower and citations landing on a relevant page fell from 100% to 88%. Two answers got much shorter with the new chunks (`hmw90-relay` 4 statements to 1, `hmw90-factory-reset` 7 to 2). **The judge is a confound:** with the Gemini fallback chain, 4 of the 11 questions whose outcome changed were judged by a different model this time (3.5-flash-lite before, 3.1-flash-lite now), and the flips went both ways (three down, one up), so the judge adds noise in both directions. Pinning the judge (one model in `GEMINI_MODELS`) would remove that for future comparisons. Faithfulness stayed at 100% and no unanswerable question was answered.

Full dense run cost about $0.0095 for 40 answers, p50 latency 1.6 s, p95 3.3 s (judge tokens are free-tier Gemini and not priced). Precision@4 is low for every retriever partly by design: questions have one or two golden pages but four chunks are retrieved, often from the same page.
