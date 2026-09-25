# Decision log

Append-only, oldest first. It records what was decided, why, what happened, and whether the decision held up. Wrong decisions stay in: the point is to show how the project actually went, not to present it as planned. Entries are written after the fact; dates are 2026-09-24 and 2026-09-25 unless stated. "Owner" is the project owner, "assistant" is the AI assistant that built it with them.

### 1. Cloud APIs, no local model
- **Decision.** OpenAI for answers and embeddings, Gemini as the judge and fallback, Gemma reached through the Gemini API as a stand-in for a local-class model.
- **Why.** The quickest route to a working demonstration of the text MVP.
- **What happened.** It works. The Gemma path has never been called against the real API (limitation 10), so "what a local-class model can do" is claimed but untested.
- **Verdict.** Kept. One open item: test Gemma.

### 2. A device is required before anything is processed
- **Decision.** The user must name the device (manual); retrieval is limited to that manual.
- **Why.** The manuals share wording ("restart the transmitter"), so answers were contaminated across manuals.
- **What happened.** Measured: hit@4 80% without the filter, 93% with it (limitation 7).
- **Verdict.** Right. The cost is that cross-manual questions are impossible and the user must know the device.

### 3. Refuse rather than guess
- **Decision.** "The documentation does not contain enough information" is a valid, preferred answer.
- **Why.** A wrong instruction for industrial equipment is worse than no instruction.
- **What happened.** All 10 unanswerable questions were refused in every full run, with no hallucinated answer. The price is 20 to 23% of answerable questions being refused (limitation 2).
- **Verdict.** Right, and the price is stated.

### 4. Golden-set page numbers (a wrong assumption, fixed)
- **Decision.** The golden set first used the printed page numbers.
- **What happened.** For HMW90 the PDF page index is the printed number plus 2, so the golden pages were wrong and retrieval scores were understated until it was fixed (limitation 9).
- **Verdict.** Wrong. Numbers measured before the fix were misleading. Lesson: check the ground truth before tuning against it.

### 5. Own BM25 plus reciprocal rank fusion, dense stays the default
- **Decision.** Build keyword search and a hybrid, but keep dense as the default retriever.
- **Why.** Embeddings miss exact codes (`Internal 1` ranked 7 of 691).
- **What happened.** BM25 fixed that case, but equal-weight fusion let keyword noise on chatty questions push good dense hits down: hybrid hit@1 was below dense at every measurement (limitation 6). Removing stopwords helped (hybrid MRR 0.72 to 0.78) but did not close the gap.
- **Verdict.** Keeping dense as the default was right; hybrid is useful for recall at 8, not as the final top 4.

### 6. The LLM reranker: opt-in, and a first verdict from samples that could not show the benefit
- **Decision.** Build it behind `--rerank`, off by default, then evaluate it.
- **What happened.** A 3-question smoke run and a 10-question sample first suggested it was not worth it (answers did not improve, cost about doubled). But the random sample contained no question that dense retrieval missed, so it had nothing to fix. A targeted set of dense misses then showed it did fix retrieval where dense failed (and could import a conflicting chunk on another question). Full 40-question retrieval runs, twice (before and after the chunk merge): best hit@1 and MRR, tied at hit@4, and it lost two questions dense finds (`hmw90-relay`, never in hybrid's top 8; `waa151-dont-work`, in the pool but scored low).
- **Verdict.** Not adopted by default, and for reasons measured rather than assumed. The wrong step was judging from a sample that could not contain the cases where it might help.

### 7. Quote-check fix for table labels: proposed by the assistant, rejected by the owner
- **Proposal.** Ignore `Possible cause:` and `Solution:` labels when matching quotes, because one answer was dropped for that in a 2-question smoke run.
- **Why rejected.** One example, tuned to the case at hand, and it loosens the check that carries the trust claim.
- **Verdict.** The owner was right. It stays documented as limitation 3 with the fix and its tradeoff.

### 8. Running-header stripping: generic, but the threshold was too high
- **Decision.** Remove lines that repeat (digits ignored) at the top or bottom of at least 25% of a document's pages. The owner asked whether this was overfitting; the rule was made generic (no manual-specific pattern) to answer that.
- **What happened.** It removed 9% of the HMW90 text and under 2% of the others, but per-chapter running heads (`Chapter 4 ____ Operation`) appear on only 4 to 15% of pages and survived: 31 chunks made only of a header remnant were left in the index (counted before the merge). Lowering the threshold would start removing real table headers (`Register`, `number`), so it was not lowered.
- **Verdict.** Partly wrong (recall too low), left as a documented limitation. Keeping the headers would not have been better (see 10).

### 9. Skipping the merge of tiny chunks (a wrong call, recommended by the assistant)
- **Decision.** Heading-aware splitting had produced many small chunks. The assistant recommended skipping the merge (it looked like a judgment call, and the owner had said perfection is not necessary); it was not pursued.
- **What happened.** After the re-embed, dense hit@1 fell from 80% to 67% and MRR from 0.86 to 0.78. Chunks under 100 characters (13% of the index) took 11 of 120 top-4 slots against 1 of 120 before, and only 1 of those 11 was on a golden page: a one-word chunk `Display` outranked the real answer for "display shows Internal 3".
- **Reversal.** Chunks under 120 characters are now merged into the next chunk on the page. That recovered part of it (hit@1 70%, MRR 0.81). It did not recover the rest, which is chunk-boundary effects and a bare-code query that was luck before (limitation 14).
- **Verdict.** Wrong. The chunk count going from 691 to 859 was a warning sign that was noted but not acted on. Lesson: a cheap change that alters every chunk boundary should be measured before it is skipped or adopted.
- **A second miss in the same episode.** The first explanation given for the drop ("new chunk boundaries and section labels shift which chunk holds the answer") was a hypothesis; the actual mechanism was found only by reading the hits of the moved questions.

### 10. Header stripping or the old underscores? Tested, no difference
- **Question (owner).** Would keeping the header lines (underscores included) have been better, given the old index scored higher?
- **What was done.** Two variants were built and evaluated on the same 30 questions: A (merge, headers stripped) and B (merge, headers kept). A had hit@1 70%, MRR 0.81; B 70%, 0.80.
- **Verdict.** No measurable difference; A kept because it has fewer junk chunks, not because it was shown to be better. The old index could not be rebuilt (its chunker is no longer in the repository), so the remaining gap to it could not be isolated.

### 11. Adopting the new index
- **Decision.** Keep variant A instead of going back to the old index.
- **Why.** The old index had the DKP manual (which must not be embedded), old section labels, and header noise; going back was not possible anyway.
- **What happened.** The index was embedded twice (515 chunks, then 69 more after the merge). Both estimates are upper bounds and add up to under $0.003; unchanged text comes from the embedding cache.

### 12. Calling the index "ready" on retrieval metrics alone (a reporting mistake)
- **What happened.** The re-embed and the chunk merge were checked with retrieval-only runs. Answer quality on the new index was not measured until hours later. The first answer run (10 questions) showed 2 of 8 correct against 5 of 8 in an older run, and the comparison was reported against an older run without stressing that it was on a different index, which led to the reasonable question whether pgvector had changed the results (it had not: that run used the numpy store). A full 40-question run then gave 14 of 30 correct (47%) against 17 of 30 (57%) on Sep 24.
- **What that run does and does not show.** The 3-question difference is inside the spread of the two Sep 24 full runs on the same index (47% and 57%), so it cannot be called a regression. It does agree with the retrieval numbers (hit@1 down, citations on a relevant page down from 100% to 88%). The judge also changed: with the Gemini fallback chain, 4 of the 11 questions whose outcome changed were judged by a different model (3.5-flash-lite before, 3.1-flash-lite now), and the direction of those flips was mixed (three went down, one went up).
- **Verdict.** "Ready" was claimed too early, and the judge model should be pinned for any comparison that goes into the docs.

### 13. pgvector behind a flag, default unchanged
- **Decision (owner).** The Postgres store is selected with `--store pgvector`; numpy stays the default, so it can be reverted by dropping the flag. The interface's `save`/`load(directory)` became a no-op and an error for Postgres instead of refactoring the interface into a factory (the smaller change).
- **A wrong guess.** A 50 ms search time (against 0.3 ms for numpy) was first suspected to be JIT compilation. Turning JIT off changed nothing. Timing query variants showed the cause: the query vector was passed twice, once in the select list and once in `ORDER BY`. Computing the distance once in a subquery brought it to about 3 to 7 ms.
- **Verified.** 720 searches on the real index returned the same chunks in the same order as numpy; a full 40-question retrieval run through `--store pgvector` returned the same chunks in the same order for all 40 questions, and a 10-question answer run returned the same chunks, the same answers and the same judge verdicts (see the update in the pgvector section).
- **A tooling slip.** A scripted file write dropped an escape and produced a broken f-string; the import check caught it. It is why the working agreements say to assert that scripted replacements matched.
- **Verdict.** Held up. Not done: a delete-document method, and the CI job has never run.

### 14. What stays out of the public repository
- **Decision (owner).** The evaluation runs (`eval/runs/`) are not for GitHub because they contain manual text in the saved chunks, and the manuals (`pdfs/`) are left out because their copyright status is unclear. The old top-level PDFs remain in the first commit's history.
- **Consequence.** Cloning the repository is not enough to reproduce the numbers: the manuals must be added by hand (see the setup guide). The CI tests were written to skip cleanly without them.

### 15. Gemma as the local-class stand-in: never called until the end (a gap that hid a dead default)
- **Decision (entry 1).** Reach Gemma through the Gemini API instead of running a local model. The path was written and unit-tested with fakes, and never called.
- **What happened when it was.** The default model `gemma-3-12b-it` had been retired, so every call was a 404 and the first smoke run errored on all three questions. The two Gemma models the key can see (`gemma-4-26b-a4b-it`, `gemma-4-31b-it`) were then unreliable: 500s, a 503 "high demand", and calls with no answer after 4 minutes, for every variant of the prompt that was tried. Digging into it found two real bugs in our client (a 500 crashed the whole eval, and there was no timeout), both fixed with tests. With a timeout, the smoke set finished: 1 answer (valid, all quotes verified, 98 s), 1 correct refusal, 1 timeout.
- **A slip along the way.** The first attempt at a hard cutoff used a thread pool whose context manager waits for the stuck call, so it did not cut anything off and cost several minutes; a daemon thread with a hard exit did.
- **Verdict.** Wrong to leave a provider path uncalled for the whole project: "tests pass with fakes" said nothing about a model name that no longer existed. The claim "Gemma shows what a local-class model can do" is now backed by one real answer and one honest finding that it is slow and flaky here, not by a benchmark.


