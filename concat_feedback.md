# Front Matter and Abstract

## Overall assessment

This draft is clearly much stronger than an early concept-stage manuscript. The thesis topic is now technically focused, the contribution is understandable, and the evaluation appears substantially more developed.

The front matter still needs careful revision before submission, mainly because the abstract is too long, a few template details are incorrect, and some claims are stated more strongly than the chapter evidence supports.

---

## Issues to address

### 1. Must fix: Shorten the abstract to one page and make it more selective

**What is wrong:** The abstract currently reads more like a mini results chapter. It includes many implementation details, domain counts, model names, percentage ranges, and cost figures.

**Why this matters:** The abstract should be a concise summary of the problem, method, artifact, main findings, and conclusion. If it becomes too detailed, the central message becomes harder to see and the university template requirement is not met.

**How to fix:** Reduce it to a tighter structure: problem, objective, method, artifact, main findings, and conclusion. Keep only the most important quantitative results and move secondary detail back into Chapters 5 and 6.

### 2. Must fix: Correct the supervisor title and front-matter formatting

**What is wrong:** The supervisor line currently says **"Sami Ben , Principal Lecturer"**. You also mentioned that the correct title should be **Senior Lecturer**, and the current line has an unnecessary space before the comma.

**How to fix:** Please change the line to:

> **Supervisors: Sami Ben Cheikh, Senior Lecturer**

Also review the front-matter labels for consistency with the university template.

### 3. Must fix: Remove or repair malformed citation markup in the abstract

**What is wrong:** The abstract contains a URL-style citation inside the running text: **effect[https://arxiv.org/abs/2307.03172]**.

**Why this matters:** This breaks the academic style and looks unfinished.

**How to fix:** Replace it with the thesis's normal in-text citation format and make sure the relevant source is already present in the reference list.

### 4. Must fix: Correct template and spelling issues in the front matter

**What is wrong:** There are several visible presentation issues such as **"Tite"** instead of **"Title"**, duplicate appendix placeholders in the contents, and a few inconsistent labels.

**Why this matters:** These are small issues individually, but together they make the manuscript look less polished than the technical work deserves.

**How to fix:** Do one final template-cleaning pass across the title page, abstract page, contents page, and abbreviations page.

### 5. Should fix: Make the abstract claims slightly more cautious

**What is wrong:** Some statements are close to conclusion-level claims, especially around latency budgets, savings, and broad cross-model effectiveness.

**How to fix:** Keep the results positive, but phrase them more proportionally. For example, say that pruning **substantially reduced payload size and often reduced token-related cost and latency**, rather than implying the same effect under all conditions.

# Appendices

## Overall assessment

The appendix section is currently incomplete.

This is important to fix, because the thesis would benefit from supporting material that strengthens transparency and reproducibility.

---

## Issues to address

### 1. Must fix: Replace the placeholder appendix with real content or remove it

**What is wrong:** The appendix currently contains only placeholder text such as **"Title of the Appendix"** and **"Content of the appendix is placed here"**.

**Why this matters:** Empty appendices look unfinished and weaken the presentation quality of the whole thesis.

**How to fix:** Add real appendix material, such as benchmark questions, scoring criteria, selected schemas, CLI examples, or supplementary result tables. If no appendix material is needed, then remove the placeholder appendix entirely and update the table of contents.

### 2. Must fix: Remove duplicate appendix titles from the table of contents

**What is wrong:** The table of contents currently lists the appendix title twice.

**How to fix:** Clean the appendix structure and regenerate the contents after the appendix is finalized.

# Detailed Thesis Audit Addendum

## Purpose of this addendum

The earlier chapter feedback focused on the highest-priority revision points. This addendum is more exhaustive. It captures additional manuscript issues that should be checked before final submission so that smaller structural, consistency, and presentation problems are not missed.

---

## Front matter and table of contents

### 1. Must fix: Title-page and front-matter presentation still look unfinished

**Observed issues:**
- The front matter begins with **"Tite"** instead of **"Title"**.
- The supervisor line is incorrect and should use **Senior Lecturer**, not **Principal Lecturer**.
- The supervisor name is shortened to **"Sami Ben"** instead of the full form used elsewhere.
- The professional-major field still appears generic and should be checked against the actual degree-program template.

**Why this matters:** These are visible first-page issues. They weaken the professional presentation of the whole thesis.

### 2. Must fix: The abstract is too long and too results-heavy

**Observed issues:**
- It is clearly beyond the normal one-page expectation.
- It reads almost like a compact results section rather than a thesis abstract.
- It includes many details that belong later: exact domain types, measurement counts, cost percentages, and interpretation of boundary conditions.

**How to fix:** Reduce it to the minimum needed to communicate problem, method, artifact, main results, and conclusion.

### 3. Must fix: Table of contents contains structural anomalies

**Observed issues:**
- There is both **"9 References"** and a second **"References"** line in the contents.
- The appendix placeholders appear twice.
- Some subsection numbering suggests a hierarchy that is not consistently realized later.

**How to fix:** Clean the source headings and then regenerate the contents.

---

## Chapter 1 addendum

### 4. Must fix: Opening citation formatting is broken

**Observed issue:** The sentence introducing MCP contains an empty citation marker: **"late 2024[]"**.

**Why this matters:** This is a visible sign of incomplete editing on the first page of the thesis.

### 5. Must fix: Research-question numbering is inconsistent

**Observed issue:** The list shows **RQ1**, **RQ2**, then again **2. RQ3** instead of a clean third item.

**How to fix:** Correct the numbering and confirm that the same RQ ordering is used consistently across Chapters 1 and 7.

### 6. Must fix: Scope statements conflict with later chapters

**Observed issues:**
- Chapter 1 says the evaluation spans **four synthetic tool domains and one live API source**, while Chapter 2 describes four total sources.
- Chapter 1 says each condition is executed across **five independent trials**, while the results and summary material use **three repetitions**.

**Why this matters:** This is a direct cross-chapter consistency problem.

### 7. Should fix: Several sentences overstate the field generally

**Observed issue:** A few introduction sentences make broad claims about industry shifts, grammar-constrained decoding, and protocol adoption in a way that sounds stronger than the cited support.

**How to fix:** Either qualify the claims more carefully or support them more precisely.

---

## Chapter 2 addendum

### 8. Must fix: The chapter mixes precise evaluation design with imprecise methodological wording

**Observed issues:**
- It says the experiment is centered on a **single dependent variable**, but the thesis clearly measures several variables.
- The phrase **"design and evaluation methodology"** is still vague compared with a cleaner Design Science Research framing.

### 9. Should fix: The method text still contains missing or damaged phrasing

**Observed issues:**
- Some sentences appear to be missing words, such as the framework binding via a **native mechanism** with the actual interface name omitted.
- The section is understandable overall, but several sentences read like draft notes that need polishing.

### 10. Should fix: The benchmark logic needs one explicit paragraph on what counts as fairness

**Observed issue:** The chapter states that the three modes are compared fairly, but it would help to spell out more explicitly what is held constant across conditions and what is intentionally different.

---

## Chapter 3 addendum

### 11. Should fix: Heading hierarchy is inconsistent with other chapters

**Observed issue:** The chapter begins with **"## 3 Current State Analysis"** rather than a top-level chapter heading matching the rest of the thesis.

**How to fix:** Make the heading style consistent with other chapters.

### 12. Should fix: The chapter is currently too short for the role it plays

**Observed issue:** It identifies the protocol gap but does not yet convert that gap into explicit artifact requirements or evaluation implications.

---

## Chapter 4 addendum

### 13. Must fix: The theoretical motivation includes an incomplete mathematical statement

**Observed issue:** Section 4.2.2.1 introduces a cumulative input-token-cost idea, but the formula is missing or left unfinished.

### 14. Should fix: Some terms need more disciplined wording

**Observed issues:**
- **"industry standard"** for MCP should be stated more carefully unless defended strongly.
- The distinction between protocol facts, architectural interpretation, and performance speculation should remain clear.

### 15. Should fix: Related-work comparisons should be normalized

**Observed issue:** The chapter compares MCP-Prunex with RAG, prompt compression, and schema compression, but a compact comparison table would make the contribution boundaries much clearer.

---

## Chapter 5 addendum

### 16. Must fix: Several headings are malformed in the source manuscript

**Observed issues:**
- **5.2.1 Runtime Characteristics** is not marked as a proper heading.
- There is a stray empty **##** heading under Section 5.5.
- **5.5.1 LLM Provider Integration** also appears without consistent heading markup.

**Why this matters:** These are the kinds of source-markup problems that later create broken PDF structure and contents entries.

### 17. Should fix: The artifact description is stronger than the exact implementation boundary

**Observed issue:** The thesis sometimes sounds as if the proxy is a broader MCP middleware layer than it really is. In the submitted code, it intercepts `tools/list` and `tools/call`, not the full MCP operation surface.

**How to fix:** Keep the contribution strong, but define its exact interception boundary clearly.

### 18. Should fix: Some implementation-performance statements should point to Chapter 6 more directly

**Observed issue:** A few measured claims appear in Chapter 5 before the reader reaches the formal results chapter.

---

## Chapter 6 addendum

### 19. Must fix: Section heading markup is inconsistent in the results chapter

**Observed issues:**
- **6.1.1** appears without a proper Markdown heading marker.
- **6.5 Multi-Domain Multi-Model Validation** also appears without a proper chapter-subsection heading marker.

### 20. Must fix: The denominator note in the accuracy section is not fully compatible with the reported percentages

**Observed issue:** Some percentages imply fewer valid runs than the stated **n = 45** denominator.

**How to fix:** If timeout/error rows were excluded from some cells, say so directly in the table note.

### 21. Must fix: The latency interpretation is still vulnerable to misreading

**Observed issue:** The thesis moves between **proxy overhead**, **answer TTFT**, and **full pipeline latency**. These are different metrics and need sharper separation.

### 22. Should fix: Live-domain accuracy evidence is methodologically weaker than the other domains

**Observed issue:** Some live-domain checks are necessarily loose because the data are dynamic. This is understandable, but the thesis should acknowledge that these tests are weaker evidence for semantic correctness than the controlled synthetic domains.

---

## Chapter 7 addendum

### 23. Must fix: One future-work reference points to a non-existent section

**Observed issue:** Chapter 5 mentions future work in **section 7.5**, but Chapter 7 ends at **7.4**.

### 24. Must fix: The conclusion overstates what was shown for scalability

**Observed issue:** The thesis demonstrates repeated evaluation across multiple domains and models, but that is not the same as a throughput or concurrency study.

### 25. Should fix: Some recommendations should remain explicitly scoped to the evaluated data shapes

**Observed issue:** The deployment advice is useful, but it should be framed as guidance for similar structured-data MCP workloads rather than broad universal practice.

---

## References and appendices addendum

### 26. Should fix: Reference formatting needs one final normalization pass

**Observed issue:** The list is strong in coverage, but some entries still vary in punctuation and style.

### 27. Must fix: Appendices are currently placeholders only

**Observed issue:** The appendix file still contains **"Title of the Appendix"** and **"Content of the appendix is placed here"**.

**How to fix:** Replace the placeholders with real supporting material or remove the appendix section entirely.

---

## Closing judgment

The earlier chapter feedback was directionally correct, but not exhaustive. The manuscript is technically promising and much stronger than an early prototype draft, yet it still requires one disciplined final pass for consistency, structure, and presentation. The most important remaining work is now editorial and methodological tightening rather than a change of research direction.

# Source Code Review

## Overall assessment

The new source submission is a substantial improvement over the older version reviewed in `feedback/old/src-feedback.md`.

The most important earlier weaknesses have been addressed:

1. There is now a real proxy implementation in `mcp_prunex/proxy.py` rather than only an in-server field filter.
2. The dataset is no longer a tiny toy example; the retail generator creates 150 products and the SQL path uses a populated SQLite database.
3. There is now more than one domain and more than one server type: retail JSON, SQL, synthetic GitHub, and live GitHub API.
4. The evaluation harness is broader and records bytes, tokens, TTFT, pipeline time, and cost.
5. The packaging is better: `pyproject.toml` now includes optional evaluation dependencies and `sys.executable` is used in the evaluation harness.

So the source code now aligns with the thesis topic much better than before.

However, the code still does not fully support every claim made in the thesis, and the automated testing story is still narrow.

---

## A. Does the new source code meet the expectation of the submitted thesis?

## Short answer

**Partly yes, but not completely.**

The code now supports the core thesis claim that MCP-Prunex is a protocol-layer proxy that can prune JSON tool responses across multiple domains. That is a major improvement.

The remaining gap is not the existence of the artifact. The remaining gap is that some thesis claims are still broader than what the code and evaluation actually demonstrate.

---

## B. What has been implemented well

### 1. Real proxy architecture now exists

The old review said the project was only a field filter. That is no longer fair for this version.

`mcp_prunex/proxy.py` now:
- launches a downstream MCP server as a subprocess,
- connects to it as an MCP client,
- exposes itself upstream as an MCP server,
- intercepts `tools/list`,
- injects `field_keys_required_in_response`,
- intercepts `tools/call`,
- strips or preserves the parameter depending on downstream support,
- applies JSON pruning to returned tool content.

That is enough to justify calling the artifact a proxy middleware for the supported operation surface.

### 2. The evaluation scope is much closer to the thesis

The source tree contains:
- a retail JSON server,
- a SQL-backed server,
- a synthetic GitHub issues server,
- a live GitHub API server,
- a multi-domain evaluation harness,
- stored evaluation reports.

This is materially more credible than the old two-product prototype.

### 3. The filter itself has a reasonable correctness core

`mcp_prunex/prunex.py` correctly separates:
- simple keys,
- dot-path keys,
- path prefixes.

It also drops unmatched scalar values, which is the main no-leakage invariant. The existing unit tests are focused on exactly that behavior, which is the right place to start.

### 4. Some old reproducibility issues were fixed

Compared with the old version:
- dependency declaration is better,
- the dataset is generated deterministically,
- the SQL database is initialized from the same dataset,
- the evaluation harness is no longer Unix-only in its interpreter path selection.

---

## C. Where the code still falls short of the thesis claims

### 1. Must fix in the thesis or code: the proxy is not a full transparent MCP proxy

**What exists:** The proxy only intercepts `tools/list` and `tools/call`.

**Why this matters:** Some thesis wording suggests a broader JSON-RPC or MCP interception layer than what is actually implemented. In practice, this artifact is a **tool-surface proxy**, not a full MCP operation proxy for resources, prompts, or all protocol traffic.

**Recommendation:** Either:
- narrow the thesis wording to match the implemented boundary, or
- extend the code if full-protocol proxying is a real claim the student wants to defend.

### 2. Must fix in the thesis interpretation: latency evidence is easy to overstate

`eval_multi_domain.py` records:
- `field_selection_ttft_ms`,
- `answer_ttft_ms`,
- `total_pipeline_ms`.

But the summary/report logic mainly compares `answer_ttft_ms`, not the full pruned pipeline cost. That means the thesis can accidentally present pruning as faster while excluding the field-selection LLM call from the headline comparison.

**Why this matters:** In pruned mode there are two LLM calls. In raw mode there is only one. Comparing only answer TTFT does not represent the whole user-visible latency.

**Recommendation:** The thesis should clearly distinguish:
- proxy overhead,
- answer-generation TTFT,
- full end-to-end pipeline latency.

### 3. Must fix in the thesis interpretation: scalability is not really tested

The code does not contain a real throughput or concurrency benchmark for the proxy.

It demonstrates:
- repeated runs,
- multiple domains,
- multiple models,
- stdio-based proxying under evaluation.

It does **not** demonstrate:
- requests per second,
- concurrent clients,
- backpressure behavior,
- queueing effects,
- resource saturation.

**Recommendation:** RQ3 and the conclusion should be narrowed unless the student adds a real load experiment.

### 4. Should fix: evaluation defaults still conflict with the written thesis

In `eval_multi_domain.py`:
- the module-level default is `ACTIVE_MODEL = MODELS["phi4-mini"]`,
- the CLI default is `--reps 5`,
- the thesis and summary material repeatedly describe the core setup as 3 repetitions and focus on different final models.

**Why this matters:** The code can still produce runs that do not match the reported thesis configuration unless the operator overrides the defaults correctly.

**Recommendation:** Freeze the defaults to the thesis-final configuration or clearly document that `phi4-mini` and `--reps 5` are exploratory settings, not thesis-final settings.

### 5. Should fix: warmup logic does not warm the full proxy path

`run_warmup()` warms:
- one raw answer call,
- one field-selection call.

It does **not** warm a full proxy/pruned cycle with MCP call plus answer generation, and it does not warm proxy/unpruned as a full path.

**Why this matters:** First-run effects may still leak into measured results.

### 6. Should fix: live-domain accuracy evidence is relatively weak

The live GitHub domain uses loose expected fragments such as `count`, `login`, `label`, `pull`, or `title` for some cases.

**Why this matters:** This is understandable for non-deterministic live data, but those checks are much weaker than the deterministic synthetic-domain checks. They are closer to a smoke test than a strong semantic evaluation.

**Recommendation:** The thesis should acknowledge that the live-domain accuracy evidence is exploratory and structurally weaker.

### 7. Should fix: field-selection quality is itself a major source of variance

The evaluation depends heavily on the first LLM call choosing the right fields. The stored results show multiple cases where the model selected too few fields, leading to an avoidable failure. That is not a bug in the pruner alone; it is a system-level weakness.

**Recommendation:** The student should discuss that the artifact's success depends on field-selection reliability, prompt design, and possibly schema hints or dependency guidance.

---

## D. Code-level findings

### 1. `mcp_prunex/server.py`: private library mutation still exists

```python
mcp._tool_manager._tools["get_store_data"].output_schema = output_schema
```

This is still a fragile dependency on internal library state.

**Risk:** A future `mcp` package change may silently break schema attachment.

**Assessment:** Acceptable as prototype technical debt if it is acknowledged, but too fragile to present as a robust production pattern.

### 2. `mcp_prunex/server_sql.py`: SQL safety check is intentionally naive

The code allows only statements beginning with `SELECT`, which is fine for a local evaluation server but not a robust query-safety policy.

**Risk:** It is not a production-quality SQL safety layer.

**Assessment:** Acceptable for a thesis fixture if clearly framed as an evaluation-only server.

### 3. `mcp_prunex/proxy.py`: configuration is read at import time

`DOWNSTREAM_COMMAND`, `DOWNSTREAM_ARGS`, `DOWNSTREAM_CWD`, and `PRUNEX_ENABLED` are read once at module import.

**Risk:** This reduces flexibility for test isolation and in-process reconfiguration.

**Assessment:** Not a major problem for a CLI-run proxy, but worth noting if the student claims operational flexibility.

### 4. `mcp_prunex/proxy.py`: pruning only applies to JSON-parseable `TextContent`

This is a defensible design choice, but it means:
- mixed text and JSON payloads are not deeply understood,
- non-JSON text is passed through unchanged,
- other content types are not pruned.

**Assessment:** This should be described clearly as an implementation boundary.

### 5. `eval_multi_domain.py`: report headline can bias latency interpretation

The summary compares `answer_ttft_ms` rather than full pipeline cost in the compact comparison section.

**Risk:** Readers may think pruned mode is universally faster even when total pipeline time is higher.

### 6. `eval_multi_domain.py`: parallel mode execution may introduce contention

`asyncio.gather()` runs pruned, unpruned, and raw evaluations concurrently for each domain.

**Risk:** This is efficient, but it can blur causality for latency results on the same machine, especially with a local model backend.

**Assessment:** Reasonable engineering choice, but it should be acknowledged as a measurement trade-off.

---

## E. Are there any tests?

## Yes, but only at a narrow unit level

The submission contains one visible automated test file:

- `test_prunex.py`

This file covers unit tests for the pruning engine, including:
- flat simple-key selection,
- nested simple-key selection,
- exact dot-path matching,
- mixed simple-key and dot-path selection,
- no sibling leakage,
- list-item pruning,
- empty-key behavior,
- missing-field behavior,
- case-insensitive matching.

That is useful, but it is only one layer of the system.

## What is missing

There are no visible automated tests for:
- the proxy intercept logic,
- tool-schema injection,
- collision handling with downstream tools that already expose the same field parameter,
- error pass-through behavior,
- retail server validation behavior,
- SQL server behavior,
- live GitHub wrapper behavior,
- evaluation harness correctness,
- CSV/JSON result writing,
- summary aggregation.

---

## F. What kind of testing should be here?

## 1. Unit tests

These should remain, and more should be added for isolated logic.

Recommended unit-test targets:
- key parsing in `prunex.py`,
- malformed field-key validation in `server.py`,
- cost computation in `eval_multi_domain.py`,
- `check_expected()` behavior,
- dataset generation invariants in `generate_dataset.py`,
- report aggregation helpers.

## 2. Integration tests

These are the most important missing category.

Recommended integration tests:
- proxy + retail server: verify `tools/list` schema injection,
- proxy + retail server: verify `tools/call` pruning output,
- proxy disabled: verify full passthrough,
- downstream tool already supporting the field parameter: verify collision handling,
- malformed JSON text content: verify passthrough without crash,
- downstream error result: verify no masking of errors,
- SQL server query flow through proxy,
- synthetic GitHub server flow through proxy.

## 3. End-to-end or acceptance tests

These should exist in a small deterministic form.

Recommended acceptance coverage:
- one full pruned pipeline run with a stubbed or fake LLM selector,
- one full raw baseline run,
- one check that produced summary metrics match the stored row data,
- one regression test proving no unrequested fields leak in a full proxy round-trip.

## 4. Non-functional tests

Because the thesis claims efficiency and overhead properties, the project would also benefit from:
- benchmark tests for isolated proxy overhead,
- load tests for concurrency or throughput if scalability remains an RQ,
- robustness tests for malformed field paths,
- fixture-based tests for large nested JSON payloads.

## 5. External API strategy

Live GitHub calls should **not** be the basis of routine automated CI tests. They should be:
- stubbed,
- recorded with fixtures,
- or clearly marked as optional manual or nightly tests.

That keeps the test suite reproducible.

---

## G. Improvements that should still be there

### High-priority improvements

1. Add integration tests for the proxy's `tools/list` and `tools/call` behavior.
2. Make thesis-facing evaluation defaults match the final reported setup.
3. Separate and report proxy overhead, answer TTFT, and full pipeline latency more clearly.
4. Narrow or properly test the scalability claim.
5. Replace or encapsulate the private `_tool_manager` mutation if the library later exposes a public API.

### Medium-priority improvements

1. Improve warmup so the full measured pipeline is warmed consistently.
2. Strengthen live-domain evaluation methodology or explicitly downgrade it to exploratory evidence.
3. Add deterministic fake-LLM tests so the evaluation harness can be sanity-checked without real model calls.
4. Add regression tests for JSON text that is not strictly a single object or array.

### Nice-to-have improvements

1. Add schema-dependency hints so the field selector is less likely to omit semantically necessary fields.
2. Add a separate benchmark script for throughput and concurrent sessions if scalability remains central.
3. Add fixture examples in the appendix or README showing unpruned vs pruned payloads for one representative case.

---

## H. Final judgment

The old source-code feedback is no longer the right overall picture. This new version is a real step forward and is much closer to the thesis that the student is now submitting.

The main remaining concern is not that the project lacks an artifact. The main concern is that some thesis claims still need to be tightened to match exactly what the code and evaluation genuinely prove.

If the student presents this as:
- a stdio-based MCP tool-surface proxy,
- evaluated across multiple structured-response domains,
- with strong evidence for payload reduction,
- useful but conditional evidence for answer-quality improvement,
- and limited evidence for broad scalability,

then the code and thesis align reasonably well.

# Thesis Feedback Overview

## Overall assessment

This is a meaningful improvement over an early-stage draft. The thesis now has a clear technical focus, a recognizable artifact, and a broader evaluation design than one often sees in a master's engineering thesis.

The strongest parts are the clarity of the core idea, the multi-model and multi-domain evaluation effort, and the honest recognition that pruning is not universally beneficial. The main remaining work is not to reinvent the thesis, but to tighten the manuscript so that the wording, structure, and claims match the evidence exactly.

For a more exhaustive follow-up, see the detailed manuscript addendum and source-code review added separately in this folder.

## Highest-priority revisions

1. Shorten and clean the abstract so it fits the university format and removes malformed citation/template issues.
2. Make the research questions, method, and conclusions fully consistent, especially around repetition counts, number of evaluation domains, and the meaning of the latency and scalability claims.
3. Fix lone subsection numbering, broken headings, table numbering, and other presentation issues across the manuscript.
4. Revise a few strong claims so they match the actual scope of the evidence, particularly the 50 ms latency wording and the high-throughput scalability conclusion.
5. Replace the empty appendix placeholders with real supporting material or remove them.

## Encouraging note to the student

The thesis has a good core contribution. The technical idea is relevant, the implementation appears thoughtful, and the evaluation effort shows commitment. With a careful final revision pass focused on consistency, presentation quality, and claim discipline, this can become a solid master's thesis submission.

# Opponent Review of the Phase 2 Feedback (Manikanta)

> Role: acting as an opponent/second reviewer of the supervisor's own feedback set
> (`0_front_matter_feedback.md` … `11_src_feedback.md`, `Feedback.md`), not of the
> student's thesis directly. Every claim below was re-checked against the actual
> submission (`submission/*.md`, `submission/src/**`) and the pre-existing
> `feedback/old/*` notes before being accepted or challenged.

## Verdict on the feedback set

The existing feedback is thorough, well-organized, and — on verification — **factually accurate on almost every specific claim it makes**: the "Tite" typo, the empty `[]` citation, the malformed URL-in-text citation, the RQ numbering ("2. RQ3"), the five-trials-vs-three-repetitions inconsistency, the missing `7.5` cross-reference, the unmarked headings in Chapter 5, the "single dependent variable" contradiction in Chapter 2, and the incomplete formula in 4.2.2.1 were all independently confirmed in the source files. This is a well-grounded review, not a superficial one.

The opponent role below therefore focuses on: (1) two places where the feedback under-states how serious a finding actually is, (2) one place where the feedback over-states a finding by not finishing its own audit, (3) a few concrete issues the "exhaustive" addendum still missed, and (4) a framing problem the feedback does not acknowledge: some of the ambiguity being criticized in the student's text originated in the supervisor's own earlier guidance.

---

## 1. The RQ1 / 50 ms critique is correct but too gentle — it should name a direct violation, not just an "ambiguity"

**What the feedback says:** Chapter 6 feedback item 2 and Chapter 7 feedback item 1 both say the thesis "may be mixing proxy overhead with end-to-end TTFT" and ask the student to clarify what the 50 ms budget refers to.

**What a full check of Table 9 shows:** For the local model (LLaMA 3.1 8B), pruned TTFT is *higher* than raw TTFT by a wide margin in two of four domains:

| Domain | Raw TTFT | Pruned TTFT | Change |
|---|---|---|---|
| Retail JSON | 9,424 ms | 19,673 ms | **+10,249 ms** |
| GitHub Synthetic | 6,163 ms | 10,465 ms | **+4,302 ms** |

That is not an ambiguous labeling problem — under any reasonable reading of "50 ms budget," a >10-second regression is a direct, quantified violation of RQ1's own stated success criterion. Chapter 7's conclusion ("achieving token reduction well within 50 ms latency budget") is only true because it silently scopes itself to "for commercial models" in the same sentence, quietly excluding the local-model result that sits in the very table it cites. That is closer to **selective reporting** than to a wording ambiguity.

**Recommendation:** Upgrade this from "clarify what the budget means" to a named "must fix": require the student to state explicitly, with the numbers above, that RQ1's latency criterion is met for the commercial models but is violated for the local, cost-free model — and to discuss why (field-selection call cost on a slow local backend), since that is arguably the more practically relevant deployment case for a token-cost-motivated thesis.

---

## 2. The n = 45 denominator "must fix" is real, but the feedback stopped its own audit halfway — and the actual picture is different from what it implies

**What the feedback says (Ch6 item 1, addendum item 20):** "Some percentages such as 90.9% and 68.2% do not fit a denominator of 45," framed as an unresolved "must fix" quantitative-rigor problem.

**What a full per-cell audit of Table 7 shows:** Every value that is achievable with n = 45 is exactly the value printed (e.g. 20.0 = 9/45, 66.7 = 30/45, 51.1 = 23/45, all four gpt-4o-mini rows fit exactly). Checking every Claude-3-Haiku cell against n = 44 instead of n = 45 finds not two but **five** exact matches:

| Cell | Printed | n=44 exact value |
|---|---|---|
| Raw, Retail JSON | 90.9 | 40/44 = 90.91 |
| Unpruned, GitHub Synthetic | 65.9 | 29/44 = 65.91 |
| Pruned, SQL | 68.2 | 30/44 = 68.18 |
| Pruned, GitHub Synthetic | 72.7 | 32/44 = 72.73 |
| Pruned, GitHub Live | 61.4 | 27/44 = 61.36 |

That is exactly **5 anomalous cells**, and Section 6.5 of the same chapter already discloses "0.3% error rate (5 errors from 1,620 measurements, all in claude-3-haiku model due to occasional API timeouts)." In other words, once the audit is completed, **Table 7 is internally consistent with the thesis's own explanation** — the numbers are not wrong, they are just under-annotated where they appear.

**However**, the audit surfaces one genuine unexplained anomaly the current feedback did not catch: **LLaMA 3.1 8B, Raw mode, SQL domain = 22.7%**, which fits neither n = 45 (nearest exact value is 22.2 = 10/45) nor n = 44 (22.7 would need k≈10.0, i.e. 10/44 = 22.73, which is close but the model's disclosed errors are stated to be entirely in claude-3-haiku, not llama). This one cell is either a transcription/rounding slip or an undisclosed sixth irregularity that the "5 errors, all claude-3-haiku" statement does not cover.

**Recommendation:** Downgrade the severity framing from "the denominator must be exact, this is a quantitative thesis" (which reads as if the numbers are wrong) to: "add a footnote to Table 7 pointing to the Section 6.5 error-rate disclosure for the five affected Claude cells, and separately verify the LLaMA raw-SQL 22.7% cell, which remains unexplained by that disclosure." This is more precise, more actionable, and more fair to the student than the current wording.

---

## 3. The "exhaustive" addendum missed a global figure/table numbering gap

The addendum explicitly audits heading and table-numbering problems chapter by chapter, but a full sweep of every `Table N:` and `Figure N:` caption across the manuscript shows:

- Tables are numbered 1 (Ch4) → 2, 3, 4 (Ch5) → 5–11 (Ch6): globally sequential, no gaps, **except** that Chapter 6's body text refers to the byte-reduction table as "Table 4" while its caption reads "Table 5" (already caught by the existing feedback).
- Figures are numbered 1 (Ch4) → 3, 4, 5, 6, 7, 8 (Ch5): **Figure 2 does not exist anywhere in the submitted manuscript.**

This is a concrete, easily verifiable presentation defect that a numbering-focused addendum should have caught. Recommend adding it as a "must fix" alongside the existing Table 4/5 item: do one global figure- and table-numbering pass, not just a per-chapter one.

Also worth flagging while auditing the front matter: the abstract page still contains the literal unfilled template placeholder **"Number of Pages: xx pages + x appendices"**, and the "Professional Major" field reads **"Networking and Services or Medical Technology"** — i.e., both template options were left in rather than one being selected. Neither of these was named explicitly in any of the eleven feedback files, even though the front-matter feedback and addendum both did a close pass over this exact page.

---

## 4. Framing issue: the feedback does not acknowledge that the "50 ms" ambiguity has a shared origin

`feedback/old/contributions-RQs.md` — the supervisor's own earlier draft of the research questions — phrased RQ1 as: *"...beyond an acceptable, system defined latency budget **(e.g. 50 ms)**."* The "e.g." marks 50 ms explicitly as an illustrative placeholder, not a committed number. The student's Chapter 1 hardened this into *"...latency budget of 50ms"* with no qualifier, and the current feedback critiques the resulting rigidity purely as a student drafting problem (Ch1 items, Ch6 item 2, Ch7 item 1) without noting that the ambiguity was inherited from supervisor-authored guidance.

**Recommendation:** When the final feedback is sent to the student, it would be more balanced — and more useful going forward — to note this shared origin explicitly, e.g.: *"The 50 ms figure in the original RQ template was an illustrative example; treating it as a fixed pass/fail bound was a reasonable but incorrect reading on your part — let's fix the wording together rather than treating this purely as your error."* This also suggests a process note for future proposal-stage RQ templates: mark illustrative numbers explicitly as non-binding (e.g., "an illustrative budget such as ~50 ms, to be refined empirically") to prevent recurrence with future students.

---

## 5. Minor point on the source-code review

`11_src_feedback.md` recommends (should-fix item 4) that the student "clearly document that phi4-mini and --reps 5 are exploratory settings, not thesis-final settings." A check of `eval_multi_domain.py` shows the code already partially does this for the model default via an inline comment (`# NOTE: phi4-mini and gemma-4-26b are defined but not used in final eval runs.`), but no equivalent comment exists for the `--reps 5` CLI default. The recommendation is still valid but should be narrowed to the `--reps` default specifically, since the model-default half of the concern is already addressed in the code.

---

## Summary of changes recommended to the feedback set

1. **Strengthen** the RQ1/latency item: cite the LLaMA +10,249 ms / +4,302 ms regressions by name; reframe Chapter 7's claim as selectively scoped rather than merely ambiguous.
2. **Soften and correct** the n = 45 item: most of the apparent inconsistency is already self-explained by Section 6.5 once fully audited (5/5 claude-only anomalies match exactly); redirect the "must fix" to (a) a cross-reference footnote and (b) the one genuinely unexplained cell (LLaMA raw SQL = 22.7%).
3. **Add** a global figure/table numbering check to the addendum: Figure 2 is missing entirely from the thesis.
4. **Add** the two untouched front-matter placeholders: "xx pages + x appendices" and the unselected "Networking and Services or Medical Technology" field.
5. **Acknowledge**, when writing to the student, that the 50 ms figure originated as an illustrative example in the supervisor's own earlier RQ draft, not as a student error in isolation.
6. **Narrow** the src-feedback recommendation about undocumented exploratory defaults to the `--reps` CLI default only.

None of the above changes the overall verdict (REVISE). They make the feedback more precise, more defensible if the student pushes back on any single point, and fairer about which parts of the ambiguity are shared with the supervisor's own earlier guidance.


# Chapter 1: Introduction

## Overall assessment

This chapter presents a relevant and timely technical problem. The focus is much clearer now: the thesis is about protocol-aware response pruning in MCP workflows, not about a broad generic AI platform.

The chapter is already stronger than many first-draft introductions, but it still needs revision to make the research framing fully precise, consistent, and master-level in tone.

---

## Issues to address

### 1. Must fix: Repair citation and wording problems in the opening pages

**What is wrong:** The first page still contains several language and formatting issues, including an empty citation marker after the MCP introduction, multiple grammar problems, and some unclear sentences.

**Why this matters:** Chapter 1 sets the academic tone for the whole thesis. If the first page looks unfinished, the reader's confidence drops before reaching the technical substance.

**How to fix:** Carefully proofread the opening two pages for grammar, punctuation, article use, capitalization, and citation formatting. This chapter would benefit from a line-by-line language pass.

### 2. Must fix: Align the research questions with what the thesis actually measures

**What is wrong:** RQ3 asks about **high-throughput environments**, but the current thesis does not appear to contain a real throughput experiment such as concurrent load testing, queueing behavior, or requests-per-second benchmarking.

**Why this matters:** A research question should only promise what the method and results actually test.

**How to fix:** Either narrow RQ3 to the measured overhead of JSON-RPC interception in the evaluated setup, or add a real throughput-oriented experiment. The simpler and safer revision is to narrow the wording.

### 3. Must fix: Remove inconsistencies in scope statements

**What is wrong:** The chapter says the evaluation spans **four synthetic tool domains and one live API source**, while the later method chapter presents **four total data sources**, one of which is the live GitHub case. The scope section also says each configuration was executed across **five independent trials**, while Chapter 6 reports **three repetitions**.

**Why this matters:** Scope inconsistency creates avoidable doubt about the reliability of the results.

**How to fix:** Use one exact description of the datasets and one exact repetition count across Chapters 1, 2, and 6.

### 4. Should fix: Make the research gap more explicit in literature terms

**What is wrong:** The chapter explains the practical problem well, but the introduction would be stronger if it stated more directly what prior work already covers and what this thesis adds beyond it.

**How to fix:** Add one short paragraph that positions MCP-Prunex against tool-calling evaluation, schema compression, and other context-reduction approaches. This does not need to be long, but it should clearly identify the gap.

### 5. Should fix: Strengthen the contribution wording

**What is wrong:** The contribution list is promising, but some phrases are still too broad, especially around scalability and architectural insight.

**How to fix:** Keep the three-part contribution structure, but state each contribution in terms that exactly match the later evidence: artifact, controlled evaluation, and practical boundary-condition guidance.

# Chapter 2: Method and Material

## Overall assessment

This chapter shows real progress. The thesis now has a recognizable evaluation design, multiple datasets, multiple models, and a reproducibility-oriented setup. That is a solid improvement.

The remaining work is to make the method description fully consistent and to ensure that each research question is supported by an appropriate measurement method.

---

## Issues to address

### 1. Must fix: Correct the research-design description

**What is wrong:** The chapter says the evaluation is centered on **a single dependent variable**, but it later measures multiple outcomes: context bytes, tokens, latency, cost, and task accuracy.

**Why this matters:** This is a methodological contradiction.

**How to fix:** Revise the wording so that the design is described as a controlled comparative evaluation with multiple dependent variables.

### 2. Must fix: Resolve the repetition-count inconsistency

**What is wrong:** Earlier sections mention five trials, while the results chapter reports three repetitions per condition.

**How to fix:** State one final repetition protocol everywhere. If three repetitions are the real study design and five were only part of pilot validation, explain that clearly and consistently.

### 3. Must fix: Match the method to RQ3

**What is wrong:** The method chapter does not describe a real high-throughput or concurrency experiment, even though RQ3 is framed around high-throughput environments.

**Why this matters:** At the moment, the method supports claims about per-request overhead better than claims about throughput scalability.

**How to fix:** Either add a throughput experiment or narrow the wording of the research question and conclusion so they match the available evidence.

### 4. Should fix: Explain the accuracy metric more carefully

**What is wrong:** The substring-based pass/fail method is useful for automation, but it is still a simplified proxy for semantic correctness.

**How to fix:** Keep the method, but explicitly state its limits. For example, mention that it rewards full factual recall of expected fragments, but may mark partially correct paraphrases as failures and cannot judge deeper semantic adequacy.

### 5. Should fix: Repair structure numbering where only one subsection exists

**What is wrong:** Section **2.1.1** appears without a visible **2.1.2**, and section **2.3.1** appears without a visible **2.3.2**.

**Why this matters:** A lone subsection number makes the structure look unfinished.

**How to fix:** Either remove the extra numbering level or add parallel subsections only if they are genuinely needed.

### 6. Should fix: Improve language precision in the method description

**What is wrong:** Some sentences are difficult to follow because of grammar problems, missing words, or compressed phrasing.

**How to fix:** Please do a language-editing pass with special care in the method chapter. This chapter needs very precise wording because it defines how the evidence was produced.

# Chapter 3: Current State Analysis

## Overall assessment

This chapter identifies the central protocol problem clearly and stays focused on the thesis topic. That focus is good.

At the same time, the chapter is currently very short. It needs a little more analytical depth so it feels like a real chapter rather than a short transition note.

---

## Issues to address

### 1. Must fix: Expand the analysis from problem statement to design requirements

**What is wrong:** The chapter explains that MCP returns full payloads and lacks field-level selection, but it does not yet translate that limitation into clear design requirements for the artifact.

**Why this matters:** The reader should understand not only what is wrong, but also what the middleware must preserve while fixing it.

**How to fix:** Add a short requirement-oriented subsection or table. For example: preserve JSON structure, avoid unknown-parameter breakage, keep error payloads intact, remain server-agnostic, and reduce unnecessary response fields.

### 2. Should fix: Add one concrete illustrative example

**What is wrong:** The chapter stays quite abstract.

**How to fix:** Add one simple example of an over-fetched tool response and show why only a few fields were actually needed for the task. That would make the motivation much easier to grasp.

### 3. Should fix: Clarify why MCP cannot simply reuse GraphQL semantics directly

**What is wrong:** The GraphQL comparison is useful, but it currently passes very quickly.

**How to fix:** Add two or three sentences explaining the difference between a developer-written query language and an LLM-mediated tool-calling protocol. That would strengthen the conceptual bridge to the middleware design.

### 4. Should fix: Polish the chapter language

**What is wrong:** There are small language issues such as subject-verb agreement and phrasing like **"Wasted bytes becomes wasted tokens"**.

**How to fix:** A short proofreading pass will make the chapter read much more professionally.

# Chapter 4: Theoretical Background

## Overall assessment

This chapter shows that you understand the core technical ideas behind the thesis. The selection of topics is sensible: context windows, MCP architecture, and related compression work all belong here.

The main remaining task is to make the chapter more academically polished and more precise in its formal explanations.

---

## Issues to address

### 1. Must fix: Repair incomplete technical exposition in Section 4.2.2.1

**What is wrong:** The token-cost discussion introduces a cumulative-cost argument, but the mathematical explanation is incomplete. The sentence ends as if a formula should appear, but no usable expression is actually presented.

**Why this matters:** This is one of the core theoretical motivations for the whole thesis.

**How to fix:** Either provide the full formula clearly and explain the variables, or rewrite the paragraph in plain language without the broken formula setup.

### 2. Must fix: Improve formal accuracy and wording throughout the chapter

**What is wrong:** Several sentences currently weaken the academic quality because of grammar, missing articles, awkward phrasing, or imprecise wording. Examples include the explanation of context windows, the self-attention sentence, and parts of the MCP lifecycle description.

**How to fix:** Please revise this chapter carefully for grammar and clarity. The ideas are valuable, but the wording needs polishing so the technical argument is easier to trust.

### 3. Should fix: Make the related-work comparison more explicit

**What is wrong:** The chapter mentions prompt compression, schema compression, and RAG, but the distinctions could be stated even more sharply.

**How to fix:** Add a short comparison paragraph or table that contrasts these approaches by optimization target, stage in the pipeline, and whether they operate on structured or unstructured data.

### 4. Should fix: Check whether every strong claim has the right type of support

**What is wrong:** Some claims are supported by official documentation, while others rely on research sources. That is fine, but the role of each source should remain clear.

**How to fix:** Re-check that protocol facts cite protocol documents, research claims cite research papers, and broad industry statements are not left sounding stronger than the evidence.

# Chapter 5: Middleware Implementation

## Overall assessment

This is one of the strongest chapters in the thesis. The reader can understand the architecture, the proxy role, the interception points, and the filtering idea. The implementation now clearly looks like a middleware artifact rather than only a small code experiment.

The main revisions needed here are structural clarity, formatting cleanup, and a few places where the explanation should be tightened.

---

## Issues to address

### 1. Must fix: Correct heading and formatting problems inside the chapter

**What is wrong:** There are visible structural issues such as **5.2.1** appearing without a proper Markdown heading marker, a stray empty **##** heading under Section 5.5, and **5.5.1** appearing without consistent heading formatting.

**Why this matters:** These issues make the chapter look less complete than it actually is.

**How to fix:** Review the full heading hierarchy and make sure every subsection is formatted consistently.

### 2. Must fix: Clarify what is and is not proxied

**What is wrong:** The chapter says the proxy only handles `tools/list` and `tools/call`, but then discusses other protocol operations in a way that may confuse the reader about whether they pass through the same process or not.

**How to fix:** Add one precise clarifying sentence or diagram note stating exactly which MCP operations are intercepted, which are ignored, and what architectural assumptions that implies.

### 3. Should fix: Make the runtime-characteristics section more evidence-linked

**What is wrong:** The runtime-characteristics subsection gives useful statements, but some of them would be stronger if explicitly linked back to the evaluation method or results chapter.

**How to fix:** Keep the explanatory text here, but make sure measured claims clearly reference the later empirical results rather than sounding purely descriptive.

### 4. Should fix: Tighten several implementation explanations for clarity

**What is wrong:** The implementation logic is understandable, but a few paragraphs are longer than necessary and contain grammar issues that make the flow harder to follow.

**How to fix:** Consider simplifying the longer paragraphs in 5.1.3, 5.3, and 5.4 into shorter sentences. The technical content is good; it just needs clearer delivery.

### 5. Good to consider: Add one compact end-to-end example

**What is wrong:** The chapter would become even easier to follow if the reader saw one short sample tool schema, tool call, and pruned response.

**How to fix:** A compact example in the appendix or inside Section 5 would make the artifact more concrete without adding much length.

# Chapter 6: Results and Analysis

## Overall assessment

This chapter is a real strength of the thesis. The evaluation is broader than many engineering theses, and the multi-model, multi-domain comparison gives the work genuine value.

The main task now is to make sure that the numerical presentation is fully consistent and that the conclusions drawn from the tables do not overreach what was actually measured.

---

## Issues to address

### 1. Must fix: Correct inconsistent denominator statements in the accuracy section

**What is wrong:** The chapter says that each accuracy cell reports **n = 45** measurements, but some percentages such as **90.9%** and **68.2%** do not fit a denominator of 45. They suggest that some cells may have fewer valid runs, possibly because of API timeouts.

**Why this matters:** This is a quantitative thesis. The denominator must be exact.

**How to fix:** State the true denominator policy clearly. If some cells have fewer valid runs because of errors, report that transparently in the table note.

### 2. Must fix: Re-check the latency interpretation against the 50 ms claim

**What is wrong:** The chapter reports TTFT values in the hundreds or thousands of milliseconds, while earlier sections frame the acceptable latency budget as **50 ms**. This creates an interpretation problem.

**Why this matters:** The thesis may be mixing **proxy overhead** with **end-to-end TTFT**. These are not the same measurement.

**How to fix:** Clarify exactly what the 50 ms budget refers to. If it refers only to middleware overhead, say so consistently and avoid wording that suggests full end-to-end TTFT stayed under 50 ms.

### 3. Must fix: Repair table cross-references and labels

**What is wrong:** The chapter refers to byte reduction as shown in **Table 4**, but the actual byte-reduction table is labeled **Table 5**. There are also a few layout inconsistencies around section headings such as **6.1.1** and **6.5**.

**How to fix:** Do a full numbering and caption pass for all tables and subsection headings in this chapter.

### 4. Should fix: Keep cost projections clearly separated from measured costs

**What is wrong:** The chapter already distinguishes measured evaluation costs from projected annual savings, which is good, but the wording should remain very explicit so readers do not confuse the projection with observed production data.

**How to fix:** Label the production-scale estimates very clearly as scenario-based projections derived from the measured token reductions and stated pricing assumptions.

### 5. Should fix: Add one short note on practical significance versus absolute accuracy

**What is wrong:** The chapter reports accuracy deltas well, but some raw accuracy levels remain modest in several cells.

**How to fix:** Briefly comment on what level of answer quality would be acceptable in a real deployment context and why these results are still useful as comparative evidence even if some absolute scores are not yet high enough for production use.

# Chapter 7: Discussions and Conclusions

## Overall assessment

This chapter contains one of the best aspects of the thesis: it recognizes that pruning is beneficial in some situations and harmful in others. That conditional conclusion makes the work more credible.

The final revision should focus on tightening a few claims so that the conclusion stays fully aligned with the measured evidence.

---

## Issues to address

### 1. Must fix: Revise the answer to RQ1 so it does not overclaim the latency result

**What is wrong:** The chapter says the system achieved the reduction **well within the 50 ms latency budget**, but the results chapter reports TTFT values far above 50 ms. This conclusion is only defensible if the budget refers specifically to proxy overhead, not end-to-end response latency.

**How to fix:** Rewrite the conclusion so it distinguishes clearly between proxy overhead and total TTFT. This is an important correction.

### 2. Must fix: Soften the scalability conclusion unless a real throughput test is added

**What is wrong:** The chapter concludes that the scalability objective is met, but the current evidence mainly shows low per-request overhead across many benchmark runs, not high-throughput concurrency behavior.

**Why this matters:** Repeated measurements are not the same as throughput benchmarking.

**How to fix:** Either narrow the wording to **operational overhead in the evaluated setup** or add a real load/scalability experiment.

### 3. Should fix: Tighten the trade-off discussion with more careful wording

**What is wrong:** The trade-off section is interesting, but some sentences are grammatically rough and therefore less persuasive than the underlying point.

**How to fix:** Revise the three trade-off paragraphs into cleaner prose. The content is strong; the language just needs polishing.

### 4. Should fix: Keep deployment recommendations tied to the measured scope

**What is wrong:** The practical implications are useful, but some statements move close to production guidance across a wide set of systems.

**How to fix:** Frame the recommendations as evidence-informed guidance for structured-data MCP servers similar to the evaluated cases, rather than as a universal deployment rule.

# Chapter 8: Summary

## Overall assessment

The summary is clear and appropriately focused on the contribution of MCP-Prunex. It gives the reader a quick reminder of the artifact, the evaluation, and the overall outcome.

Only a small revision is needed here.

---

## Issues to address

### 1. Should fix: Keep the summary slightly more balanced

**What is wrong:** The summary emphasizes the positive outcomes strongly, but it would be even stronger if it also briefly reminded the reader of the main boundary condition.

**How to fix:** Add one sentence noting that pruning performed best on structured, independently addressable data and could reduce accuracy for large semantically interdependent API responses.

### 2. Should fix: Do one final language-cleaning pass

**What is wrong:** There are a few phrasing issues near the start of the chapter.

**How to fix:** Proofread the summary carefully so the final chapter ends the thesis with clean and confident language.

# References

## Overall assessment

The reference list is much better aligned with the actual thesis topic than in many earlier-stage drafts. It includes MCP documentation, context-compression work, long-context research, and methodology sources.

Only a few cleanup items remain.

---

## Issues to address

### 1. Must fix: Standardize the reference style fully

**What is wrong:** Some entries follow Vancouver-style conventions more closely than others. A few entries mix punctuation, capitalization, venue formatting, DOI formatting, and access-date style.

**Why this matters:** A master thesis should use one reference style consistently from start to finish.

**How to fix:** Do a full formatting pass across all entries and make them consistent with the university's required style.

### 2. Should fix: Re-check web and documentation citations for completeness

**What is wrong:** Official documentation sources are appropriate in this thesis, but they should all have stable titles, dates where available, cited dates, and uniform access formatting.

**How to fix:** Verify each documentation entry one more time, especially the MCP specification pages and vendor pricing pages.

### 3. Should fix: Ensure every unusual in-text citation format has a clean matching entry

**What is wrong:** The abstract currently contains one malformed citation style. Even if the reference itself exists, the in-text usage must also be corrected.

**How to fix:** After fixing the abstract, do one final citation pass through the whole thesis to ensure that all in-text citations match the reference list cleanly.

