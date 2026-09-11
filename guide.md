# Guide to the Front Matter, Abstract, and Reading Strategy

## Purpose of this part

The opening material tells the reader what problem the thesis tackles, what artifact was built, and how the rest of the document should be interpreted. In this thesis, the central problem is not a business chatbot or a generic AI assistant. It is a protocol-level efficiency problem in Model Context Protocol (MCP) systems: tool calls often return much more structured data than the model actually needs, and those unnecessary fields consume context-window space, money, and response time.

## What the abstract claims

The abstract presents MCP-Prunex as a transparent middleware proxy between MCP clients and servers. Its main idea is simple: when the language model only needs selected fields from a JSON response, the middleware prunes the rest before the payload is sent back into the model context. The abstract also claims that this was evaluated across three model families and four domains, and that pruning produced large byte and token reductions with model-dependent effects on accuracy.

## What to keep in mind while reading

The thesis combines two contribution types.

- An **engineering artifact**: the proxy middleware itself.
- An **empirical study**: the evaluation of what happens when that proxy is used under different models and data structures.

This distinction matters. A working proxy does not automatically prove a research contribution unless the evaluation is clear, reproducible, and interpreted carefully.

## How to read the rest of the thesis

- **Chapter 1** defines the research problem, the questions, the claimed contributions, and the scope.
- **Chapter 2** explains the research design, datasets, and measurement logic.
- **Chapter 3** describes the current MCP limitation that motivates the artifact.
- **Chapter 4** gives the theoretical background on context windows, MCP, and related compression methods.
- **Chapter 5** explains how MCP-Prunex is implemented and how the evaluation harness works.
- **Chapter 6** reports the measured reductions, accuracy changes, latency, and cost estimates.
- **Chapter 7** interprets the results and explains when pruning helps and when it becomes risky.
- **Chapter 8** summarizes the full thesis.
- **References** and **Appendices** should support reproducibility.

## Main reading lens

While reading, keep four questions in mind:

1. What exactly is being optimized: bytes, tokens, latency, cost, or answer quality?
2. What is the real baseline for comparison in each experiment?
3. Does the evaluation support the strong claims made in the abstract and conclusion?
4. Where does pruning help, and where does it remove information the model actually needed?

## Key takeaway

The front matter frames the thesis as a focused study on protocol-aware context optimization in MCP. The strongest reading strategy is to treat it as a design-science artifact plus an empirical evaluation of trade-offs, and then check whether each chapter supports that framing consistently.

# Chapter 1 Guide: Introduction

## What this chapter does

Chapter 1 introduces the thesis problem: MCP tool responses often contain redundant structured data, and this extra payload consumes context-window space that large language models could otherwise use for reasoning. The chapter positions the thesis as a response to this inefficiency by proposing MCP-Prunex, a proxy middleware that prunes responses at the protocol level.

## The research problem

The chapter argues that modern LLM systems increasingly depend on tool calling, and that MCP has become an important open protocol for that interaction. However, MCP itself does not provide native field-level response selection, so tools often return complete JSON payloads even when the task only depends on a small subset of fields. This is the core inefficiency that motivates the thesis.

## The research questions

The thesis asks three questions:

1. How much payload reduction can be achieved without unacceptable latency overhead?
2. How does pruning affect semantic accuracy?
3. What is the computational overhead and scalability boundary of this approach?

These questions are important because the thesis is not only asking whether pruning is possible. It is asking whether pruning is useful, safe, and general enough to matter.

## Scope and boundaries

The chapter limits the artifact to MCP over stdio and to field-level pruning of JSON-RPC payloads. It does not claim row-level query optimization, persistent memory, or broad transport support. When reading later chapters, check whether the implementation and evaluation remain faithful to these boundaries.

## What to watch carefully

Pay attention to how the chapter defines efficiency, integrity, and scalability. These terms sound clear, but they only become meaningful if the later method and results chapters measure them in a consistent way. Also watch for whether the scope statements match the actual experiment counts, repetitions, and data sources reported later.

## Key takeaway

Chapter 1 sets up a clear and relevant technical problem. Its success depends on whether the later chapters can prove the three claimed benefits with evidence that matches the exact research questions stated here.

# Chapter 10 Guide: Appendices

## Purpose of this part

The appendices should support transparency and reproducibility. In a thesis like this, appendices are a good place for detailed benchmark questions, scoring rules, larger tables, CLI examples, or selected schema and response examples that would interrupt the main chapters.

## How to interpret the current appendix section

If the appendices are empty or contain placeholders only, the reader loses a valuable support layer for the technical claims made in Chapters 2, 5, and 6. This is especially important in an engineering thesis where reproducibility and implementation clarity matter.

## Key takeaway

For this thesis, the appendices should strengthen the evidence trail. They should not remain as template placeholders.

# Chapter 2 Guide: Method and Material

## What this chapter does

Chapter 2 explains how the thesis was studied, not only what was built. It places the work within a design-oriented research approach and describes the experimental setup used to compare pruned and unpruned MCP workflows.

## Research design

The chapter frames the work as design and evaluation research informed by Design Science Research. In practical terms, this means the thesis builds an artifact first and then studies its behavior under controlled conditions. The artifact is MCP-Prunex, while the evaluation compares different execution modes and model families.

## Datasets and evaluation modes

The chapter introduces four data sources: a retail JSON dataset, a synthetic issues dataset, a SQLite-backed dataset, and a live GitHub API case. These were chosen to test the middleware on different response structures, not only one convenient example. The evaluation also distinguishes pruned, unpruned, and raw baseline modes, which is necessary to separate the effect of pruning from the effect of pipeline overhead.

## Accuracy logic

A key part of this chapter is the pass/fail scoring approach based on expected answer fragments. This is an efficient way to automate many runs, but it is also a simplification. When reading the results, keep in mind that substring matching is a narrow measure of semantic accuracy and does not capture all forms of partially correct answers.

## Reliability and validity

The chapter tries to strengthen reproducibility through fixed temperatures, repeated runs, deterministic dataset generation, and automated data collection. This is one of the more important parts of the thesis, because many of the later claims depend on whether the test setup is consistent and fair.

## Main reading lens

As you read this chapter, check whether the method really supports all three research questions equally well. The design appears strongest for payload, token, latency, and cost measurements. It is weaker if the thesis wants to make a strong claim about high-throughput scalability or broad semantic correctness beyond this evaluation rubric.

## Key takeaway

Chapter 2 is the bridge between the artifact and the evidence. If this chapter is precise, the results become convincing; if it is inconsistent, the strongest later claims become difficult to defend.

# Chapter 3 Guide: Current State Analysis

## What this chapter does

Chapter 3 explains the immediate technical problem in current MCP practice. It is shorter than a business case study chapter because the thesis problem is protocol-centered rather than organization-centered.

## The current limitation

The chapter states that MCP tool calls return complete results with no native field-selection mechanism. This means the caller either receives everything or nothing, even when only a few fields matter for the question. The chapter uses this limitation to justify why a middleware layer could be useful.

## Why GraphQL is mentioned

GraphQL is used here as a contrast case. In GraphQL, clients can request only the fields they need. The chapter argues that MCP cannot simply copy this approach directly because the caller is an LLM interacting through an existing tool-calling protocol rather than a human developer writing a structured query.

## How to read this chapter

This chapter should be read less as a survey of all MCP history and more as a problem-definition chapter. The key question is whether it describes the current limitation precisely enough to motivate the artifact in Chapter 5.

## What to watch for

Because the chapter is brief, the reader should look for whether it turns the problem into clear requirements. In other words: what exactly should the middleware preserve, what should it reduce, and what should never be broken by the pruning logic?

## Key takeaway

Chapter 3 provides the immediate motivation for MCP-Prunex. Its value is in making the protocol problem concrete so the later implementation can be judged against a clear need.

# Chapter 4 Guide: Theoretical Background

## What this chapter does

Chapter 4 gives the theory needed to understand why response pruning might matter. It links three ideas: long-context limitations in large language models, the architecture of MCP-based tool calling, and prior research on context-compression methods.

## Context-window theory

The first section explains that context windows are finite and expensive. Even when models support very long contexts, more tokens still increase computation, latency, and cost. The chapter also highlights the "lost in the middle" problem, where relevant information becomes harder for the model to use when it is buried inside long inputs.

## MCP lifecycle

The MCP section is especially important because it explains where tokens accumulate. The thesis is not trying to optimize model weights or application prompts. It is trying to optimize what flows through the protocol when tools are listed, called, and returned back into the model context.

## Related work

The chapter distinguishes protocol-level response pruning from nearby ideas such as prompt compression, schema compression, and retrieval-augmented generation. This helps clarify the niche of the thesis. MCP-Prunex is not selecting records from a corpus; it is trimming the fields of already selected structured responses.

## What to watch carefully

This chapter includes several dense technical claims. While reading, check whether the theory directly supports the artifact design later on. Also note whether formulas, examples, and citations are complete enough to make the argument academically solid rather than only technically plausible.

## Key takeaway

Chapter 4 provides the conceptual justification for the thesis. It matters because it explains why field-level response pruning could improve not only token counts, but also the usefulness of model context under some conditions.

# Chapter 5 Guide: Middleware Implementation

## What this chapter does

Chapter 5 is the engineering core of the thesis. It explains what MCP-Prunex is, where it sits in the architecture, how it intercepts MCP messages, and how the pruning logic works.

## Proxy architecture

The middleware is described as a transparent proxy that acts both as an MCP server to the upstream client and as an MCP client to the downstream server. This dual role is important because it allows message interception without modifying the original tool server. The reader should treat this as the architectural basis for the claim that the artifact is reusable and domain-agnostic.

## What the middleware changes

The thesis says the proxy only intercepts `tools/list` and `tools/call`. In `tools/list`, it augments schemas with a field-selection parameter. In `tools/call`, it removes or forwards that parameter as needed, then prunes the returned JSON if pruning is enabled. This is the operational heart of the thesis.

## Filtering mechanism

The pruning logic supports simple keys and dot-path expressions. The algorithm walks the JSON tree recursively and preserves only the requested parts while trying not to corrupt the response structure. This section is where the reader should evaluate whether the thesis has described the artifact clearly enough for another engineer to reproduce it.

## Evaluation harness

The chapter also explains the external evaluation framework. This matters because the framework is not the same as the middleware itself. The thesis needs both: one artifact to optimize MCP responses, and one harness to test that artifact under controlled conditions.

## Key takeaway

Chapter 5 tells the reader exactly how the optimization works. If read carefully, it also reveals the practical boundaries of the artifact, especially its dependence on stdio transport, JSON payload structure, and reliable field selection by the model.

# Chapter 6 Guide: Results and Analysis

## What this chapter does

Chapter 6 presents the empirical evidence. It is the chapter that decides whether the thesis remains only a good engineering idea or becomes a convincing research contribution.

## What is measured

The chapter reports byte reduction, token reduction, semantic accuracy, Time-To-First-Token, and cost. These metrics correspond directly to the claims introduced earlier, although they do not all carry the same strength of evidence. Byte and token reductions are the clearest and most direct results. Accuracy and scalability claims require more careful interpretation.

## How to read the tables

The strongest reading strategy is comparative. For each model and domain, compare pruned mode against raw and unpruned mode separately. This helps reveal whether improvements come from pruning itself or whether the two-step pipeline introduces new overheads that the reader should not ignore.

## Important pattern in the findings

The chapter suggests that pruning helps most when the data are structured and independently addressable, such as SQL-style tables. It also suggests that pruning can hurt when the response contains semantically connected fields, such as live GitHub issue payloads. This boundary condition is one of the most important insights in the whole thesis.

## What to watch carefully

Pay close attention to denominator consistency, table references, and the exact meaning of latency claims. The chapter contains valuable results, but the reader should verify whether every numeric conclusion is stated with the correct scope and definition.

## Key takeaway

Chapter 6 is the empirical center of the thesis. Its most useful message is not that pruning always helps, but that pruning helps under specific structural conditions and can fail when meaning is distributed across many fields.

# Chapter 7 Guide: Discussion and Conclusions

## What this chapter does

Chapter 7 interprets the numbers from Chapter 6 and answers the research questions. This is where the thesis must shift from reporting measurements to making disciplined conclusions.

## Research-question answers

The chapter argues that MCP-Prunex reduces payload substantially, often reduces token-related latency and cost, and has model-dependent effects on accuracy. It also claims that the proxy overhead itself is small and that practical deployment should depend on the structure of the data being returned.

## Boundary conditions

One of the most valuable parts of the chapter is its recognition that pruning is not universally good. If a response contains highly interdependent fields, removing some of them can damage semantic interpretation. This moves the thesis away from a simplistic success story and toward a more credible engineering conclusion.

## Trade-offs and deployment meaning

The chapter is also where the reader should test the realism of the thesis. Claims about latency budgets, scalability, and production relevance should be read carefully against what was actually measured. Strong conclusions are only valid when the evidence matches the wording.

## Key takeaway

Chapter 7 is strongest when it presents MCP-Prunex as a conditional optimization strategy rather than a universal fix. The chapter should therefore be read as a statement about where protocol-level pruning is appropriate and where a fuller response remains necessary.

# Chapter 8 Guide: Summary

## What this chapter does

Chapter 8 compresses the whole thesis into a short final statement. It should restate the problem, the artifact, the major results, and the practical significance without introducing new technical detail.

## How to use this chapter

Read the summary after Chapters 6 and 7, not before. Its value is as a check on proportionality: does the final summary represent the real evidence fairly, or does it simplify too aggressively and overstate the contribution?

## What to watch for

The summary should repeat the most defensible contribution of the thesis: strong payload reduction, meaningful cost and latency implications, and clear boundary conditions for when pruning works well. If it omits limitations, the reader should mentally carry those forward from Chapter 7.

## Key takeaway

The summary is best read as the thesis's final calibration step. A good summary should sound slightly more cautious than the abstract, because by this point the exact strengths and limits of the evidence are already known.

# Chapter 9 Guide: References

## Purpose of this part

The references show whether the thesis is grounded in the right kinds of sources: protocol documentation, research on long-context behavior, context compression, tool-using LLMs, and research methodology.

## How to read this section

Do not read the list only for quantity. Instead, ask whether the references match the role they play in the thesis. Official MCP specification pages are appropriate for protocol facts. Research papers are appropriate for claims about context limits, tool use, compression, and evaluation. Methodology sources should support the research-design chapter directly.

## What to watch for

The most useful check is consistency. Citation style, completeness of entries, dates, and source types should follow one stable format. References should also align with the exact claims made in the abstract, introduction, theory, and method chapters.

## Key takeaway

The references are not just a bibliography. They are the evidence network behind the thesis's technical and methodological claims.

