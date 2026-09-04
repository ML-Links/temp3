# Front Matter and Abstract

## Overall assessment

The front matter communicates the practical intent clearly. It only needs a more careful wording of results, and a cleaner use of the university template.

---

## Issues to address

### 1. Must fix: Reduce and restructure the abstract

**What is wrong:** The abstract is longer than one page. It also spends too much space on implementation tools and makes result claims that are stronger than the evidence currently shown in Chapter 6.

**Why this matters:** The abstract is a short summary of the whole thesis. It should only make claims that you can prove later in the results chapter.

**How to fix:** Keep the abstract to one page. Use this simple order: problem, objective, method, what you built, how you tested it, main measured results, and conclusion. Remove long tool lists and keep only the details needed to understand the contribution.

### 2. Must fix: Explain the two parts of the project clearly

**What is wrong:** The abstract mentions the customer assistant and churn prevention, but it does not clearly explain what each part does, what data it uses, or how it was tested.

**Why this matters:** These are two different technical tasks. A good result for the chatbot does not automatically prove that the churn model works.

**How to fix:** Keep both contributions. Give each one a clear objective and one short summary of how it was tested. State that the work used test data or properly anonymized data. For churn, briefly define what counts as a customer leaving and make clear that campaigns are recommendations in a test or pre-production setting.

### 3. Should fix: Make the abstract's claims measurable

**What is wrong:** Phrases such as *"The RAG pipeline returned relevant results"* do not tell the reader how many questions were tested or what result was measured.

**How to fix:** Replace general claims with the real numbers. For example:

> "On a bilingual test set of [n] queries, the system achieved [metric] for English and [metric] for Finnish; these findings are limited by [key limitation]."

### 4. Must fix: Correct the supervisor information

**What is wrong:** In the abstract the supervisor field has "Sami, Instructor Lead assessor".  

**How to fix:** Please use the format of the thesis template name+title:

> **Supervisors: Sami, Senior Lecturer**


### 5. Should fix:

**What is wrong:** The preface has a warm and sincere tone, but some sentences are informal.

**How to fix:** Please polish the wording and check for small language issues such as `see **an** working system. `


---

# Chapter 1: Introduction

## Overall assessment

The chapter gives a clear operational account of the prepaid customer-service problem and explains the core idea of combining RAG with live API tools. It also identifies the accepted churn-prevention objective. The chapter needs major revision before it can serve as a Master thesis introduction because it lacks literature-grounded research framing and does not define separate validation strategies for its two contributions.

---

## Issues to address

### 1. Must fix: Define a coherent two-part research scope

**What is wrong:** The chapter presents churn prediction and retention-campaign generation as a secondary contribution, but does not define the churn event, prediction horizon, labelled test data, model output, or campaign-selection boundary.

**Why this matters:** The  proposal accepts the two related systems, but each makes a distinct claim. The customer assistant and churn model therefore need distinct research questions, data-governance controls, evaluation criteria, and reported results.

**How to fix:** Retain both contributions. Bound the churn component to a POC using test or anonymized data, define what counts as churn and over what period, and clarify that campaign generation is a recommendation or test-environment output rather than an autonomous production intervention. Introduce the operational churn definition already in Chapter 1 and keep the same definition in Chapters 3, 5, 6, and 7 instead of waiting until the implementation chapter to make it precise.

### 2. Must fix: Replace the two broad questions with precise, answerable research questions

**What is wrong:** The first research question asks generally how a system can be designed and implemented; the second asks generally how data can be used for churn prevention. Neither is sufficiently measurable, and the assistant question omits Finnish-versus-English performance and latency/accuracy trade-offs.

**How to fix:** Use precise RQs for both contributions. The assistant RQs should cover secure tool/API integration, Finnish-versus-English retrieval/answer quality, and latency. The churn RQ should evaluate prediction performance against a defined churn label and the suitability of campaign recommendations. Define the metric, dataset, and unit of analysis later in Chapter 2.

### 3. Must fix: Ground the problem and contribution in literature

**What is wrong:** The chapter makes many external claims about LLM capabilities, customer-service cost, and multilingual needs without citations, and it does not identify a research gap beyond the case company's need.

**Why this matters:** A Master thesis introduction must establish why the engineering problem remains unsolved in the field, not merely that a company has a useful project.

**How to fix:** Add a short, cited problem framing that distinguishes prior RAG/customer-service work from the thesis's intended contribution: evaluating a bilingual, tool-using RAG artifact in a telecom setting. State the practical and academic contributions separately.

### 4. Must fix: State how the contribution will be validated

**What is wrong:** The introduction describes the solution but not a reproducible validation plan.

**How to fix:** Add a concise method overview: Design Science Research; a documented bilingual golden dataset; retrieval and answer-support metrics; tool-routing correctness; latency measurement; comparison by language; and the stated limitations. Do not report results here.

### 5. Should fix: Tighten scope and chapter structure

**What is wrong:** The scope lists implementation activities but does not clearly bound the research population, data access, model/provider conditions, or evaluation setting.

**How to fix:** Define the specific case context, supported languages, test environment, API data restrictions, excluded production deployment and user study. Keep the thesis-structure paragraph as the final subsection.

### 6. Should fix: Make the citation problem visible already in the introduction

**What is wrong:** The chapter currently presents many external claims about telecom service needs, multilingual support, LLM capabilities, and churn behavior without in-text citations.

**Why this matters:** This is not only a Chapter 4 literature issue. If the introduction itself is uncited, the full thesis immediately reads more like a project report than a research document.

**How to fix:** Add citations in Chapter 1 wherever you define the practical problem, motivate the bilingual assistant, describe RAG/tool-calling relevance, or characterize prepaid churn behavior. Keep the detailed literature review in Chapter 4, but make sure the introduction still cites the sources it depends on.

## Next steps

1. Rewrite the RQs, contribution statement, and validation overview as one consistent two-part research frame.
2. Define the churn label, test-data boundary, and campaign-output limitation.
3. Add cited field context and submit Chapter 1 together with Chapter 2.

---

# Chapter 2: Method and Material

## Overall assessment

The four-phase structure makes the development sequence easy to follow, and the chapter appropriately acknowledges that UAT-only testing and author-led manual assessment limit the evidence. The chapter nevertheless remains a project process description rather than a defensible Master-level research method. It needs a clearly defined and cited research design, with Design Science Research as the recommended option.

---

## Issues to address

### 1. Must fix: Use and justify one established research design

**What is wrong:** This chapter calls the method "constructive research" without definition, citation, or justification. The earlier suggested proposal recommended Design Science Research, but that recommendation does not replace the student's original proposal. The chapter also does not explain how the method produces knowledge beyond a working system.

**How to fix:** Use Design Science Research, or another established method that is clearly defined and justified. If you choose Design Science Research, cite the methodological basis and map the work to its stages: problem identification, objectives, artifact design/development, demonstration, and evaluation. Explain why it fits the RQs. If you keep the term "constructive research" where it first appears, cite the existing methodology source already present in the reference list directly in that sentence instead of naming the method without attribution.

### 2. Must fix: Define the materials and evaluation protocol reproducibly

**What is wrong:** The chapter does not specify the data sources and snapshots, the bilingual golden dataset, query categories, sample size, inclusion criteria, ground-truth creation, model version, prompts, retrieval settings, or scoring procedure. "Manual evaluation" is insufficient.

**Why this matters:** Another researcher cannot reproduce or assess the claims about retrieval quality, routing correctness, Finnish/English performance, or latency.

**How to fix:** Add a Materials section and an Evaluation Protocol. State exactly what was collected, how personal data were excluded/anonymized, the number and balance of Finnish/English queries, expected tools/answers, evaluation metrics, baselines, repeated runs, and analysis method. Report results only in the later results/evaluation chapter.

### 3. Must fix: Separate software tests from research evaluation

**What is wrong:** HTTP status, authentication middleware, conversation-memory, and hashing tests are useful engineering tests, but they do not answer the research questions on RAG quality, grounded answers, language comparison, or latency.

**How to fix:** Retain unit/integration tests as artifact verification, then add research evaluation measures, for example retrieval Recall@k or context precision, grounded-answer correctness against a rubric, tool-selection accuracy, latency percentiles, and a Finnish-versus-English comparison. State who performs human scoring and how agreement or bias is addressed.

### 4. Should fix: Strengthen reliability, validity, ethics, and AI-use reporting

**What is wrong:** Public frameworks and Git version control do not themselves establish research reliability. The chapter does not address API credentials, PII handling, web-scraping permission/terms, evaluator bias, dataset leakage, or model non-determinism. The AI-use statement appropriately names the tools and confirms the author's responsibility for the final text and code, but it should state the boundary of this assistance more precisely.

**How to fix:** Describe controls for reproducibility, versioning, seeds/settings, repeatability, scoring rubric, data governance, and threats to construct/internal/external validity. Retain the existing authorship declaration, then clarify whether AI assistance was limited to brainstorming and coding support, how generated suggestions were checked, and how the student retained responsibility for the research decisions. Retain evidence of that process if required.

### 5. Should fix: Cite methodological and measurement claims

**What is wrong:** Definitions of constructive research, reliability, and validity are presented without sources.

**How to fix:** Cite credible methodology sources and evaluation literature; use the citation style consistently. At minimum, ensure that the first mention of the chosen research approach in this chapter has an immediate supporting citation instead of postponing all references to the bibliography chapter.

### 6. Must fix: Explain the missing pre-production user verification

**What is wrong:** The original proposal says that a group of company users would verify the solution in a pre-production environment. This chapter reports only author-led UAT and says the system was not tested with real customers, but it does not say whether the planned internal user verification happened.

**Why this matters:** Internal user verification is different from testing with real customers. If it happened, it is important evaluation evidence. If it did not happen, the thesis needs to report this honestly as a limitation.

**How to fix:** State clearly whether a company user group evaluated the pre-production POC. If yes, describe their roles, test tasks, feedback method, confidentiality handling, and results. If no, explain why it was not completed and list it as a limitation and future step.

## Next steps

1. Rewrite the chapter around a single, cited Design Science Research design.
2. Specify the datasets, metrics, procedures, and validity controls before producing Chapter 6.
3. State whether the planned pre-production user verification was completed.
4. Align all method choices with the final research questions.

---

# Chapter 3: Current State Analysis

## Overall assessment

This chapter gives a clear picture of the current prepaid customer-service channels and the problems that the new system should solve. It is a good start, especially the distinction between general help content and account-specific API data. The next step is to show evidence for the problems and turn them into clear requirements for both parts of the project.

---

## Issues to address

### 1. Must fix: Support the current-state findings with documented evidence

**What is wrong:** You say that customers are frustrated, call-center work is expensive, and information is fragmented, but the chapter does not show where this information comes from.

**Why this matters:** Your company knowledge is valuable, but a thesis still needs to show the reader why these statements can be trusted.

**How to fix:** Use safe company evidence. You can cite internal reports, anonymized analytics, or a short interview with a manager. You can also describe observations from your professional role, but label them clearly as participant observation. Protect confidential information by using percentages or ranges instead of exact numbers. For example:

> "Internal customer-service data from Q2 2026 show that approximately 40-50% of prepaid queries concerned balance and packages (Internal Customer Service Report, 2026)."

### 2. Must fix: Turn the problems into clear requirements

**What is wrong:** The chapter ends with a list of problems, but it does not clearly say what the customer assistant and churn POC must do to solve them.

**How to fix:** Add a simple table with: problem, requirement, priority, and how you will test it. For example, the language gap should become a requirement that the assistant handles Finnish and English, followed by a test that compares results in both languages. Add separate requirements for churn prediction and campaign recommendations.

### 3. Must fix: Evidence the churn-prevention problem and define its boundaries

**What is wrong:** You explain that the company has no strong churn-prevention process, but you do not explain how this is known or what exactly counts as churn in your project.

**How to fix:** Keep this section and support it with safe company evidence, such as an internal report or an interview with a prepaid-service manager. Define churn clearly, for example, no refill within a chosen number of days. Use the same operational definition consistently across later chapters instead of introducing a more specific threshold only in Chapter 5. Explain that the model uses test or anonymized data and that campaign suggestions need business approval before any real customer action.

### 4. Should fix: Analyze root causes and constraints, not only symptoms

**What is wrong:** The chapter describes the channels, but it does not yet explain the main reasons why the problems happen.

**How to fix:** Add a simple process diagram or table showing where customers get stuck, where information is missing, and where secure access to account data is needed. Keep the focus on problems that affect your research questions.

### 5. Should fix: Clarify the case-company evidence and API assumptions

**What is wrong:** It is not clear whether the APIs use production data, UAT data, mock data, or test data. The reader also needs to know what customer data are allowed in the project.

**How to fix:** State the environment and the permitted data clearly. Explain which data are test or anonymized, and what users are allowed to access. Leave the technical security solution for Chapter 5.

## Next steps

1. Add traceable evidence for the current-state findings.
2. Produce a problem-to-requirement table for the customer assistant and churn POC.
3. Add evidence, scope boundaries, and traceable requirements for the churn-prevention component.

---

# Chapter 4: Theoretical Background

## Overall assessment

You clearly understand the technologies behind the customer assistant and the churn POC. The sections on RAG, tool calling, web scraping, and churn prediction show useful technical knowledge. To make this a Master-level literature chapter, you now need to support the claims with sources and explain why your chosen methods fit this project.

---

## Issues to address

### 1. Must fix: Add complete, credible citations throughout

**What is wrong:** The chapter makes many claims about LLMs, RAG, agents, web scraping, and machine learning, but it has no in-text citations. The current references do not support the main topic.

**Why this matters:** A literature chapter must show where the information comes from and how your work builds on earlier research.

**How to fix:** Add a citation whenever you explain an external idea. Use research papers for theory and official documentation for technical facts about a tool. Use one style consistently, preferably Vancouver numbering. Every citation in the text must appear in the reference list.

### 2. Must fix: Turn descriptive summaries into critical synthesis linked to the RQs

**What is wrong:** Much of the chapter explains what each technology is, but it does not compare options or explain the limits of the chosen approach.

**How to fix:** For each important choice, explain the alternatives and why your choice fits the project. For example: Why use RAG? Why use an agent instead of fixed rules? What makes Finnish harder than English? What are the risks when an LLM calls customer-data APIs? End each topic by explaining how it affects your design or testing.

### 3. Must fix: Make the churn-prevention literature stream rigorous and research-relevant

**What is wrong:** The churn section belongs in the thesis, but it says that XGBoost and the chosen risk thresholds are suitable without sources or comparison. It also does not explain how churn is defined or how the model will be tested fairly.

**How to fix:** Keep the section. Compare XGBoost with a simple baseline model, such as logistic regression. Explain the churn definition, the prediction period, imbalanced data, and the measures you will use. Do not choose risk thresholds only because they look reasonable; justify them with data or business rules. Also explain that a risk score does not prove that a campaign will stop churn.

### 4. Must fix: Correct technical overgeneralizations and separate theory from implementation

**What is wrong:** Some technical explanations are too general or not fully accurate. The chapter also includes detailed implementation choices that belong in Chapter 5.

**How to fix:** Check each technical claim using a research paper, standard, or official documentation. Keep the general theory here and move exact settings, library configuration, and code to Chapter 5.

### 5. Should fix: Add missing research-critical topics

**What is wrong:** The chapter gives too little attention to how Finnish and English will be compared, how answer quality will be measured, and how telecom customer data will be protected.

**How to fix:** Add short, source-based sections on bilingual evaluation, RAG evaluation measures, prompt injection, tool authorization, data minimization, and GDPR. Only include theory that helps explain a design choice or a test in your thesis.

### 6. Must fix: Address Finnish retrieval and hosted embeddings

**What is wrong:** The thesis supports Finnish and English, but it does not discuss Finnish compound words and inflection, or whether the selected multilingual embedding model and chunking approach work equally well for Finnish. It also uses Google's hosted `text-embedding-004` model without discussing the privacy or open-source implications.

**Why this matters:** Finnish and English may have different retrieval quality. Sending operator content or user queries to a hosted embedding service needs the same data-governance review as sending data to a hosted LLM.

**How to fix:** Add a cited discussion of Finnish-language retrieval challenges and compare the chosen approach with suitable local multilingual alternatives. Do not assume that 500-character recursive chunks work well: measure retrieval quality separately for Finnish and English. Explain what data are sent to the hosted embedding service, why this is permitted, and what limitation this creates for the open-source objective.

## Next steps

1. Rebuild the chapter from a literature matrix of claim, source, finding, limitation, and thesis implication.
2. Rebuild the churn literature as a critical, cited basis for the POC model and its evaluation.
3. Add Finnish retrieval and hosted-embedding evidence.
4. Ensure Chapter 5 design decisions cite the revised theoretical synthesis.

---

# Chapter 5: System Design and Architecture

## Overall assessment

This is the strongest technical chapter. The customer-assistant design is modern and well thought out: it combines RAG, tool calling, bilingual content, session handling, and deployment in a clear way. The churn idea is also useful, and the attempt to combine a practical assistant with a separate churn POC shows initiative. That creativity should be acknowledged, but any divergence from the originally stated architecture or from standard practice still needs to be justified explicitly. To make the chapter complete, show the missing churn architecture, explain the model choice, and connect every important design choice to a requirement and test.

---

## Issues to address

### 1. Must fix: Specify and validate the accepted churn subsystem

**What is wrong:** FR9-FR11 include churn detection and campaign recommendations, but the chapter does not explain what churn means, how the model was trained and tested, or who approves a campaign recommendation.

**How to fix:** Keep these requirements and make them easy to test. Define the churn label, data features, training and test sets, and a simple baseline model. Report suitable measures, such as precision, recall, F1, PR-AUC, and calibration. Make clear that campaign recommendations are for testing or pre-production review, not automatic action for real customers.

### 2. Must fix: Add the missing churn architecture

**What is wrong:** The abstract says that the churn solution uses MariaDB, APScheduler, FastAPI, and XGBoost. These technologies are missing from the Chapter 5 architecture, diagram, and implementation explanation.

**Why this matters:** The reader cannot understand how the churn POC actually works if the database, scheduled analysis, model, and campaign recommendations are not described together.

**How to fix:** Add a subsection and diagram showing: data stored in MariaDB; the APScheduler job; feature creation; XGBoost or baseline model scoring; risk storage; campaign recommendation; and the review or approval step. Explain how this connects to the FastAPI service.

### 3. Must fix: Establish requirements traceability and design justification

**What is wrong:** The requirements are listed, but the chapter does not show which problem each requirement solves or how each feature will be tested. Some design choices are described as good choices without explaining why.

**How to fix:** Add a simple table: current problem -> requirement -> design feature -> test result. For important choices such as Gemini, ChromaDB, agent routing, chunk size, PUK verification, and language support, explain the reason for the choice and link it to Chapter 4.

### 4. Must fix: Explain the open-source-model decision

**What is wrong:** The original proposal says the POC will use open-source models and tools. Chapter 5 uses Google Gemini as the default model, which is a proprietary hosted service, even though the architecture also supports a local Llama deployment.

**Why this matters:** This is an important difference between the proposal and the implementation. It is acceptable to make a practical choice, but the reader needs an honest explanation.

**How to fix:** Explain why Gemini was used, for example performance, cost, language quality, or available hardware. Then show that the architecture supports a local open-source Llama model. If possible, include one small comparison or state clearly that evaluating open-source models is a limitation and future task.

### 5. Must fix: Treat security and privacy as verifiable properties

**What is wrong:** The chapter uses a PUK code for account verification in a web chat, but it does not justify this choice against the operator's approved identity and access-management practice. It also does not show whether PUK values, MSISDNs, or account data are removed before messages are stored in conversation history or passed to a hosted LLM.

**How to fix:** Treat PUK verification as a prototype assumption, not a production authentication design, unless the operator's security team has approved it. Add a data-flow diagram showing whether PUK values and MSISDNs can enter LLM context or SQLite history. Keep verification data in a separate server-side flow, redact it from logs and history, and never send it to the LLM. Add tests for invalid PUK, expired session, unauthorized account access, and attempted secret disclosure.

### 6. Must fix: Explain the regex gate and agent routing together

**What is wrong:** The chapter says that the LLM agent autonomously selects tools and that no intent classifier is needed. Later, it says a regular expression detects account-specific phrases to trigger PUK verification. It does not explain which component runs first or how the regex gate interacts with the LangGraph ReAct agent.

**Why this matters:** This may be a sensible hybrid safety design, but the reader cannot tell whether routing is autonomous, rule-based, or both. It also affects how agent-routing accuracy should be evaluated.

**How to fix:** Add a request-flow diagram and explain each step. State whether the regex runs before the agent, after an account-tool request, or as a separate policy check. Explain how Finnish variants and unexpected wording are handled. Evaluate the rule-based gate separately from LLM tool selection.

### 7. Must fix: State how the two POCs are related

**What is wrong:** The thesis calls the customer assistant and churn prevention "complementary," but the five listed agent tools do not access the churn service, risk score, campaign eligibility, MariaDB, or scheduler. The integration boundary is not described.

**Why this matters:** The POCs may be intentionally separate, but the reader needs to know this. Otherwise it is unclear whether an at-risk customer can receive a recommendation through the assistant or whether the two systems only share the same business domain.

**How to fix:** State clearly whether the POCs are separate prototypes or whether the churn service provides an approved campaign-eligibility input to the assistant. If they are integrated, show the API/data flow and authorization boundary. If they are separate, explain why and evaluate them separately.

### 8. Must fix: Move long code listings and make diagrams/tables useful

**What is wrong:** Long Python and Docker code blocks take up a lot of space, while the diagrams and tables are not fully explained.

**How to fix:** Keep only short code examples that explain an important design choice. Move full code and the full Docker file to an appendix or project repository. Introduce every figure and table before it appears, then explain what the reader should learn from it.

### 9. Should fix: Clarify reproducibility and multilingual design

**What is wrong:** The chapter gives settings such as chunk size and temperature but omits source snapshots, versions, model identifiers/dates, prompt version, retrieval configuration, fallback behavior, and how language detection or mixed-language queries are handled.

**How to fix:** Add a reproducibility table and a concise bilingual-design specification. These details will make the Chapter 6 comparison credible.

### 10. Should fix: Correct chapter role and numbering

**What is wrong:** The title and contents call this a design/implementation chapter, while the suggested outline positions design and implementation as results supporting RQ evaluation. It also labels the provider abstraction as Section 5.7 in the text while earlier material refers to Section 5.8.

**How to fix:** Use one consistent structure across contents and text. Ensure the chapter leads naturally to a completed evidence-based Chapter 6 rather than ending with a technical implementation inventory.

### 11. Should fix: Cross-reference the appendices from the chapter text

**What is wrong:** The appendices contain useful supporting material, especially the system prompt and project structure listings, but Chapter 5 does not clearly point the reader to them where those artifacts are discussed.

**Why this matters:** Without chapter-to-appendix cross-references, useful technical evidence remains detached from the main explanation and an examiner may miss that the supporting material exists.

**How to fix:** Add explicit references from the relevant Chapter 5 sections to the appendices, especially where the system prompt, the assistant project structure, and the churn project structure are discussed. Make sure the appendix labels, figure/listing captions, and chapter text use the same naming.

## Next steps

1. Define separate, testable requirements for the accepted customer-assistant and churn-prevention POCs.
2. Explain the Gemini and hosted-embedding choices against the open-source objective and data-governance requirements.
3. Add a request-flow diagram for regex, PUK verification, and the LLM agent.
4. State whether the two POCs are separate or integrated, then add traceability and security/data-flow diagrams.
5. Move detailed code to appendices, cross-reference those appendices from the chapter text, and prepare the evaluation evidence for Chapter 6.

---

# Chapter 6: Results and Evaluation

## Overall assessment

This chapter is a meaningful step forward because it attempts automated evaluation rather than relying only on informal demonstration. The use of pytest, RAGAS, and TruLens shows initiative, and that is worth acknowledging.

The chapter still needs major revision before it can support the thesis claims. At present, it mixes software verification, LLM-judge evaluation, and broad success statements without a sufficiently defined evaluation protocol. The customer-assistant results are only partly evidenced, and the churn-prevention results are largely asserted rather than demonstrated.

---

## Issues to address

### 1. Must fix: Separate software tests from research evaluation

**What is wrong:** The chapter begins with API, authentication, memory, and hashing tests, then moves directly to RAGAS and TruLens scores. These are not the same type of evidence. Engineering tests show that certain components run correctly. They do not by themselves answer the research questions about answer quality, bilingual performance, routing quality, or churn-prediction usefulness.

**How to fix:** Split the chapter into clearly labelled evidence layers: artifact verification, assistant evaluation, and churn evaluation. Under each layer, state exactly which research question or requirement the evidence addresses.

### 2. Must fix: Define the evaluation dataset and protocol reproducibly

**What is wrong:** The chapter does not state the exact size and composition of the evaluation dataset, the Finnish-versus-English split, the inclusion criteria, the expected answers or tools, the source of ground truth, or whether runs were repeated. Phrases such as "full evaluation dataset" remain too vague.

**Why this matters:** Without a reproducible protocol, the reported scores cannot be properly interpreted or repeated by another researcher.

**How to fix:** Add a compact protocol table with at least: number of queries, language balance, query categories, expected tool path, ground-truth construction method, judge or scorer, number of runs, model versions, and how final scores were aggregated.

### 3. Must fix: Do not treat LLM-judge scores as sufficient proof on their own

**What is wrong:** Both RAGAS and TruLens appear to rely on Gemini-based judging while the system itself also uses Gemini-family services. That creates a risk of circular evaluation and judge bias. The chapter currently treats the resulting scores as direct proof that the assistant is accurate and correctly routed.

**How to fix:** Keep the automated metrics, but describe them as supporting evidence rather than final proof. Add independent checks such as manually verified gold answers, expected-tool labels, or a human scoring rubric on a clearly defined subset.

### 4. Must fix: Support the routing claims with direct routing evidence

**What is wrong:** The chapter says that the agent correctly routed queries to the right tools, but it does not show the number of routing cases, the expected tool labels, the number of errors, or any confusion breakdown.

**How to fix:** Report routing accuracy explicitly. For example, show how many queries were expected to use RAG only, API only, multiple tools, or rejection. Then report how many were routed correctly in each category.

### 5. Must fix: Add the missing churn-prevention evaluation

**What is wrong:** The abstract, Chapter 5, and Chapter 7 all say that the churn-prevention subsystem detected usage drops, calculated risk, and generated campaigns correctly. Chapter 6 does not provide enough direct evidence for these claims. There are no dataset details, no label definition, no train-test split, no baseline comparison, and no performance metrics for the churn model.

**Why this matters:** This is the largest evidence gap in the full thesis. The churn component is currently described more strongly than it is evaluated.

**How to fix:** Add a separate churn-results subsection. Define churn, describe the data source and anonymization boundary, report the training and test setup, include at least one baseline, and provide appropriate metrics such as precision, recall, F1, PR-AUC, and calibration or threshold analysis. If this evaluation was not completed, say so clearly and downgrade the claims across the thesis.

### 6. Must fix: Avoid unsupported requirement-completion claims

**What is wrong:** The summary says that all functional and non-functional requirements were met, including the churn requirements. The evidence presented in this chapter does not yet support such a complete claim.

**How to fix:** Replace the blanket statement with a traceable requirement-status table. Mark each requirement as demonstrated, partially demonstrated, or not yet fully evaluated.

### 7. Should fix: Report bilingual performance separately

**What is wrong:** The chapter says the system worked for both Finnish and English, but the results are not separated by language.

**How to fix:** Report at least a small Finnish-versus-English comparison for retrieval quality, answer quality, and routing behavior. This would strengthen one of the more interesting aspects of the thesis.

### 8. Should fix: Make latency reporting more informative

**What is wrong:** One average latency value from TruLens is not enough to understand practical responsiveness.

**How to fix:** Report at least median and high-percentile latency, and distinguish simpler knowledge-base queries from account-specific multi-step flows.

## Next steps

1. Reorganize the chapter into artifact verification, assistant evaluation, and churn evaluation.
2. Add a reproducible evaluation-protocol table before interpreting any scores.
3. Either add real churn-model evidence or reduce the churn claims across the thesis.
4. Replace blanket success statements with requirement-by-requirement evidence.

---

# Chapter 7: Discussions and Conclusions

## Overall assessment

The chapter is confident, readable, and clearly written. It also shows that you are thinking beyond pure implementation toward operational usefulness, which is a positive sign.

The main problem is that the discussion and conclusion are currently more confident than the evidence in Chapter 6 allows. Some claims are reasonable as engineering expectations, but they are not yet fully demonstrated as thesis findings.

---

## Issues to address

### 1. Must fix: Align the conclusions with the actual evidence

**What is wrong:** The chapter says that all objectives were met, that all requirements were satisfied, and that the resulting system works effectively as an enterprise-grade AI application. These are broader claims than the current evaluation supports.

**How to fix:** Rephrase the chapter so that it distinguishes between what was implemented, what was demonstrated in testing, and what remains a plausible but unproven production benefit.

### 2. Must fix: Treat the churn contribution more cautiously

**What is wrong:** The second research question is answered as if the thesis had already demonstrated a validated ML-based churn solution. Based on the current results chapter, that conclusion is not yet sufficiently supported.

**How to fix:** Either provide the missing churn evidence in Chapter 6 or rewrite the Chapter 7 discussion so that the churn component is presented as a designed and partially implemented POC rather than a fully validated result.

### 3. Must fix: Resolve the routing-governance contradiction

**What is wrong:** The chapter presents the assistant as a purely autonomous ReAct agent, but earlier chapters also describe a regex-based gate and PUK verification logic that constrain how routing works.

**Why this matters:** This is not a small wording issue. It affects the actual thesis contribution and the meaning of the routing results.

**How to fix:** Describe the architecture honestly as a hybrid design if that is what was built: policy checks and verification logic around an LLM-driven tool-selection core.

### 4. Should fix: Separate findings from recommendations

**What is wrong:** Some practical recommendations are written as if they were confirmed results.

**How to fix:** Use clear wording such as "the prototype suggests" or "a reasonable deployment path would be" when you are extrapolating beyond the measured evidence.

### 5. Should fix: Expand the limitations to include evaluation-governance issues

**What is wrong:** The limitations section already includes several relevant points, but it omits some important issues such as small labelled evaluation size, possible LLM-judge circularity, missing independent human scoring protocol, and incomplete evidence for the churn subsystem.

**How to fix:** Add these explicitly. A stronger limitations section will improve the credibility of the whole thesis.

### 6. Should fix: End with a tighter research next step

**What is wrong:** The future-work section is broad and useful, but it would be stronger if it prioritized the single most important next validation step.

**How to fix:** End the chapter with one focused next study: a controlled bilingual assistant evaluation with fixed gold labels and a separate, labelled churn-model evaluation with approved business thresholds.

## Next steps

1. Rewrite the discussion after Chapter 6 is finalized.
2. Keep the conclusions narrower than the current implementation ambition.
3. Present the churn subsystem as validated only to the extent that Chapter 6 can actually demonstrate.
4. Use the final paragraph to state the strongest supported contribution, not the broadest possible one.

---

# References

## Overall assessment

The references list is now materially better than in the earlier partial submission: it includes relevant sources on LLMs, RAG, telecom, and churn. That is a useful improvement. However, the chapter text still contains almost no in-text citation use, and the reference list itself is not yet clean or consistent enough for a Master thesis.

---

## Issues to address

### 1. Must fix: Add complete in-text citation coverage

**What is wrong:** All seven submitted chapters contain no usable in-text citations. A check of the full text (front matter, Chapters 1-7) found zero citation markers of any kind, not just a shortfall in one or two chapters.

**How to fix:** Cite each external claim where it appears, using one consistent style. A reader must be able to identify whether a sentence is supported by a research paper, standard, company/internal document, or official technical documentation.

### 2. Must fix: Build a relevant and current source base

**What is wrong:** The list already includes three relevant telecom churn-prediction papers (Ahn et al., Vafeiadis et al., and Huang et al.), which is a good base to build on. However, the list still misses some source categories that this thesis now clearly needs: Design Science Research or equivalent methodology sources (only the Kasanen et al. constructive-research paper is present), bilingual/Finnish retrieval evaluation, privacy/GDPR and security-governance sources, and evidence for retention-campaign effectiveness (e.g. uplift modelling) rather than only churn-risk prediction.

**How to fix:** Build a balanced reference base including peer-reviewed work on RAG, retrieval evaluation, answer evaluation, LLM agents/tool calling, multilingual or Finnish retrieval, conversational AI in telecom, and Design Science Research. Supplement the existing churn-prediction references with literature on label definition, class imbalance, leakage, calibration, fairness, and campaign-effectiveness evaluation. Use standards and official documentation for GDPR/privacy, security controls, and framework/API implementation facts.

### 3. Must fix: Use a complete, consistent citation format

**What is wrong:** The entries are inconsistently formatted. Reference 3 (Vaswani et al., "Attention is all you need") is written in APA style, with authors joined by "&" and "(2017)" placed right after the author list, while every other entry uses numbered Vancouver style. There are also encoding problems (e.g. "Kaiser, ?." in reference 3, "R?ileanu" in reference 11) and inconsistent capitalization and punctuation across entries.

**How to fix:** Use the university-required format consistently, preferably numbered Vancouver style throughout, including reference 3. Verify every URL, DOI, author list, title, venue/publisher, volume/issue/pages where applicable, and access date for web sources. Check that special characters render correctly.

### 4. Must fix: Remove weak or incomplete entries before the final pass

**What is wrong:** Some entries appear incomplete or weakly documented for final submission. For example, reference 21 (the XGBoost paper) lists no authors at all, and several arXiv entries would benefit from cleaner version/date details (an arXiv ID and access URL, as reference 16 already has) or replacement with peer-reviewed sources where available.

**How to fix:** Check each entry one by one against the original source. Where a peer-reviewed publication exists, prefer it over a generic arXiv citation unless the arXiv version is intentionally required.

### 5. Should fix: Keep a source-to-claim record while revising

**How to fix:** Maintain a simple literature matrix with the source, the claim it supports, its limitation, and the chapter/section where it is cited. This will prevent unsupported descriptive text and help Chapter 4 become critical rather than encyclopedic.

## Next steps

1. Add citations while rewriting Chapters 1–5; do not postpone references until the final formatting stage.
2. Check that every reference is cited and every in-text citation resolves to one complete entry.
3. Add the missing methodology, bilingual-evaluation, privacy, and security-governance sources.
4. Resubmit the complete reference list with the revised theoretical background.

---

# Appendices

## Overall assessment

The appendices are useful and relevant. Including the system prompt and the project structures helps the reader understand the implemented artifact, and that is a good choice.

To support a Master-level thesis more effectively, the appendices should now do more than describe the system. They should also strengthen traceability, reproducibility, and auditability.

---

## Issues to address

### 1. Must fix: Cross-reference the appendices from the main chapters

**What is wrong:** The appendices exist and are relevant, but the main chapters do not clearly direct the reader to them where the prompt and project structures are discussed.

**Why this matters:** Good appendices support the thesis argument. They should not sit beside the main text as disconnected attachments.

**How to fix:** Add explicit cross-references from the relevant body sections, especially in Chapter 5, so the reader is told where to find the system prompt and project-structure evidence.

### 2. Must fix: Add evaluation artifacts, not only implementation artifacts

**What is wrong:** The appendices show the system prompt and file structures, but they do not include the most important supporting material for the research claims: evaluation dataset examples, scoring rubric, expected-tool labels, or retained result artifacts.

**How to fix:** Add appendices or appendix tables for the evaluation protocol, sample test questions, expected outcomes, and retained evidence such as logs, trace exports, or summary result files if confidentiality allows.

### 3. Must fix: Make the system-prompt appendix traceable to the evaluated system

**What is wrong:** The prompt text is helpful, but the appendix does not say whether this is the exact prompt version used in the reported evaluation, whether it changed during development, or how it relates to the final reported results.

**How to fix:** Label the prompt with a version, date, or hash, and state whether it is the final evaluated prompt. If several prompt versions were used, say so clearly.

### 4. Must fix: Document the sensitive-data boundary in the prompt and evidence

**What is wrong:** The prompt says that account-specific tools may be used when an MSISDN is included in the message, but it does not show that the identifier, PUK code, and returned account data are verified and handled in a separate server-side flow before the LLM receives any context. It also does not state how these values are redacted from conversation history, logs, and evaluation traces.

**Why this matters:** An MSISDN and PUK code are sensitive. A customer identifier in a chat message must not act as authorization, and the thesis needs to show that private values do not enter persistent records or hosted-model context unnecessarily.

**How to fix:** Add a short data-flow appendix or cross-reference to Chapter 5. Show server-side authentication and authorization, explicit redaction from prompts, histories, logs, and traces, and the prototype limitation of PUK verification. Include the relevant negative tests, such as an unverified account request and an attempted disclosure of another subscriber's data.

### 5. Must fix: Distinguish implemented structure from fully validated structure

**What is wrong:** The project file trees are informative, but for the churn subsystem they may give the impression that every listed component was implemented and evaluated to the same maturity level.

**How to fix:** Add one short note that distinguishes implemented modules, planned modules, and evaluated modules where necessary. This is especially important for the churn-prevention project structure in Appendix 2, since Chapter 6 does not yet provide churn-model evaluation evidence to match it.

### 6. Should fix: Add a compact artifact inventory

**What is wrong:** An examiner cannot easily see which artifacts exist and which ones support which chapter claims.

**How to fix:** Add a simple inventory table with artifact name, purpose, chapter connection, and access status.

### 7. Should fix: Check appendix formatting and readability

**What is wrong:** The appendix formatting is understandable, but the file-tree listings are visually dense.

**How to fix:** Improve spacing, indentation, and captioning so the appendices are easier to read in the final PDF.

## Next steps

1. Keep the existing appendices, but add evaluation-supporting appendices.
2. Version the system prompt used for the reported results.
3. Document server-side authorization and redaction of sensitive data before it reaches the LLM or persistent records.
4. Add an artifact inventory so the appendices support auditing as well as explanation.

---

# General Feedback: Full Thesis Submission

## Overall assessment

The submitted work shows substantial practical effort, clear independence in implementation, and a creative attempt to combine multilingual self-service with proactive churn prevention in one telecom-focused thesis. The project idea is strong and the artifact-building effort is visible throughout the document. The thesis is not yet ready for final evaluation because the academic framing, evidence quality, and validation coverage still lag behind the implementation ambition.

---

## Accepted scope and required framing

The original proposal includes both an AI-powered self-service assistant and a churn-prevention POC using test data. The thesis may therefore retain both contributions. It must, however, use a two-part research frame: the bilingual Agentic RAG assistant requires retrieval, answer-quality, tool-routing, and latency evidence; the churn component requires an explicit churn definition, labelled test data, model-performance and calibration evidence, campaign-decision limitations, and privacy/fairness assessment.

## Recommendation

Continue with both accepted contributions, but make their research questions, datasets, methods, measures, and results visibly separate. Keep churn detection and campaign generation as a test or pre-production decision-support POC; do not make claims about production impact or autonomous campaigns without real-world evidence and authorization.

## Submission status

The full submission has now been reviewed, including the front matter, Chapters 1-7, references, and appendices. The largest remaining problem is not missing text volume. It is the gap between what the thesis claims and what it currently demonstrates in a reproducible, research-grade way.

## Main strengths

1. The thesis tackles a relevant real-world telecom problem with good practical intuition.
2. The customer-assistant architecture is thoughtfully assembled and shows technical maturity.
3. The bilingual focus and the attempt to combine RAG with live tools are valuable and potentially distinctive.
4. The churn-prevention idea is relevant and shows initiative, even though its validation is still too thin.

## Main risks

1. The thesis often presents implementation success as research validation.
2. The churn component is described confidently across the thesis, but Chapter 6 does not yet provide sufficient evidence to support those claims.
3. The literature base exists, but it is not yet integrated into the chapter text through proper citation and critical synthesis.
4. The citation problem is thesis-wide, not local to one literature chapter: Chapters 1-7 currently do not provide normal in-text source support for external claims.
5. Relevant appendices exist, but the main text does not yet use them well enough through explicit cross-references.
6. Security, privacy, and evaluation-governance questions are acknowledged unevenly and need more explicit treatment.

## Priority order for revision

1. Apply the accepted two-part scope consistently across the document.
2. Add proper in-text citation coverage across Chapters 1-7 instead of treating referencing as mainly a Chapter 4 cleanup task.
3. Rewrite Chapters 1 and 2 around precise RQs, a clearly justified research method (Design Science Research is recommended), separate datasets and metrics, and reproducible evaluation.
4. Evidence the Chapter 3 problems and derive traceable requirements.
5. Rebuild Chapter 4 with critical, cited literature.
6. Revise Chapter 5 to show requirement/design/validation traceability and to cross-reference the supporting appendices.
7. Rebuild Chapter 6 so that assistant evaluation and churn evaluation are both explicit, reproducible, and proportionate to the claims being made.
8. Revise Chapter 7 so that conclusions match only the evidence actually demonstrated.
9. Document the pre-production user-verification status, security/data flows, and the boundary between the assistant and churn POC.

## Bottom line

This thesis has a promising and quite original practical direction. The next revision should not add more implementation detail. It should convert the current project report into a disciplined Master-level research document with narrower claims, stronger evidence, and cleaner traceability from problem to method to result.
