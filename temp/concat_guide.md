# Guide to the Preface, Abstract, and Reading Map

## Purpose of this part

This opening material tells the reader what the thesis is about before the technical chapters begin. The thesis proposes two connected AI systems for a Finnish telecom operator's prepaid mobile business:

1. An AI customer assistant that answers subscriber questions.
2. A churn-prevention service that identifies prepaid subscribers likely to stop using the service and prepares retention offers.

The central practical problem is that prepaid customers often need help with balances, packages, top-ups, eSIMs, validity, and recurring payments. Existing help pages and call centres can answer these questions, but they can be slow, costly, and difficult to use. Prepaid customers also have no contract, so they can leave silently by reducing use and stopping refills.

## What the preface contributes

The preface gives the project motivation in plain language. The author wanted to determine whether AI and machine learning could make answers faster for prepaid subscribers. It also identifies the key engineering challenge: an assistant must be reliable and must not invent facts. This is why the thesis combines two mechanisms:

- **Retrieval-Augmented Generation (RAG):** finds relevant, current operator content before the language model writes an answer.
- **Tool-calling:** lets the language model request live information from backend APIs when an answer depends on a particular customer account.

The preface gives a simple end-to-end example: a subscriber asks in Finnish or English, the system identifies the required data source, searches the knowledge base or calls an API, and answers in the subscriber's language. This is important because it shows the thesis is not only about an abstract model; it is about an implemented service workflow.

## The abstract explained

The abstract is the compact version of the complete thesis. It states the problem, method, technology, results, and conclusion.

### The primary system: AI self-service assistant

The assistant is designed to distinguish between two classes of questions:

- **General questions** such as available top-up plans or how eSIM works. These are answered from a searchable knowledge base built from the operator's web content.
- **Account-specific questions** such as the remaining balance or active package. These require real-time backend API calls, because the answer depends on the individual subscriber.

An LLM, Google Gemini by default, acts as the decision-maker. It chooses whether to use the RAG search tool or a backend data tool. The assistant was implemented with Python, FastAPI, LangChain, LangGraph, ChromaDB, and Google Gemini, and supports Finnish and English.

### The secondary system: churn prevention

The churn-prevention service uses subscriber behaviour as an early warning signal. It checks refill frequency and voice, data, and SMS usage. When these measures decline beyond configured thresholds, the service calculates a churn-risk score using machine learning and creates a suitable retention campaign. In other words, the assistant reacts to a customer question today, while the churn service tries to prevent customer loss tomorrow.

The abstract says the churn component is a separate service built with Python, FastAPI, and APScheduler. It periodically examines data and can trigger campaigns when declining behaviour is detected.

## Claimed results in the abstract

The thesis reports that tests demonstrated correct query routing, relevant RAG retrieval, correct API controls such as authentication and rate limiting, and proper handling of conversation memory. For the churn service, simulated declines in usage were detected and targeted campaigns were generated. These are implementation and evaluation claims that Chapter 6 later explains in detail.

The overall conclusion is that combining RAG with tool-calling agents is a suitable architecture for a telecom customer assistant: RAG grounds answers in operator information, and tools provide live personalized data. The churn system extends the business value by enabling proactive retention.

## How to read the remaining thesis

The document follows a familiar applied-research sequence:

- **Chapter 1** defines the business problem, objectives, research questions, scope, and technology stack.
- **Chapter 2** explains the constructive research method used to build and evaluate a working solution.
- **Chapter 3** describes the case company's existing support channels, available APIs, and gaps.
- **Chapter 4** introduces the technical and business concepts needed to understand the solution.
- **Chapter 5** specifies the implementation architecture and design decisions.
- **Chapter 6** reports functional tests and RAG/agent evaluation results.
- **Chapter 7** interprets the results, acknowledges limitations, and proposes future work.

## Abbreviations worth knowing first

A beginner should keep these terms in mind while reading:

- **LLM:** a Large Language Model, such as Gemini, which understands and generates text.
- **RAG:** Retrieval-Augmented Generation; adding retrieved source material to an LLM prompt so answers are based on relevant evidence.
- **API:** an interface through which one software component requests structured data or an action from another.
- **MSISDN:** the subscriber's phone number in telecom systems.
- **PUK:** a code associated with a SIM that is used here for identity verification before disclosing account data.
- **Churn:** a customer becoming inactive or leaving for another provider.
- **XGBoost:** a machine-learning method used to calculate churn probability.
- **JWT:** a signed token used for authentication.
- **UAT:** a user-acceptance-test environment, distinct from live production use.

## Key takeaway

The opening material frames the thesis as a practical, two-part telecom AI solution. Its success depends on matching the correct data source to the question, protecting personal account information, and using subscriber behaviour to act before churn occurs.


---

# Chapter 1 Guide: Introduction

## What this chapter does

Chapter 1 establishes the business need, the two research questions, and the boundaries of the work. It explains why prepaid telecom support is a suitable use case for AI and why the thesis builds both a customer assistant and a churn-prevention solution.

## The business problem

Prepaid subscribers regularly ask about subscription validity, balance, data use, campaigns, top-ups, eSIM activation, cancellation, and general product information. Although many of these questions are repetitive and relatively simple, answering them through call centres is costly and may force customers to wait.

The case operator already provides a web portal and FAQ pages, but these are menu-driven. A customer must know where to look and work through pages or forms. The chapter argues that this is weaker than a conversational interface where a customer can ask a plain-language question directly.

The chapter also introduces a second business problem: **churn**. Prepaid users have no long-term contract, so they can gradually stop refilling or using the service without a formal cancellation. The operator therefore needs to spot declining behaviour before a subscriber is effectively lost.

## The two proposed solutions

### 1. AI-powered customer assistant

The assistant is intended to choose between two data-access approaches.

- **RAG for general information.** For a question such as "What top-up plans are available?", the system searches a curated knowledge base containing operator content and uses the retrieved material to compose the answer.
- **Tool calls for personal, current information.** For a question such as "What is my account balance?", the system calls a backend API. This matters because balance and active packages are specific to one account and can change at any time.

An LLM interprets the customer message and determines which tool is needed. The contribution is therefore not merely a chatbot response; it is an orchestrated system that connects language understanding with live business data.

### 2. Churn-prevention system

The second solution uses refill history and usage of voice, data, and SMS to identify patterns that may indicate churn. If behaviour declines, the system calculates a risk score and prepares a retention campaign. A retention campaign could offer a relevant discount, balance credit, or other incentive before the customer leaves.

## Research questions

The thesis asks two practical design questions:

1. How can a telecom customer assistant use RAG and tool-calling agents to give accurate, real-time, context-aware responses to prepaid subscribers?
2. How can refill and usage data be used to predict churn risk and generate retention campaigns automatically?

These questions set the standard for the rest of the thesis. Chapters 3 and 4 provide the problem analysis and theory; Chapter 5 designs the systems; Chapter 6 tests them; Chapter 7 judges how well they answer these questions.

## Scope: what the thesis includes

For the customer assistant, the scope covers analysing existing service channels, reviewing relevant theory, implementing the architecture, testing it, and packaging it in Docker. For churn prevention, it covers the architecture, subscriber-data model, risk-scoring logic, and campaign-generation logic.

The implementation stack is Python, FastAPI, LangChain, LangGraph, ChromaDB, Gemini, Playwright, and Docker. The churn component additionally uses APScheduler for recurring jobs and XGBoost for risk scoring.

## Out of scope

The thesis does **not** claim a full production rollout. It excludes deployment to live customers, A/B testing or customer-satisfaction measurement, call-centre/CRM integration, and embedding the assistant in the existing web or mobile apps. These limits are essential when interpreting later positive test results: the systems were built and evaluated in a controlled development or UAT setting, not proven at commercial scale.

## Why the architecture matters

The core design premise is that language models alone should not be trusted as the source of telecom facts. Product information changes and account data is private and dynamic. RAG supplies grounded domain content, while backend tools supply live personalized data. The LLM is used for understanding, routing, synthesis, and multilingual conversation rather than as an unverified database.

## Key takeaway

Chapter 1 frames the thesis as applied systems work. The goal is to reduce repetitive support effort and improve self-service while keeping answers relevant, current, and secure. The customer assistant handles immediate information needs; the churn system uses historical behaviour to make proactive retention possible.


---

# Chapter 2 Guide: Method and Material

## What this chapter does

Chapter 2 explains how the work was carried out and how the author assessed whether the results can be trusted. Its purpose is to show that the thesis is a structured research-and-development effort, not simply a programming project.

## Research approach: constructive research

The thesis uses **constructive research**. In this approach, the researcher creates a concrete solution to a real problem and evaluates whether it works. The output is an artefact, here two software systems, plus evidence about their usefulness.

This method suits the thesis because the research questions ask how to design and implement a telecom assistant and churn solution. A purely theoretical review would not answer those questions. The systems must be built, tested, and assessed against requirements.

## The four research phases

The work progresses through four connected phases. Each phase supplies inputs for the next one.

### Phase 1: Current-state analysis

The author investigates the existing prepaid service landscape. This includes customer channels such as help pages and refill tools, available backend APIs, and gaps in the current experience. The aim is to make sure the proposed solution responds to real operational problems rather than imagined ones. These findings appear in Chapter 3.

### Phase 2: Theoretical framework

The author studies LLMs, RAG, tool-calling agents, relevant frameworks, and related technical literature. This phase supports design decisions. For example, understanding hallucinations explains why RAG is required; understanding agent tool calls explains how a language model can access account APIs. The resulting concepts appear in Chapter 4.

### Phase 3: Design and implementation

Requirements are defined, technologies are selected, components are built, and integrations are created. The stated intent is that choices in Chapter 5 should follow from the problems in Chapter 3 and the theory in Chapter 4. This traceability is important: it shows why a component exists, such as Playwright for JavaScript-heavy web pages or SQLite for low-overhead session memory.

### Phase 4: Testing and evaluation

The systems are evaluated with automated tests and manual inspection. Pytest covers API behaviour, authentication, history, and content hashing. RAGAS evaluates retrieval and generation quality, and TruLens evaluates the complete agent flow. The results are presented in Chapter 6.

## Reliability explained

**Reliability** asks whether a process or measurement produces consistent results. The thesis supports reliability through repeatable engineering practices:

- publicly available, documented frameworks such as FastAPI, LangChain, LangGraph, and ChromaDB;
- Git version control for source changes;
- automated pytest tests that can be rerun;
- public web pages as knowledge-base sources; and
- deterministic SHA-256 hashes to detect whether source content changed.

For a beginner, deterministic means that the same input produces the same result. Hashing identical content should always yield the same 64-character value, allowing the ingestion process to skip unchanged documents.

## Validity explained

**Validity** asks whether the test actually measures what it claims to measure. The thesis ties each evaluation to a concrete intended behaviour. API tests check status codes and input validation; authentication tests check accepted and rejected credentials; memory tests check storage and isolation; retrieval evaluation checks whether relevant chunks are found; and routing tests check whether the correct tool is selected.

This is stronger than claiming the system "seems good," because it identifies observable criteria. However, the chapter also openly limits the claim: UAT testing is not production testing, manual RAG evaluation can be subjective, and the system was not tested with real customers.

## Use of AI tools

The author reports using GitHub Copilot, Claude, GPT, and Gemini for architectural brainstorming and coding assistance. The chapter states that the author reviewed and edited the output and retains responsibility for the final text and code. This is a transparency statement, not an evaluation of AI-tool quality.

## Key takeaway

Chapter 2 gives the thesis its evidence structure. The author first identifies an operational problem, studies appropriate concepts, constructs a working solution, and tests it with defined criteria. Its credibility depends on repeatability of the automated checks and on recognizing that a limited UAT evaluation cannot demonstrate full real-world effectiveness.


---

# Chapter 3 Guide: Current State Analysis

## What this chapter does

Chapter 3 describes the business setting before any new AI system is introduced. It identifies what the Finnish telecom operator already offers to prepaid customers, what data interfaces already exist, and which customer-service problems remain unsolved. These findings justify the design choices in Chapter 5.

## The prepaid context

A prepaid subscriber pays for balance or packages in advance and then uses voice, SMS, and data services. The operator offers starter kits through stores and partners, and eSIM through its online shop. The product portfolio includes unlimited packages, data packages, balance top-ups, Internet of Things packages, voice/SMS packages, and recurring payment options.

This setting matters because prepaid accounts need frequent, practical support. Customers need to understand and manage a varied set of products, but their relationship with the operator is less durable than a contract-based postpaid relationship.

## Existing customer-service channels

The chapter identifies three existing forms of self-service or assistance.

### Online refill channel and FAQ pages

Customers can check balances and packages, purchase top-ups, and manage recurring payments online. The help content covers starter-kit purchases, top-ups, eSIM, recurring payments, and roaming. The weakness is discoverability: information is distributed across conventional web pages and accordion sections. A customer must navigate to the right page and locate the right answer.

### Call centre

Human customer-service agents can resolve any issue, but this channel is expensive for the operator and slow for customers. The chapter treats routine queries as an automation opportunity, not an argument to remove human support for complex cases.

### SMS commands

SMS short codes offer a quick way to check balance, packages, and expiry or activate a package. However, the method requires customers to know the correct commands. It is fast but not conversational or discoverable.

## Existing backend APIs

An important finding is that the required account data already exists in machine-readable form. The system does not need to recreate core telecom backends. Available REST APIs provide:

- validation that a phone number is a valid prepaid MSISDN;
- subscription details such as balance, expiry, packages, usage, and recurring payments;
- product catalogue and campaign information;
- refill history; and
- SIM information, including PUK codes used for verification.

This availability directly enables the later tool-calling approach. APIs are best for live, structured account information, while web/FAQ content is best suited to a searchable RAG knowledge base.

## Identified problems and their meaning

### No conversational interface

Customers cannot express their need naturally. They must navigate menus and forms. The proposed LLM assistant is intended to convert a free-text question into an answer or an API request.

### Fragmented knowledge

A complete answer may require several web pages. A RAG knowledge base can retrieve relevant content from multiple sources and let the LLM synthesize it.

### No personalized self-service

FAQ content is generic and cannot reveal a specific balance or package. Secure API calls are needed for personal answers.

### High cost of routine calls

Routine questions consume agent time even though their answers may already be present in APIs or help content. Automation could reduce that workload.

### Language gaps

Some information is only available in Finnish. The solution must ingest content in both Finnish and English and generate a response in the customer's language.

### No proactive churn prevention

The operator can observe churn after it happens, but lacks an automated way to look for declining refill or usage beforehand. This drives the separate data-collection, scoring, and campaign components.

## How it connects to later chapters

Every major architecture component answers one of these problems: RAG consolidates fragmented guidance; agent tools reach existing APIs; PUK verification protects personalized data; multilingual ingestion and prompting address language needs; and churn scoring turns behavioural data into an early intervention workflow.

## Key takeaway

Chapter 3 is the business justification for the thesis. It shows that the operator has valuable content and APIs but lacks a single conversational, multilingual, personalized, and proactive service layer. The proposed systems are designed to connect those existing resources into a better prepaid experience.


---

# Chapter 4 Guide: Theoretical Background

## What this chapter does

Chapter 4 provides the vocabulary and rationale for the thesis architecture. It explains how language models work at a high level, why they need grounding, how agents use software tools, why particular libraries were chosen, and how these ideas apply to telecom customer service and churn prevention.

## Large Language Models (LLMs)

An **LLM** is an AI model trained on large volumes of text. Given a prompt, it predicts and generates text one token at a time. A token is a small unit of text, often a word or part of a word. LLMs can understand requests, summarize, translate, extract information, and create responses in several languages.

The thesis uses an LLM because a customer assistant needs broad language abilities: it must understand varied wording in Finnish and English, identify what a customer needs, and turn raw system data into a clear response. An LLM is not treated as the authoritative source of account or product facts; that distinction becomes crucial in the RAG and tool-calling sections.

### Transformer architecture

Modern LLMs use the **Transformer** architecture. Its central mechanism, self-attention, lets the model assess how each word relates to other words in the same input. This is more effective than older recurrent approaches for relationships between distant parts of a sentence and can process text in parallel during training.

The chapter mentions encoder and decoder roles, while noting that many current LLMs are decoder-only: they generate the next token from previous context. The important practical lesson is that an LLM produces likely text from context. It does not query a guaranteed fact database by itself.

### Providers and portability

The thesis discusses Google Gemini, OpenAI GPT, Anthropic Claude, and Meta LLaMA. Gemini 2.5 Flash is the default because it targets high-volume use with speed and cost in mind. The implementation hides provider choice behind a factory function and an environment variable. This design reduces dependence on one vendor and supports future substitution for cost, privacy, performance, or availability reasons.

### Prompt engineering

A prompt is the instruction and context sent to an LLM. The chapter distinguishes system prompts, user prompts, few-shot examples, and chain-of-thought-style instructions. The system prompt is especially important here: it tells the assistant to rely on supplied data, be concise, remain in scope, answer in the customer language, preserve privacy, and avoid proactively requesting a phone number. The prompt therefore acts as a behavioural policy, although it is not a replacement for technical authorization controls.

## Retrieval-Augmented Generation (RAG)

### The problem: hallucination

A hallucination is a plausible-sounding but incorrect model response. In telecom, this could mean citing an outdated price, invented product condition, or wrong process. LLM training data can be old, incomplete for a particular operator, or inappropriate for a specific current account.

**RAG** reduces this risk by retrieving relevant operator material when a question arrives and supplying it to the LLM as context. The desired answer is therefore grounded in current domain sources rather than only the model's general training.

### The three RAG stages

1. **Indexing:** collect source documents, split them into chunks, turn each chunk into an embedding vector, and store vectors in a vector database.
2. **Retrieval:** embed the incoming question and find stored chunks with similar meaning.
3. **Generation:** add the retrieved chunks and question to the LLM context so it can produce a useful answer.

An **embedding** is a vector of numbers representing semantic meaning. Texts with similar meaning are placed near each other in vector space even when they do not use exactly the same keywords. Similarity is commonly measured with cosine similarity: values closer to 1 indicate vectors pointing in a similar direction.

The thesis sets $k = 4$, meaning the retriever returns four candidate chunks. Increasing $k$ may improve coverage but can add distracting material; decreasing it can improve focus but omit relevant facts.

### Embeddings, vector storage, and chunking

The implementation uses Google's multilingual `text-embedding-004` model, which creates 768-dimensional vectors. Multilingual support is necessary because the knowledge base contains Finnish and English content.

ChromaDB stores these vectors and supports similarity search. It was selected for simple local, persistent deployment and LangChain integration. The chapter recognizes that it has single-machine scaling limits and lacks built-in replication, which makes it appropriate for this thesis prototype rather than necessarily for a large production installation.

Source documents are split with a recursive splitter using 500-character chunks and 100-character overlap. Recursive splitting favours natural boundaries such as paragraphs and sentences. Overlap keeps context that would otherwise be cut at a boundary. Chunk size is a quality trade-off: large chunks contain more context but more irrelevant text; small chunks can be precise but incomplete.

## Tool-calling agents

RAG answers general questions from stored sources. A balance, current package, or refill history cannot be safely pre-indexed because it is private and changes over time. **Tool calling** lets the LLM request a defined function with structured arguments. The application runs the function, returns its result to the LLM, and the model writes a customer-friendly answer.

The thesis uses the **ReAct** pattern: Think, Act, Observe, then repeat or respond. For example, the model can recognize an account question, select a subscription-details tool, observe the API result, and then formulate a response. This replaces a large manually programmed intent-routing matrix with flexible runtime selection.

Agents are most appropriate when several tools exist, tool choice depends on the wording and task, a query may require multiple sources, and conversation is open-ended. The telecom assistant meets all four conditions.

## Frameworks and libraries

- **LangChain** supplies common interfaces for models, prompts, document loaders, chunkers, vector stores, tools, and memory.
- **LangGraph** coordinates stateful multi-step workflows. Its `create_react_agent` helper manages the agent loop instead of requiring the author to implement parsing, tool execution, and stopping logic from scratch.
- **ChromaDB** is the persistent local vector store used for semantic retrieval.
- **FastAPI** exposes the application through typed HTTP endpoints, validates inputs, supports middleware, asynchronous I/O, and auto-generated API documentation.

These libraries reduce implementation effort, but they do not eliminate design responsibilities such as tool authorization, prompt rules, data quality, or scalability planning.

## Conversational AI in telecom

The chapter compares three generations of chatbots:

- **Rule-based systems** follow keywords and scripted decision trees. They are predictable but brittle.
- **Intent-based systems** use NLU to map questions to predefined intents. They handle wording variation but require continuous intent and training-data maintenance.
- **LLM-powered assistants** understand flexible language, maintain context, work across languages, and synthesize several sources, but have costs, latency, and less deterministic behaviour.

Telecom is a strong fit because it has high volumes of repetitive questions, structured backend data, rapidly changing offers, and multilingual customers. RAG handles changing product information; tools handle account and service data.

## Building the RAG knowledge base

The operator's web pages are single-page applications (SPAs), so simple HTTP downloading may only return an empty HTML shell. The thesis uses **Playwright**, a headless browser, to run JavaScript, wait for rendered content, interact with accordions, and extract the final DOM. BeautifulSoup then removes non-content elements and turns the HTML into clean text for indexing. This is an important practical detail: poor source extraction would undermine every later retrieval result.

## Churn prevention theory

**Churn** means a subscriber stops using the service. Prepaid churn is difficult because customers can leave without a cancellation; falling usage or delayed top-ups may be the only signal. The chapter frames prediction as binary classification: from past behavioural features, a model predicts probability of churn within a specified period.

Useful features include refill frequency and amount, time since last refill, voice/data/SMS trends, account age, package engagement, and recharge channel. The thesis selects **XGBoost**, a gradient-boosted tree method, because it handles non-linear patterns and imbalanced data well. It returns a score from 0 to 1, grouped as low ($0.0$-$0.3$), medium ($0.3$-$0.6$), and high ($0.6$-$1.0$) risk.

Prediction only creates value if the operator acts. The chapter links risk levels to proactive interventions such as free credit, discounted recharges, loyalty rewards, or recommendations tailored to use patterns. Timing, personalization, and delivery channel influence whether a retention campaign succeeds.

## Key takeaway

Chapter 4 establishes the logic behind the whole solution: LLMs provide flexible language interaction; RAG grounds general answers in operator content; tool calling reaches live private data; and churn models translate changes in subscriber behaviour into timely retention actions. The technologies are complementary rather than interchangeable.


---

# Chapter 5 Guide: System Design and Architecture

## What this chapter does

Chapter 5 turns the earlier problem analysis and theory into an implementable design. It explains requirements, component choices, end-to-end data flow, security controls, deployment, and the separate churn-prevention service. This is the most implementation-focused chapter.

## Requirements

The functional requirements describe what the systems must do. The customer assistant must answer general questions through RAG, answer personal questions through APIs, select tools with an LLM agent, support Finnish and English, remember a session, verify identity through PUK, validate numbers, and show products/offers. The churn solution must monitor trends, score churn risk, and generate campaigns.

The non-functional requirements concern qualities rather than user features: API-key and JWT authentication, 60 requests per minute rate limiting, Docker deployment, a configurable model provider, incremental knowledge-base updates, input validation, structured logs, and configurable CORS. These requirements make clear that the thesis aims for a service architecture rather than an isolated LLM demo.

## Key design decisions

The system uses LLM-driven routing instead of a separate intent-classification model. Tools have names and descriptions, and the LLM determines which to invoke. This reduces the need to manually define every possible user intent.

The knowledge-base search is itself a tool. As a result, the agent has one consistent mechanism for deciding among general information, product data, and account-specific APIs. SQLite is used for low-overhead conversation memory, ChromaDB for local semantic search, and Playwright because the source sites are JavaScript-rendered SPAs. SHA-256 content hashes enable efficient updates by skipping unchanged sources.

## Layered assistant architecture

The assistant has five layers:

1. **Presentation:** a browser chat interface.
2. **API:** a FastAPI server that receives requests and enforces service controls.
3. **Agent:** a LangGraph ReAct agent that interprets the conversation and coordinates actions.
4. **Tools:** Python functions that expose specific operations.
5. **External systems:** the LLM provider, ChromaDB, and telecom backend APIs.

A typical request flows as follows: the browser sends a message; the API validates it and loads the session history; the agent decides whether to use a tool; one or more tools return data; the LLM writes an answer from that data; the message and response are stored; then the browser receives the response.

Separating layers makes the system easier to test and change. The UI need not know API details, the agent need not embed HTTP logic, and individual tools can handle their own failures.

## RAG knowledge-base pipeline

### Sources and scraping

The `sources.yaml` configuration identifies relevant prepaid pages and product-catalogue APIs. For each website, Playwright starts headless Chromium, selects a locale, waits for network activity to settle, waits for required content, expands accordions when necessary, and extracts rendered HTML. BeautifulSoup then cleans it into usable text. Sources can be fetched in parallel, limited to two concurrent pages.

### Ingestion

The command-line ingestion workflow can load all sources, only web or API sources, clear and rebuild the store, or force re-ingestion. Conceptually, it performs the following sequence:

1. Load source definitions and language variants.
2. Fetch each page or API result.
3. Calculate a SHA-256 content hash.
4. Skip an unchanged source unless a force option was given.
5. Split new or changed text into chunks.
6. Create embeddings for chunks.
7. Store chunks and source metadata in ChromaDB.

The stored metadata includes items such as source name, type, and URL. This supports traceability and possible later filtering.

### Chunking and embeddings

The system uses 500-character chunks with 100-character overlap. The recursive splitter tries paragraph, sentence, and word boundaries before using a less natural split. Google `text-embedding-004` creates the vectors. The design balances retrieval precision, enough local context, and multilingual support.

## The tool-calling agent

The agent exposes five tools:

- `validate_number` checks whether an MSISDN is a valid prepaid number.
- `get_subscription_details` obtains balance, usage, packages, expiry, and recurring-payment information.
- `get_products` retrieves product and offer information.
- `get_refill_history` returns top-up history.
- `search_knowledge_base` retrieves general source content from ChromaDB.

Each API tool handles connection errors, timeouts, and HTTP failures internally. This design prevents one unavailable backend from crashing the entire agent workflow and lets the LLM present a meaningful failure response.

The agent is created by passing the selected LLM, tools, and system prompt to LangGraph's `create_react_agent`. The prompt confines the assistant to prepaid scope, asks for concise data-grounded answers, preserves privacy, uses the customer language, and prohibits unnecessary requests for phone numbers. The agent is an orchestrator: it does not replace the backend systems or the authorization layer.

## API server and security

The FastAPI server exposes unauthenticated health and UI routes plus authenticated chat, verification, and clear-history routes. Its main protections are:

- **API keys:** configured keys are compared with `hmac.compare_digest`, which avoids timing leaks in ordinary string comparison.
- **JWT validation:** the server verifies the HMAC-SHA256 signature and expiry of a token.
- **PUK verification:** personal account information requires the customer to verify a PUK against the SIM details API. Verified numbers are cached for the session.
- **Rate limiting:** a sliding window stores recent per-user request timestamps. Excess traffic returns HTTP 429.
- **Input validation and CORS:** FastAPI/Pydantic reject malformed input, and permitted browser origins are configured through `ALLOWED_ORIGINS`.

Security should be viewed as layered. A prompt request to protect privacy is helpful, but account authorization is actually enforced by API authentication and PUK verification.

## Conversation, provider, UI, and language design

Conversation history is stored with LangChain's SQLite-backed `SQLChatMessageHistory`. Each session uses its own history so one user's context is not mixed with another's. SQLite keeps deployment simple but later limits horizontal scaling.

The `get_llm()` factory selects Google Gemini, OpenAI, Anthropic, or Ollama based on `LLM_PROVIDER`; the default is Gemini 2.5 Flash. A low temperature of 0.2 favours more consistent responses. The simple HTML UI has a chat area, input, conditional phone-number input, and PUK dialog that communicate through the `/chat` endpoint.

Finnish and English support appears at four layers: pages are ingested in both languages, the prompt tells the model to match the customer, APIs accept a language parameter, and the model automatically recognizes the message language.

## Docker deployment

Docker packages the Python application, dependencies, and Chromium needed by Playwright. Docker Compose supplies ports, environment variables, persistent volumes, and health checks. Containerization is intended to make the same application run reliably across environments, though it does not by itself solve production scaling or operational monitoring.

## Churn-prevention architecture

The churn service is a separate Python/FastAPI application with four main responsibilities:

1. A daily APScheduler job collects refill and usage data from MariaDB.
2. Feature engineering transforms raw records into signals such as days since refill and change in usage.
3. A rule-trigger plus XGBoost scoring engine produces a churn probability and risk level.
4. A campaign engine chooses an offer, and an API exposes the campaign for downstream SMS, push, or app delivery.

The listed features include last-refill age, refill-amount change over 30 days, voice/data/SMS trends over two consecutive 14-day periods, and account age. Rules flag a more than 30% drop in refills, more than 40% fall in usage, or no refill for 30 days. The XGBoost classifier scores flagged users. Scores above 0.3 trigger campaigns: soft offers for medium risk, immediate campaigns for high risk, with choices adapted to the usage profile and customer state.

## Key takeaway

Chapter 5 describes a modular, data-grounded architecture. The assistant combines conversational LLM behaviour with controlled access to knowledge and live APIs, while the churn service turns behavioural trends into operational campaign decisions. It is designed for practical deployment, but the later discussion recognizes its single-server and prototype limitations.


---

# Chapter 6 Guide: Results and Evaluation

## What this chapter does

Chapter 6 reports whether the implemented systems behaved as intended. It combines conventional automated software tests with AI-focused evaluation: RAGAS assesses the retrieval-augmented question-answering pipeline, and TruLens observes the broader agent workflow. The chapter provides evidence for implementation-level claims, while Chapter 7 later explains the limits of that evidence.

## Automated test suite

The thesis uses pytest for automated tests. In API tests, the LLM agent is mocked. Mocking means replacing the real external model with a controlled substitute so that a test can isolate the API server's behaviour. This is good testing practice: a network problem or varying LLM answer should not make an authentication or input-validation test unreliable.

The tests cover:

- HTTP endpoint behaviour and validation;
- API-key authentication;
- PUK identity-verification flow for account requests;
- rate limiting;
- SQLite conversation memory; and
- SHA-256 content hashing used by knowledge-base ingestion.

## API and security results

All listed test cases passed. The health endpoint returns HTTP 200. A valid chat request receives a response. Empty messages and messages over 2,000 characters receive HTTP 422 validation errors. When authentication is enabled, missing or invalid keys receive HTTP 401 while a valid key is accepted.

The account-query flow is also tested. If a customer asks for balance without a number, the system indicates that an MSISDN is needed. If the number has not been verified, it requests PUK verification. Correct PUK produces a positive verification result and incorrect PUK a negative result. Clearing history succeeds, and exceeding the configured request limit returns HTTP 429.

For a beginner, these codes have distinct meanings:

- **200:** successful request.
- **401:** the client has not supplied valid authentication.
- **422:** the input exists but fails declared validation rules.
- **429:** the client is sending too many requests.

These tests show that the controlled API behaviour was implemented. They do not by themselves prove security against every attack or real production traffic condition.

## Conversation memory and content hashing

Memory tests show that messages are returned in order, deletion clears them, and separate sessions do not share history. This validates the expected session-local conversation experience.

Hashing tests show that identical content yields identical hashes, different content yields different hashes, and hashes have the expected 64-character SHA-256 form. These findings support the incremental ingestion design: the system can reliably detect content that has not changed and avoid repeating expensive embedding work.

## Conversation demonstrations

The chapter provides two representative conversations.

For a general eSIM question, the agent invokes only `search_knowledge_base`. No personal API data is needed. This demonstrates the intended RAG route.

For a Finnish balance question, the API asks for a phone number, requests PUK verification, validates that code, resends the original query to the agent, calls `get_subscription_details`, and returns the live result in Finnish. This demonstrates an account-specific route and shows that language selection, identity verification, tool calling, and response formatting work together.

These examples are illustrative. They show expected workflows but are not a substitute for broad user testing.

## RAGAS evaluation

**RAGAS** is a framework that scores RAG systems using LLM-based evaluation. The thesis runs questions through the full RAG path: retrieve four chunks from ChromaDB, have the agent answer, then score each question-answer pair with Gemini as judge.

The reported average scores are:

| Metric | Score | Interpretation |
|---|---:|---|
| Faithfulness | 0.9446 | The answer is well supported by retrieved context. |
| Answer relevancy | 0.7966 | The response generally addresses the question. |
| Context precision | 0.8556 | Relevant documents tend to be ranked highly. |
| Context recall | 0.9500 | Retrieved context covers most needed ground-truth information. |

A score closer to 1 is better. The strongest outcome is high context recall: the required information was usually present in what was retrieved. High faithfulness suggests the answer generally stayed grounded in that retrieved evidence. Precision is somewhat lower, indicating that not every high-ranked chunk is equally relevant. Relevancy is lowest at about 0.80; the author explains that answers sometimes added helpful next steps or related links, which the metric can treat as outside the exact question.

The per-question table shows mixed detail behind averages. The eSIM activation question has faithfulness of 0.80 and recall of 0.75, showing room to improve coverage in particular cases. Recharge and recurring-payment questions have context precision of about 0.64, suggesting the ranking of retrieved chunks could be better even when recall is high.

## TruLens agent evaluation

**TruLens** evaluates and traces the full agent, including general knowledge searches, API calls, multi-tool paths, and out-of-scope rejection. OpenTelemetry tracing records tool choices, tool results, and final responses. Gemini, accessed through LiteLLM, acts as the judge.

The thesis reports answer relevance of 1.0, latency of roughly 3.22 seconds, and cost of roughly $0.001368 per evaluated interaction. The result supports the claim that the tested agent responses were relevant across the supplied dataset. The mention of an out-of-scope weather question is useful: it was rejected without a tool call and was therefore the fastest route.

## Overall result claim

The author concludes that functional requirements FR1-FR8 and non-functional requirements NFR1-NFR8 were met, and that churn requirements FR9-FR12 were implemented through collection, scoring, and campaign modules. The reported evidence supports correct component-level behaviour and strong RAG scores in the selected test set.

## How to interpret the results carefully

The scores are promising but should not be read as proof of production readiness. The evaluation dataset is small, the LLM judges are themselves models, and the customer experience was not evaluated with a broad real-user sample. The tests validate the system as implemented in its controlled environment; they do not establish long-term reliability, campaign effectiveness, or customer satisfaction in live telecom operations.

## Key takeaway

Chapter 6 shows that the prototype works along its intended paths: it validates inputs, protects account flows, retains session history, avoids unnecessary re-ingestion, retrieves largely relevant context, and routes agent queries appropriately. Its quantitative RAGAS and TruLens findings are encouraging, but they are bounded by the scope of the test data and environment.


---

# Chapter 7 Guide: Discussion and Conclusions

## What this chapter does

Chapter 7 interprets the implementation and evaluation results. It asks whether the original objectives and research questions were met, records limitations, identifies practical next steps, and states the final contribution. This chapter is where the thesis moves from "the tests passed" to "what the work means and what remains uncertain."

## Did the system meet its objectives?

The author concludes that both objectives were met.

For the customer assistant, the claimed achievements are:

- general prepaid questions are answered using the RAG knowledge base;
- account-specific questions use live backend APIs;
- the agent routes tested questions to appropriate tools;
- Finnish and English are supported;
- SQLite maintains conversation context; and
- API keys, JWTs, PUK verification, and rate limiting provide security controls.

For churn prevention, the system collects refill and usage data, derives behavioural features, calculates risk scores, and maps scores to retention campaigns. The conclusion is based on implementation and controlled evaluation, not evidence that customers in production received campaigns or stayed longer as a result.

## Answer to research question 1: customer assistant design

The thesis gives a concrete architecture recipe for an accurate, real-time, context-aware prepaid assistant:

1. Use a layered design with a chat UI, secured API server, ReAct agent, defined tools, and external data systems.
2. Scrape dynamic web content with a headless browser, split it into overlapping chunks, embed it with a multilingual model, and store it in a vector database for RAG.
3. Wrap each backend API as a tool with clear descriptions and internal timeout/error handling.
4. Let a ReAct agent choose and sequence tools rather than forcing every request into a manually maintained intent classification.
5. Store target-language source content, direct the model to use the customer language, and pass a language value to backend APIs.
6. Enforce authentication, authorization/identity verification, and rate limiting around account access.

The key point is that accuracy comes from architecture. The LLM is not expected to know all telecom facts on its own. It is supplied relevant knowledge or live data as needed.

## Answer to research question 2: churn prevention

The proposed churn workflow is:

1. Periodically gather refill and usage records.
2. Turn them into behavioural features, such as refill-frequency change, days since activity, and voice/data/SMS changes.
3. Use an XGBoost classifier to produce a churn probability between 0 and 1.
4. Map the score to low, medium, or high risk.
5. Select an intervention suited to risk and subscriber profile.

The chapter offers examples: medium-risk customers may receive free balance credit; high-risk customers may receive discounted recharge offers. The goal is to act at the first meaningful decline rather than after a customer has become inactive.

## Important limitations

The discussion is strongest when it distinguishes a promising prototype from a production-proven service.

### No end-user testing

The system was assessed by the author and colleagues, not a broad group of actual prepaid customers. Real customers may use unexpected wording, different languages, ambiguous requests, or have usability needs that test scenarios miss. A user study is therefore necessary before judging satisfaction or adoption.

### Small evaluation dataset

RAGAS and TruLens scores were calculated over a limited set of questions. A production evaluation should include hundreds of questions across all product topics, both languages, ambiguous phrasing, edge cases, and adversarial attempts. Strong averages from a small curated set may not generalize.

### Single-server state

The in-memory rate limiter and SQLite memory work well for one server but do not share state across a scaled-out deployment. Multiple servers could independently accept more requests or lose conversation context. Redis for shared rate limiting and PostgreSQL for shared conversation data are proposed remedies. A managed vector database could also replace a local store.

### Dependence on an external model provider

Gemini provides both language and embedding capabilities. If that service is unavailable, the assistant is affected. The system can be reconfigured for another provider but does not automatically fail over. Operational resilience would require a tested fallback strategy.

### Manual knowledge-base updates

Content hashes make updates efficient, but ingestion still has to be triggered manually. A scheduled job or update webhook should initiate re-ingestion when source content changes. Until then, an answer can be grounded in stale rather than current information.

### Response latency

The response is returned only after the full agent workflow completes. A complex multi-tool request can take five to eight seconds. Server-Sent Events or WebSockets could stream the answer as it is generated, improving perceived responsiveness.

### PUK verification state

Verified numbers live only in server memory, so a restart loses them; conversely, they lack time-based expiry while the server stays up. Persisting short-lived verification state in a store such as Redis would improve both usability and security.

### Missing operational analytics

Logs exist, but there is no dashboard for query volume, latency, tool use, errors, or outcome trends. Monitoring is essential for proving operational value and locating failures after deployment.

## Recommended future work

The thesis recommends a real-customer pilot with surveys and satisfaction ratings, a larger evaluation set integrated into CI/CD, scalable state and infrastructure, streamed responses, automated RAG updates, integration with the operator's web/app and call-centre systems, and analytics with tools such as Prometheus and Grafana.

For churn prevention, the most important next experiment is A/B testing. Customers at similar risk should receive alternative offers or a control condition, and the operator should compare subsequent retention. Without this, the service can identify risk and generate campaigns, but cannot demonstrate that the campaigns cause better retention.

## Final contribution

The thesis claims to demonstrate a reusable pattern for enterprise AI applications: combine a language model with retrieval over trusted sources and controlled tools for live data. In the telecom setting, this enables conversational self-service, personal account assistance, multilingual responses, and a route to reducing routine call-centre demand. The separate churn service extends the same data-driven philosophy from reactive support to proactive customer retention.

## Key takeaway

Chapter 7 concludes that the prototype meets its stated technical objectives, but responsibly limits the strength of that conclusion. It is a well-defined, tested foundation for a real service, not yet evidence of large-scale user satisfaction, cost reduction, or churn reduction. Production validation requires users, scale testing, monitoring, automated content refresh, and controlled campaign experiments.


---

# References Guide: How the Sources Support the Thesis

## Purpose of the references chapter

The references list provides the academic and technical basis for the thesis. It contains 22 sources covering the research method, LLMs and Transformers, prompt engineering, RAG, embeddings and vector databases, tool use, chatbot evolution, telecom-specific LLM applications, hallucination, and churn prediction.

For a beginner, references serve two roles:

- They show where central concepts originated or were studied.
- They let a reader verify claims and explore a topic in more depth.

## Source groups and their role

### Research method

The constructive research approach is supported by Kasanen, Lukka, and Siitonen (reference 1). This source underpins the method in Chapter 2: create a practical solution to a real problem and evaluate it.

### LLMs and Transformers

References 2-6 and 14 cover the foundations of large language models and prompt engineering. Brown et al. is the GPT-3 few-shot-learning paper. Vaswani et al.'s "Attention Is All You Need" is the foundational Transformer paper. Bommasani et al. discusses opportunities and risks of foundation models. The Gemini technical report supports the discussion of Gemini, and White et al. addresses prompt patterns. Zhao et al. provides a wider LLM survey.

These sources provide context for why an LLM can interpret flexible language but also needs controls and trustworthy external data.

### RAG, hallucination, embeddings, and vector databases

References 7-10, 15, and 18 provide the main RAG background. Lewis et al. is the landmark RAG paper. Gao et al. surveys RAG approaches. Taipalus discusses vector database systems, while Muennighoff et al. is relevant to embedding-model evaluation. Shuster et al. investigates retrieval augmentation as a way to reduce hallucination, and Ji et al. surveys hallucination in natural-language generation.

These sources support the architectural argument that an LLM alone can generate inaccurate or outdated content, whereas retrieval supplies domain material at answer time.

### Tool-calling agents and chatbots

Toolformer by Schick et al. (reference 11) supports the idea that language models can use external tools. Adamopoulou and Moussiades (reference 12) examines chatbot history, and Braun et al. (reference 13) relates to evaluating NLU systems. Together they give background for moving from rigid chatbot designs toward flexible agent systems.

### Telecom use and churn prevention

Reference 16 surveys LLMs in telecommunications. References 19, 20, and 22 address telecom churn, its determinants, and machine-learning prediction methods. The XGBoost source (reference 21) supports the selected gradient-boosted-tree algorithm.

This group explains why the thesis focuses on prepaid churn and why behavioural data, classification, and targeted interventions are relevant.

## How to use this list while reading

When a chapter introduces a concept, a strong thesis normally links the statement to the relevant sources. For example:

- constructive research should connect to reference 1;
- Transformer claims should connect to reference 3;
- RAG claims should connect to references 7 and 8;
- hallucination claims should connect to references 15 and 18;
- tool-using LLM claims should connect to reference 11; and
- churn/XGBoost claims should connect to references 19-22.

The list itself is broad enough to support the thesis's main technical and business ideas. It contains primarily academic papers and surveys, supplemented by well-known system papers. A reader who wants to reproduce the implementation would still need current official documentation for libraries such as LangChain, LangGraph, ChromaDB, FastAPI, Playwright, RAGAS, and TruLens; those implementation documents are not explicitly listed here.

## Reading priority for beginners

A concise learning path is:

1. Read the Transformer paper title and a plain-language explanation to understand why modern LLMs became possible.
2. Read the RAG paper and hallucination survey to understand why retrieved evidence matters.
3. Read the Toolformer idea to understand why an LLM can call APIs rather than only write text.
4. Read the telecom LLM survey and churn papers to see how these techniques apply to a business setting.
5. Read the XGBoost paper only after the basic churn workflow is clear; it is more algorithmically technical.

## Key takeaway

The bibliography supports the thesis's overall chain of reasoning: constructive research justifies building a practical artefact; LLM and Transformer work explains the conversational layer; RAG and vector research explains grounded answers; tool-use research supports API orchestration; and telecom churn research motivates behavioural scoring and retention campaigns. It is the intellectual map behind the implementation.


---

# Appendices Guide: System Prompt and Project Structure

## Purpose of the appendices

The appendices provide supporting implementation detail that would be too specific for the main narrative. Appendix 1 gives the assistant's system prompt, which defines expected conversational behaviour. Appendix 2 gives high-level file structures for the self-service assistant and the churn-prevention service, showing how the design in Chapter 5 could be organized as code.

## Appendix 1: system prompt

A **system prompt** is a persistent instruction given to the LLM before customer messages. It defines the assistant's role as a helpful prepaid telecom assistant and tells it how to use the question and supplied data.

The prompt's key rules are:

1. **Use provided data; do not guess.** This is the behavioural counterpart to RAG and API tools. The LLM should turn evidence into an answer, rather than inventing account details.
2. **Be concise and clear.** Lists are preferred for multi-item answers, and prices should be converted from cents to euros.
3. **Be proactive.** When a balance is low, the assistant can suggest recharge options.
4. **Remain within prepaid scope.** Questions outside the service domain should be redirected politely.
5. **Match the user's language.** This supports Finnish and English customers without requiring a separate user-facing language selector.
6. **Protect privacy.** One subscriber's data must never be exposed to another.
7. **Do not ask for a phone number.** The assistant may use general tools when no MSISDN is available. Account-specific tools are allowed only when an MSISDN is already supplied in the message.

The seventh rule is particularly important. It minimizes unnecessary collection of personal identifiers and separates general support from account access. In the broader system, server-side logic also requests and verifies PUK for account-specific questions. Prompt rules guide language-model output, while the backend controls actual access.

## What the prompt does and does not guarantee

The prompt expresses desired behaviour. It can steer an LLM toward evidence-based and privacy-conscious responses, but it is not a security boundary on its own. A robust implementation still needs authentication, identity verification, API authorization, input validation, rate limiting, and careful tool design. Chapter 5 includes these technical controls.

Likewise, the instruction to use only provided data helps reduce hallucination, but retrieval quality and source freshness remain important. If the knowledge base lacks the answer or is outdated, an obedient model may still be unable to offer a useful result.

## Appendix 2: self-service assistant structure

The project layout separates deployment, application logic, tests, prompt configuration, RAG processing, UI assets, and tools.

- **Root configuration:** `.env.example` documents required environment values; `requirements.txt` fixes dependencies; `Dockerfile` and `docker-compose.yml` package the service.
- **Tests:** API, memory, and ingestion tests map directly to the evidence reported in Chapter 6.
- **`app/main.py`:** FastAPI server and route handling.
- **`app/agent.py`:** construction of the tool-calling agent.
- **`app/llm.py`:** factory that chooses the configured provider.
- **`app/memory.py`:** SQLite-backed chat history.
- **`app/prompts/system_prompt.txt`:** the behavioural instruction described in Appendix 1.
- **`app/rag/`:** source configuration, ingestion, and vector-store logic.
- **`app/static/index.html`:** browser chat interface.
- **`app/tools/`:** integrations for number validation, products, refills, and subscription details.

This layout follows separation of concerns. A developer can change the UI without changing agent logic, add a new backend tool without rewriting the API server, or modify ingestion without touching conversation memory.

## Appendix 2: churn-prevention structure

The churn service is organized as its own deployable application, which reflects its batch/analytical role and different data dependencies.

- **Configuration and deployment:** environment template, dependencies, Dockerfile, and Compose file.
- **Tests:** scoring and campaign tests validate the analytical and decision-making parts.
- **`app/main.py` and `config.py`:** service entry point and adjustable thresholds.
- **`app/scheduler.py`:** APScheduler job definitions for periodic work.
- **`app/db/`:** MariaDB connection, SQLAlchemy models, and migrations.
- **`app/collectors/`:** refill and usage data retrieval.
- **`app/scoring/`:** feature engineering, XGBoost model handling, and risk-score calculation.
- **`app/campaigns/`:** offer selection, templates, and delivery through SMS, push, or app channels.
- **`app/api/`:** REST routes and Pydantic request/response schemas.

The directory structure mirrors the logical pipeline: collect data, create features, score risk, choose an intervention, deliver or expose campaign information. It also makes each stage testable independently.

## How the appendices connect to the thesis

The main chapters explain *why* the systems use RAG, APIs, agents, identity verification, and churn scoring. The appendices show *where* those responsibilities would reside in an implementation and *how* the LLM is instructed at runtime. Together, they make the thesis easier to reproduce conceptually even though they are not a complete source-code listing.

## Key takeaway

The appendices make the thesis concrete. The system prompt defines the customer assistant's expected communication and privacy behaviour; the project structures show a modular implementation for both the real-time assistant and the scheduled churn-prevention service. They reinforce that reliable AI systems depend on ordinary software boundaries, configuration, tests, and data pipelines as much as on the LLM itself.

