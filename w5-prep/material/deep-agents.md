# Understanding Deep Agents, Middleware, and Subagents

A standard RAG pipeline follows a direct pattern: retrieve relevant chunks from a document collection, add them to the LLM's context, and generate an answer. **Deep Agents** does not replace RAG. Instead, it gives an LLM structured capabilities to decide *how* to use RAG, tools, and other agents when solving more complicated tasks.

---

## 1. How Ordinary RAG Works

Suppose a user asks:

> *"What is the baggage limit for the Skylark-4 seaplane?"*

A typical RAG system follows a fixed sequence defined in application code: retrieve relevant chunks from the vector store, add them to the prompt, call the LLM, and return the answer (e.g., that the limit is 14 kg).

For straightforward, single-fact questions, ordinary RAG is simple, fast, and sufficient.

## 2. What Changes When a Question Becomes Complicated?

Now consider a multi-part question:

> *"I'm bringing a hard-shell suitcase on the Skylark-4 seaplane. What should I do, what will storage cost, and what happens if Protocol Gull is activated on Kaldor?"*

This question spans multiple sections or handbooks. A single `top-k` similarity search for the entire query can easily miss one of the parts.

An **agent** gives the LLM another option: instead of answering immediately, it can choose actions in a loop—for instance, searching for luggage restrictions first, storage fees second, evacuation rules third, and then combining the findings.

## 3. What Is a Deep Agent?

A **Deep Agent** is an agent harness equipped with ready-made capabilities for managing multi-step work. It is still powered by a standard LLM (not a special kind of model): the LLM handles reasoning and decisions, while Deep Agents provides tools, a virtual filesystem, subagent delegation, context management, and optional task planning.

```mermaid
flowchart LR
    subgraph Linear["Ordinary RAG (Fixed Pipeline)"]
        Q1["User Question"] --> R1["Similarity Search"] --> C1["Add Top-k Chunks"] --> A1["Generate Answer"]
    end
    subgraph Agentic["Deep Agent with RAG (Flexible Loop)"]
        Q2["User Question"] --> O2["Main Agent (LLM)"]
        O2 --> Act["Plan / Search / Delegate"]
        Act --> Eval{"Evidence Complete?"}
        Eval -- "No, search again" --> O2
        Eval -- "Yes" --> A2["Synthesize Answer"]
    end