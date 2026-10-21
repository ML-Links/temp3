# Solutions — `w1demo.ipynb`

**Mini Bookstore — Learn Tool Calling Step by Step**  
**Basis:** the supplied notebook's fictional four-book catalog, its Python functions, schemas and evaluator. These are reference answers. **The saved notebook includes a completed local-Qwen trace and local evaluation, but the Groq run was skipped because no API key was configured.**

## Step 1 — Ordinary Python functions

### Reasoning checks

**1. Which function should run first for “Find a sci-fi book under $16 with a rating above 4.5”?** Call `search_books("sci-fi", 16)` to obtain candidate books and their IDs, then call `get_book_rating(id)` for each candidate. The catalog search does not include rating data.

**2. Would `search_books("sci-fi", 20)` include *Foundation*?** Yes. It returns *Dune* (`id=1`, `$14.99`) and *Foundation* (`id=2`, `$18.50`), because both prices are at most `$20`.

**3. What does `get_book_rating(999)` indicate?** The response has `found=False`, meaning that ID 999 is not in the four-book catalog. It should not be represented as a book with a zero-star rating.

### Checkpoint 1 — Why two functions?

`search_books` inspects catalog fields (`id`, `title`, `genre`, `price`) and can find *Dune*, but it does not read `RATINGS`. Only `get_book_rating(1)` supplies the rating `4.67` from three reviews, which confirms `4.67 > 4.5`. For a fantasy book under `$10`, `search_books("fantasy", 10)` returns `[]`. An empty match set should be reported as no available match, not filled with invented records.

### Expected deterministic output

```python
search_books("sci-fi", 16)
# [{'id': 1, 'title': 'Dune', 'genre': 'sci-fi', 'price': 14.99}]

get_book_rating(1)
# {'book_id': 1, 'found': True, 'average_rating': 4.67, 'review_count': 3}

search_books("fantasy", 10)
# []
```

`BOOKS` and `RATINGS` are teaching fixtures, not current bookstore inventory. `book.copy()` prevents callers from accidentally modifying the original catalog through returned dictionaries. The `if __name__ == "__main__":` block runs direct-script assertions without running them during module imports. Type hints communicate intended types, while explicit runtime checks enforce requirements.

## Step 2 — Registered tools and schemas

### What the tool wrappers accomplish

The LangChain `@tool` decorator assigns each public function a name, description and input contract. Pydantic `SearchArgs` and `RatingArgs` constrain accepted fields and values; `extra="forbid"` blocks unexpected fields. `SCHEMAS` makes the contracts available to the model, and `TOOLS` lists the functions that the Python executor actually permits. The model is shown the contracts, **not** the values inside `RATINGS` until a tool executes.

The packages have distinct purposes: `transformers` loads the model and prepares chat messages; `accelerate` helps distribute model computation; `bitsandbytes` supports low-memory 4-bit loading; `langchain-core` supplies tool interfaces; `pydantic` supplies schemas and validation; `groq` supplies a hosted API client; and `pillow` supplies image-processing dependencies. Installing packages does not itself load model weights.

### Checkpoint 2 — Request, validation, result

Given:

```json
{"name": "get_book_rating", "arguments": {"book_id": 1}}
```

1. **Request:** the model (or test code) proposes a tool name and its arguments as data. No bookstore function has executed yet.
2. **Validation:** the executor checks the tool name against `TOOLS` and checks that `book_id` is a valid, positive integer using the schema.
3. **Execution and result:** Python invokes the registered wrapper, which calls `get_book_rating(1)`. The function returns a result containing `book_id=1`, `found=True`, `average_rating=4.67`, `review_count=3`; the wrapper serializes this as JSON.

**Optional experiment — `book_id="Dune"`:** The value is a title, not a numeric ID, so `RatingArgs` rejects it. The application should return a validation failure rather than trying to guess which ID the string meant. The unregistered `buy_book` request is blocked irrespective of its arguments.

## Step 3 — Local Qwen and the agent loop

**Prediction:** The intended path for `DEMO_QUESTION = "Find a sci-fi book under $16 with a rating above 4.5."` is approximately three **model turns** and two **tool calls**:

| Model turn | Anticipated action | Evidence supplied |
|---|---|---|
| 1 | Request `search_books` with `genre="sci-fi"`, `max_price=16` | *Dune*, ID 1, `$14.99` |
| 2 | Request `get_book_rating` with `book_id=1` | Rating `4.67`, three reviews |
| 3 | Answer the question | *Dune* qualifies based on both tool results |

The saved **`LOCAL_TRACE` confirms this sequence**: Qwen called `search_books` with `genre="sci-fi", max_price=16`, then `get_book_rating` with `book_id=1`, then returned `"Dune, $14.99, 4.67"`. The run finished with `status=final_answer`, **three model turns and two tool calls**. Other runs are not guaranteed to be identical.

**Why keep `history`?** It contains the system instruction, user question, assistant tool-call messages and corresponding `tool` result messages. The next Qwen generation reads the updated history, allowing it to derive `book_id=1` from the earlier search instead of inventing an ID.

**Checkpoint 3 — Where does the ID come from?** The `id: 1` field in the `search_books` result is the supporting source for `get_book_rating(book_id=1)`. The model may already know the title, but the ID should be verified against the tool output.

**Why `%run -i demo_qwen.py`?** The script runs in the notebook's interactive namespace and retains `model`, `processor` and `LOCAL_TRACE`. `!python demo_qwen.py` runs in a child process; its Python variables do not remain accessible to subsequent notebook cells. 4-bit quantization reduces memory use, while `model.eval()` configures inference rather than training.

**Loop interpretation:** `MODEL TURN` is a generation event; `PYTHON executes:` is an actual function invocation; `FINAL:` is the natural-language answer; `STATUS:` indicates completion or a guard; and `TOOL CALLS:` counts requests/executions, not generations. The loop is bounded by a four-turn limit and a repeated-request check.

## Step 4 — Compare with hosted Groq

**Who executes the local functions?** Even if Groq returns a structured request for `get_book_rating`, **Python in the Colab runtime** validates and runs it. Groq hosts the language model, not the local `BOOKS` and `RATINGS` objects. The API's `tool_call_id` associates each tool response with the request that produced it.

**Prediction — identical tool ordering?** No. The two models can differ in tool-choice decisions, argument extraction, formatting and number of turns even when given the same request and schemas. The first expected call is `search_books`; the expected second is `get_book_rating`. Both have to return the correct facts for the intended recommendation to be grounded.

**Trace-comparison answers:**

| Check | Grounded target | What must be checked in a real run |
|---|---|---|
| First call | `search_books("sci-fi", 16)` | **Observed:** local Qwen requested it and received Dune |
| Second call | `get_book_rating(1)` | **Observed:** local Qwen used ID 1 and retrieved rating 4.67 |
| Tool evidence | Dune `$14.99`; rating `4.67`, three reviews | **Observed:** both local tool executions succeeded |
| Final response | Dune meets both constraints | **Observed:** `Dune, $14.99, 4.67` |
| Elapsed time | Local: `12.37 s` in the saved evaluator | Groq was not run; no comparable hosted measurement |

**Observed comparison status:** The local Qwen trace completed as described. The notebook's Groq cell printed `No Groq key` and `Groq test skipped`, so there is **no actual hosted trace** to compare. The evaluator recorded **12.37 seconds** for this local run and no duration for Groq. This single measurement cannot establish a general speed ranking. Keep `GROQ_API_KEY` in Colab Secrets or a secure environment variable; do not publish it.

## Step 5 — Evaluate the evidence

`demo_evaluate.py` derives the oracle from `search_books("sci-fi", 16)` and `get_book_rating(1)`. Its rubric checks whether the agent finished, made successful calls in the intended order, retrieved the expected facts, mentioned *Dune*, `14.99` and `4.67`, and records elapsed seconds. These checks can establish narrow observable conditions but not overall factuality or typical speed.

### Checkpoint 4 — Evaluate the evaluation

1. **Could the model state the three correct strings without tools?** Yes. It might recall or guess them. The separate `tools_in_order` and `verified_tool_facts` checks would detect that those two tool-evidence conditions were not met.
2. **Why use varied questions and repeated runs?** They test generalization to different phrasings, missing data, argument validation and nondeterministic model behavior. One successful trial does not estimate reliability.
3. **What if the tool is wrong but the answer sounds reasonable?** It should not count as fully verified tool-grounded success. Report separately whether the final answer happened to be correct, whether required tools were used and whether the resulting claims were supported.

The saved evaluation reports local Qwen `task_done=True`, `tools_in_order=True`, `verified_tool_facts=True`, and `answer_mentions_facts=True`; Groq appears as **`not run`**. A final answer can pass keyword checks while containing other incorrect claims. A failed keyword check may also miss a correctly paraphrased answer. Manual inspection and broader case coverage remain necessary.

## Final reflection — all six questions

1. **Ordinary function vs registered tool vs agent loop:** A Python function performs local computation. A registered tool adds an LLM-readable contract around a callable. An agent loop coordinates model requests, validation, execution, history updates and stopping conditions.
2. **Who selects and executes tools?** The model proposes a named tool and arguments. The host application applies the allowlist and schema checks, then Python executes the approved function.
3. **Why no ratings in search?** The catalog and `RATINGS` are intentionally separate. `search_books` returns metadata, and `get_book_rating(book_id)` supplies the evidence required to compare a candidate against a rating threshold.
4. **What is an argument schema?** A structured description of permitted fields, types and constraints. Unregistered names are blocked by the executor; invalid arguments are rejected by validation.
5. **Local versus hosted:** Local Qwen generates on the Colab GPU with model weights loaded through Transformers; Groq generates on hosted servers through an API and returns structured tool requests. The underlying Python functions, allowlist and need to send back tool results remain unchanged.
6. **What evaluation verifies and misses:** It checks selected trace events, expected returned data, selected answer strings, status and one-run latency. It cannot prove complete factual accuracy, safety against every attack, consistent performance, statistical superiority or production readiness.

## Next week's coding marathon — example design

**Example service: a public-transport assistant.** Two read-only tools could be `search_stops(query: str, limit: int = 10)` and `get_departures(stop_id: str, limit: int = 5)`. A real data source to investigate is the **Digitransit routing API** (`https://digitransit.fi/en/developers/`), subject to its current access, authentication and usage policies. Both tools should validate input, bound output, handle empty results and return structured data; neither should perform purchases or alter external state. This is a design proposal, not a live API integration.
