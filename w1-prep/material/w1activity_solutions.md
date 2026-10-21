# Solutions — `w1activity.ipynb`

**Activity: Tool Calling and Agentic AI with Local and Hosted LLMs**

**Evidence convention.** Answers are based on the uploaded notebook's definitions, SQLite seed records, exercises and evaluation logic. Values deterministically implied by the supplied code are identified as **expected** values. **The notebooks contain saved Colab execution outputs, including local and hosted tool traces, model timings and Part I scoring tables.** These saved observations are reported as observations of those particular runs. The separate cover-image file `dl.jpg` and exported JSON/CSV data files are not included, so independent image-ground-truth assessment and trace details not printed in the notebook remain limited. These solutions do not invent additional experimental results.

## Part A — Bookstore backend

### A1 — Database setup and exploration

**1. Relationship between `books` and `reviews`.** It is one-to-many: one book can have multiple reviews; each review links to one book through `reviews.book_id → books.id`. Keeping reviews in a separate table avoids repeating book metadata, supports any number of reviews and permits straightforward aggregate queries.

**2. Why `is_bestseller` is stored as 0 or 1.** SQLite does not use a separate storage class equivalent to Python's `bool`; a Boolean is conventionally represented using integers `0` and `1`. The schema constrains values to these two integers. Python maps them to `False` and `True` at the application boundary.

**3. Meaning of an empty average rating.** `NULL` means that no rating values are available for the book. It differs from numeric `0`, which would be an actual average value (and could represent submitted zero-star reviews). It is incorrect to describe an unrated book as zero-rated.

**4. `LEFT JOIN` versus `INNER JOIN`.** The `LEFT JOIN` retains every book, including books with no reviews. Those books receive `review_count=0` and `average_rating=NULL`. An `INNER JOIN` would exclude books with no matching rows in `reviews` entirely.

**5. Safe repeated setup.** `CREATE TABLE IF NOT EXISTS`, `INSERT OR IGNORE` for books, and the conditional insertion of seed reviews keep an ordinary rerun from duplicating data. Existing orders remain intact. Only an explicit `--reset` removes the database and its orders. This is important because rerunning a notebook should not silently lose purchases or distort review aggregates. Note that the review seeding condition checks whether the *entire* review table is empty; it is intended for initial seeding rather than repairing partly missing reviews.

**Experiment A1 — only sci-fi rows.** Expect **four** rows: *Dune*, *Project Hail Mary*, *Neuromancer*, and *Foundation*. One minimally invasive edit is:

```python
# In db_view.py, inside load_catalog(), add before GROUP BY:
#     WHERE b.genre = ?
# and pass the argument to pandas:
return pd.read_sql_query(sql, conn, params=("sci-fi",))
```

The full query fragment is:

```sql
FROM books AS b
LEFT JOIN reviews AS r ON r.book_id = b.id
WHERE b.genre = ?
GROUP BY b.id
ORDER BY b.id
```

To restore the original 20-row catalog, remove the `WHERE` clause and remove `params=("sci-fi",)` from the `pd.read_sql_query` call. On a fresh database the expected setup counts are **20 books, 25 reviews, 0 orders**; rerunning without `--reset` does not change them.

### A2 — Plain Python API

**1. Filtered sci-fi bestseller search.**

```python
search_books(genre="sci-fi", max_price=16, is_bestseller=True)
# Expected:
# [{'id': 1, 'title': 'Dune', 'author': 'Frank Herbert',
#   'genre': 'sci-fi', 'price': 14.99,
#   'format': 'Paperback', 'is_bestseller': True}]
```

*Dune* is the only sci-fi bestseller priced at most `$16` in the seed data. *Project Hail Mary* is a bestseller but costs `$18.99`. The other sci-fi entries are not bestsellers. The code uses `price <= max_price`, so its numerical boundary is **inclusive**, although everyday English “under $16” can imply `< 16`.

**2. SQL placeholders.** `?` placeholders bind values separately from SQL syntax, preventing user input from being interpreted as executable SQL and handling quoting and types properly. Do not interpolate an untrusted search string into the query text.

**3. Why `if is_bestseller:` fails for `'False'`.** Nonempty strings are truthy in Python, so `'False'` in a condition behaves as true. `_normalize_bestseller` parses recognized true/false strings, Boolean values and `0`/`1` into a real Boolean; it rejects values such as `'maybe'`.

**4. Read versus write.** `search_books` performs a `SELECT` and returns rows without altering SQLite. `order_book` performs an `INSERT`, commits the transaction and creates a persistent order record. That is a consequential state change requiring confirmation and authorization.

**5. Missing versus unrated book.** `get_book_rating(999)` returns `{'book_id': 999, 'found': False, 'average_rating': None, 'review_count': 0}` because there is no book 999. `get_book_rating(4)` returns the same missing numeric rating and zero reviews but with **`found=True`** because *Foundation* exists and is unrated.

**6. Why confirmation before ordering.** A model might misread the user, select the wrong ID or be manipulated by untrusted content. An order commits a purchase-related write. A separate approval step should show the exact title, price and action and require the user's authorization before execution.

**Experiment A2 — fantasy under `$15`.**

```python
search_books(genre="fantasy", max_price=15)
# Expected: The Hobbit only (book ID 5; $13.99).

search_books(is_bestseller="False")
# Expected book IDs: 3, 4, 6, 9, 10, 12, 16, 18, 20.
```

The Boolean-string call normalizes `'False'` to `False` and selects non-bestsellers; it does not accidentally select bestsellers because a string is truthy. The default `limit=10` is sufficient for all nine seeded non-bestsellers.

## Part B — Install and load local Qwen3.5-4B

### B1 — Check the environment

**1. Assigned GPU and total memory.** The saved B1 output identifies **Tesla T4**, with **14.56 GiB total** and **14.46 GiB free** at the time of the check. The preflight printed `PASS` (Python 3.13.15; PyTorch 2.11.0+cu130). These are observed values for that session, not a guarantee of the next Colab allocation.

**2. Why check `torch.cuda.is_available()` first?** To verify that PyTorch sees a CUDA-enabled GPU before downloading large model weights or attempting GPU placement. If false, change the Colab runtime or diagnose its drivers/dependencies.

**3. Total, free, allocated.** **Total** is the device's physical video memory capacity; **free** is memory currently not in use as reported by the CUDA driver; **allocated** is memory actively allocated by the relevant process/framework (for example, PyTorch tensors). Framework-reserved or cached memory may differ from currently allocated memory and global device use.

**Try after B3.** The saved B3 output reports a 4-bit `Qwen/Qwen3.5-4B` model (`Qwen3_5ForConditionalGeneration`) with `Qwen3VLProcessor` and **3.08 GiB GPU allocated**. The later `nvidia-smi` readout reports **3,385 MiB used, 11,528 MiB free, 15,360 MiB total**. Free memory decreased from the B1 preflight, as expected. Framework allocation and system-wide used values differ.

### B2 — Dependencies

**1. `transformers` vs `accelerate`.** `transformers` supplies model classes, processors, tokenizers, generation and chat-template interfaces. `accelerate` supports device placement and efficient model loading/execution across available hardware.

**2. Why `bitsandbytes`?** It supports reduced-precision quantization, including 4-bit representations, allowing large model weights to fit within limited GPU memory. Quantization can trade some numerical precision for memory savings.

**3. Installation vs weight loading.** Installing downloads software libraries and their dependencies into the Python environment. Loading the model downloads or reads learned parameter files, constructs the network, and moves or maps weights onto available devices. They are separate operations.

**4. `!python` vs `%run -i`.** Running `qwen_install.py` in a separate process changes installed packages without needing retained variables. Running `qwen_load.py` through `%run -i` keeps the loaded `model`, `processor` and `tokenizer` available in the notebook's current kernel. Running the loader through `!python` would not preserve those objects in the notebook.

### B3 — Local multimodal model

**1. Weights vs tokenizer.** **Weights** are the learned numerical parameters used to compute predictions. A **tokenizer** maps between text and token IDs (and related text-specific input conventions). They serve different purposes.

**2. Why `AutoProcessor`?** The model is multimodal. `AutoProcessor` can prepare text, chat messages and image inputs through the appropriate processing components; `AutoTokenizer` alone is text-centered and insufficient for the later actual-image task.

**3. 4-bit trade-off.** Quantized weights generally consume less GPU memory than full-precision weights, but approximating values can introduce numerical error and possibly change output quality or tool-selection behavior. The degree of effect is empirical and task-dependent.

**4. Loading using `!python`.** The model would be created in a child process, and the next notebook cell would normally raise a `NameError` when trying to use its unavailable `model` variable.

**5. Why no Groq key?** The local model runs in the assigned GPU runtime through Transformers, without calling Groq's inference API. Downloading public weights from a model repository is distinct from obtaining a Groq API credential.

### B4 — Smoke test and exercise

**1. Prompt vs generated tokens.** Prompt tokens encode the input context. Generated tokens are new tokens produced by the model in response.

**2. Decode only new tokens.** Autoregressive generation often returns the full prompt-plus-completion sequence. Slicing past input length prevents reprinting the input as if it were new output.

**3. `max_new_tokens`.** It caps the number of newly produced tokens. Too small a value may truncate the response; a higher cap may permit more complete output but consumes additional compute and memory.

**4. Can it know exactly how many books are in `bookstore.db`?** Not reliably. At this stage the model is not connected to SQLite and cannot inspect the live database. From the provided seed script, **we** can predict 20 seeded books, but the model's fluent claim would not verify the current file.

**5. `enable_thinking=False`.** It asks the model/chat template to use a non-thinking generation mode. That can make a short smoke test more focused and avoid spending the generation budget on intermediate reasoning or planning output.

**Student exercise — two-sentence bestseller explanation.** Change only the `QUESTION` string and rerun the writefile and smoke-test cells:

```python
QUESTION = "In exactly two sentences, explain what a bestseller is."
```

A possible compliant **example answer** for the proposed changed prompt (not run in the saved outputs) is: “A bestseller is a book that achieves comparatively high sales according to a specified retailer, market or ranking list. Criteria and measurement periods vary, so a bestseller designation does not necessarily describe review quality.” The outcome should be checked for exactly two sentences and relevance. A model rerun may deviate; no measured deviation is available here.

### B — Exit ticket

> In Part A, `search_books` was a regular **Python** function that queried SQLite. In Part B, we loaded a **local Qwen language** model, but it cannot call `search_books` because we have not yet **registered/exposed** it as a tool.

`!python filename.py` starts an independent Python process, while `%run -i filename.py` runs in the notebook's interactive namespace. The saved outputs **do establish** that the B1 preflight passed, the 4-bit model and processor loaded successfully, and B4 generated a 25-token answer to the 23-token prompt: “A book genre is a category of literature that groups works together based on shared themes, styles, or conventions.” Optional acceleration-kernel fallbacks were reported as warnings, not execution failures.

## Part C — Tool contracts, validation and safe execution

### C1 — Installation and formal schema

1. **LangChain's role:** `langchain-core` provides tool interfaces, decorator behavior and schema exposure; Pydantic enforces argument contracts. This section invokes Python tools directly and **does not use an LLM**.
2. **Formal input schema:** It specifies field names, optionality, types, ranges and forbidden extras, limiting ambiguity and catching unsuitable arguments before SQL access.
3. **Process difference:** Installing dependencies with `!python` does not need to preserve notebook objects. `%run -i` is appropriate for scripts that use or produce persistent notebook variables, such as a loaded model or trace.

### C2 — Register the read-only tools

1. **`@tool`:** Wraps the function as a named LangChain tool with a description, validation schema and standardized `.invoke(...)` interface.
2. **Description matters:** A model needs to know when a tool is useful and what it does or does not return. `search_books` explicitly omits ratings, while `get_book_rating` supplies them. Names alone do not convey those limits.
3. **`Optional[bool]` vs `str`:** `Optional[bool]` accepts a true/false value or no filter (`None`), subject to Pydantic validation. A `str` type describes free-form text and would not itself express the intended Boolean semantics.
4. **`"False"` vs `"maybe"`:** Under the notebook's Pydantic schema, the recognized string `"False"` is parsed to Boolean `False`. `"maybe"` is invalid and should be rejected, rather than guessed.
5. **Why JSON output?** Structured JSON preserves the exact returned fields and values so the model and test harness can inspect them consistently. An arbitrary paragraph may hide IDs, omit fields or introduce interpretation.

### C3 — Inspect schemas and invoke manually

**Prediction before running:** For `search_books`, `query`, `genre`, `max_price` and `is_bestseller` are optional (default `None`), and `limit` has default `10`. The `get_book_rating` tool requires `book_id`, a strict positive integer. Book 999 yields a structured not-found result.

1. **Schema vs data:** The printed `TOOL_SCHEMAS` blocks contain `name`, `description`, `properties`, `required` and field constraints. Output from the later `.invoke(...)` calls is **actual data returned from SQLite**.
2. **Who chooses?** The **programmer** explicitly selects tools in `c_inspect_tools.py`; there is no model choice in these tests.
3. **True, False, omitted:** `True` filters to bestseller rows; `False` filters to non-bestseller rows; `None`/omission does not apply a bestseller filter.
4. **List vs one record:** Searches may match many book rows, so `search_books` returns a list; a rating lookup targets one catalog ID, so it returns one structured dictionary including found status and aggregate review values.

### C4 — Reproducible tool checks

The offline tests establish that valid tool inputs return expected values, malformed inputs are rejected, the string `"False"` is normalized appropriately, and the order count remains unchanged. They do **not** prove a model will choose the right tool or generate an accurate answer.

### Part C practical exercises

**1. Change `max_price` from 16 to 10.** The `c_inspect_tools.py` example filters to sci-fi bestsellers. The expected result is `[]`: *Dune* costs `$14.99`, and there is no sci-fi bestseller priced at most `$10`.

**2. Negative test for `book_id=0`.** Add to `c_tool_checks.py`:

```python
from pydantic import ValidationError
from bookstore_tools import BookRatingArgs

try:
    BookRatingArgs.model_validate({"book_id": 0})
    raise AssertionError("Expected book_id=0 to be rejected")
except ValidationError:
    pass
```

`gt=0` requires a value strictly greater than zero and `strict=True` requires an integer rather than an implicitly coerced string.

**3. Read-only `get_book_details(book_id)` schema (design exercise).**

```json
{
  "type": "object",
  "properties": {
    "book_id": {
      "type": "integer",
      "exclusiveMinimum": 0,
      "description": "Positive catalog ID"
    }
  },
  "required": ["book_id"],
  "additionalProperties": false
}
```

The result can include `found`, `id`, `title`, `author`, `genre`, `price`, `format` and `is_bestseller`. A missing ID returns `found=False`; no fake book information is added.

**4. Why syntax alone cannot authorize ordering.** A valid tool call might not reflect informed user consent, may target the wrong book and may have been influenced by untrusted data. The application must independently require approval for a write action.

**Exit ticket:** A tool definition describes what the model is allowed to request. The trusted Python executor determines what actually runs.

## Part D — Tool selection and safe execution

### D1 — Qwen's tool-call protocol

1. **Meaning of `<tool_call>`:** It marks a model-generated proposal to invoke a named tool with specified parameters. It is not executable Python on its own.
2. **Why parse `"true"`, `"false"`, `"16"`:** The model may emit text representations of typed values. Parsing them into Booleans and numbers lets the schema check their meaning and prevents string truthiness or numeric comparisons from going wrong.
3. **Why not execute generated text:** Model output is untrusted. Executing it as Python could permit arbitrary commands, unsafe database updates and other operations outside the approved tool set.
4. **Malformed request:** Reject the malformed structure, record a clear status/error and execute no inferred action. Do not guess a missing tool name or silently repair unsafe arguments.

### D2–D3 — Allowlist and protocol checks

1. **Who sets executable functions?** The application developer controls the trusted tool registry and dispatcher; the model cannot add its own executable functions.
2. **Unregistered `buy_book`:** The executor marks the request `blocked` and does not call `order_book` or alter SQLite.
3. **Why test strings first:** Known examples isolate parser, coercion and validation defects without incurring inference cost or introducing nondeterministic model behavior.
4. **Unchanged order count:** It establishes that **this particular test interval** did not add or remove orders. It does not prove that all future workflows are safe or that attempted malicious actions are impossible.

### D4 — Ask Qwen to select a tool

**Observed request:** In the saved D4 output, Qwen selected **`search_books`** using native XML-like `<tool_call>` markup with `genre="sci-fi"`, `is_bestseller=True`, `max_price=16`, and `limit=10`. The parsed dictionary preserved `True` as a Boolean and `16`/`10` as numbers. The model's choice **matched the prediction**. D4 itself only parsed output; D5 then executed the selected tool and returned the actual Dune row (`id=1`, `$14.99`, bestseller true) with executor status `ok`.

1. **Which tool / prediction:** `search_books`, as predicted.
2. **Encoding / types:** Native XML-like markup was parsed into a function-name dictionary; `is_bestseller` became `True` (`bool`), `max_price=16` and `limit=10` became integers.
3. **Why model text is not SQL execution:** The model produced a proposal. SQL ran only when the Python executor was invoked in the subsequent D5 step.
4. **No tool request case:** A normal answer is handled as model text; if the requested answer requires database facts but no tool ran, the response is not grounded by this workflow. Malformed calls should stop safely.

### Part D practical exercises

**1. Change question to rating for book ID 1.** Expected requested tool: `get_book_rating` with `{"book_id":1}`. Expected SQL-derived result: `{"book_id":1,"found":true,"average_rating":4.67,"review_count":3}`. Whether Qwen actually made that request must be confirmed in the D4/D5 trace.

**2. Ask “Hello!”** A direct greeting with **zero tools** is appropriate. The model may still call a tool due to probabilistic selection errors, confusing context, or overgeneralizing tool descriptions; if so, record that deviation.

**3. Ask “Please buy book ID 1.”** The read-only allowlist does not expose ordering. An appropriate final response states that the assistant cannot place purchases; any attempted `order_book` or `buy_book` call should be blocked. Verify `SELECT COUNT(*) FROM orders` before and after; expected change is zero.

**4. Fabricated `book_id="oops"`.** The parser may recover the string, but the `BookRatingArgs` schema rejects it as a noninteger. Expect an `invalid_arguments` or equivalent unsuccessful-execution status, not a rating lookup.

**5. Hallucination and SQL evidence:** A model may invent a book title or rating despite having access to tools. A validated, parameterized SQL result supplies inspectable ground truth about the actual seeded data, so the final answer can be checked against it.

**Exit ticket:** The model proposes; the trusted host parses, validates and executes only registered requests; later turns can use the returned data.

## Part E — Multi-step tool calling

### E0 — Predict the call sequence

1. **What search returns:** `search_books` returns matching catalog fields such as ID, title, genre, price and bestseller flag. It does **not** return average ratings.
2. **Why rating lookup is needed:** The request requires a rating strictly above `4.5`; that fact must be obtained independently through `get_book_rating`.
3. **Model turns:** In the illustrative one-tool-per-turn sequence, **three** turns: search, rating lookup, final answer. This is not guaranteed for a live model.
4. **Selection vs execution:** Qwen chooses candidate tool names/arguments; the trusted Python dispatcher executes validated functions.
5. **No matching books:** Return a truthful “no matching books found” response. Do not fabricate a title or a review score.

**Expected result for the main question:** *Dune*, `$14.99`, average rating `4.67` over three reviews, and bestseller status true. The search alone cannot prove the rating requirement.

### E1 — Agent-loop internals

1. **History after execution:** The `history` list starts with `system` and `user` messages, then adds an `assistant` entry containing structured `tool_calls` and a `tool` entry containing serialized JSON execution status and results.
2. **How the next turn sees data:** The processor formats the updated `history` into Qwen's input. Tool response messages are then in the model's conversational context.
3. **Why validate despite descriptions:** The model may ignore descriptions or be attacked through tool input. The dispatcher must independently enforce allowed names and schemas.
4. **Why budgets and repeated-call detection:** `max_rounds` and `max_tool_calls` bound cost and execution. Repeated-call detection helps prevent endless loops and duplicate redundant requests.
5. **Meaning of `max_rounds_reached`:** The loop exhausted its model-turn allowance before producing a normal final answer. It is a termination condition, **not successful task completion**.

### E2 — Scripted orchestration tests

1. **Why useful first:** They are inexpensive, repeatable and isolate code defects in history updates, parsing, repeated-call checks and guard handling.
2. **Difference from live Qwen:** Tests substitute predetermined fake generation responses; they do not involve actual model tool-selection decisions.
3. **What is and is not established:** Tests cover control-flow correctness, successful dispatch for known inputs, malformed/repeated requests, budgets and cleanup. They cannot establish Qwen's accuracy, speed, reliability or spontaneous choice of tools.

### E3 — Inspect the actual agent trace

The saved `E_TRACE` and its printed inspection give a **successful observed three-turn, two-tool path**:

| Model turn | Observed request | Actual SQLite result / outcome |
|---|---|---|
| 1 | `search_books(genre="sci-fi", is_bestseller=True, max_price=16, limit=100)` | `ok`: Dune, ID 1, `$14.99`, bestseller true |
| 2 | `get_book_rating(book_id=1)` | `ok`: `found=True`, `average_rating=4.67`, `review_count=3` |
| 3 | No tool request; final text | `final_answer`: “The book "Dune" by Frank Herbert is a sci-fi bestseller priced at $14.99 and has an average rating of 4.67 based on 3 reviews.” |

1. **First tool:** `search_books`, matching the prediction. The model chose `limit=100` instead of the `10` seen in Part D; both are within the schema and returned the same candidate here.
2. **Exact SQL evidence / where stored:** The two results above were printed with `status=ok` and stored in each turn's `tool_results` and in the corresponding `role="tool"` messages in `trace["messages"]`.
3. **Was the rating checked?** Yes. Qwen requested and received `get_book_rating(1)` before giving its recommendation.
4. **Turns versus requests:** Three model turns, two tool calls. The final turn generated ordinary text rather than a request.
5. **Why grounded:** The final title, price, rating, and review count agree with the successfully retrieved records.
6. **Hallucination check:** Compare each numeric claim against the `get_book_rating` result. In this recorded trace the stated rating **matches** the tool result; a conflicting number or unsupported rating would be a failure.

### E4 — Inspect and interpret trace details

1. **Evidence results were fed back:** Inspect `trace["messages"]`: after an assistant tool request there should be a `{"role":"tool",...}` message with serialized JSON, preceding the next generation turn. Merely printing the result would not create such history.
2. **When to stop:** Once the required evidence is available and a grounded answer can be given, or when a malformed call, repetition or safety budget requires termination.
3. **Malformed/repeated/budget-exceeding request:** The loop records an explicit failure/guard status, stops the affected execution path, and does not silently perform unsafe extra operations.
4. **If orders were exposed:** The agent could modify external state. A separate confirmation step, per-user authorization and transaction safeguards would be necessary before any write.

### E5 — Changed questions: predictions and evidence log

| Prompt | Appropriate predicted tool(s) | Expected database result / supported answer |
|---|---|---|
| Fantasy bestseller under `$15`, rating above `4.5` | `search_books` then `get_book_rating` | *The Hobbit*: `$13.99`, bestseller, rating `(5.0+4.5)/2 = 4.75`, two reviews |
| Rating for ID 999 | `get_book_rating(999)` | `found=False`; book is missing, not merely unrated |
| “Hello! What can you help me with?” | None | A short description of read-only catalog, rating and (after Part F) review help; no claim of purchase capability |
| Sci-fi under `$10`, rating above `4.9` | `search_books` | Empty list; no qualifying book is in the catalog, so no rating verification is needed |

**Trace log for each prompt:** Record `(predicted calls, actual requests, statuses and results, final answer, total turns, total calls, termination status)` from the new `E_TRACE`. The E5 cell only changed `E_QUESTION` and instructed a rerun; it does **not** include new agent traces for those four alternative prompts. Their rows remain predictions, distinct from the observed main `E_TRACE`.

### E — Exit ticket

> The model **proposes** a tool call, but **the Python host/executor** validates and executes it. The tool result is appended to the conversation as a **tool** message. The next model turn may request **another tool** or provide a **final** answer. The loop stops when **a final answer is produced or a guard/limit is reached**.

**Part F planning:** Introduce `get_book_reviews(book_id, limit)` as a new, parameterized, read-only database function, define a strict schema, add its wrapper to the existing registry and dispatcher, and test the result and safety boundary before live model use. Avoid exposing `order_book`.

**Boundary note:** SQL code uses `price <= max_price`. If “under $16” must be strictly below $16, explicitly change the query/operator or represent the intended inequality separately.

## Part F — Add a read-only review tool

### F0 — Prediction

For “Show me two customer reviews of Dune,” use `get_book_reviews(book_id=1, limit=2)` after confirming Dune's ID if needed. `get_book_rating` only returns an aggregate and count, not review text, reviewer names or individual scores.

### F1 — API function and questions

The notebook's `get_book_reviews` function is already a complete reference implementation: it validates the arguments, checks that the book exists, performs a parameterized `SELECT`, and returns reviews in descending review-ID order.

1. **`book_id`:** The foreign key connecting rows in `reviews` with one `books.id` value; it is a catalog ID, not a title.
2. **Bound `limit`:** Restricting it to 1–10 prevents accidental or model-induced oversized outputs, unnecessary database work and excessively long prompts.
3. **Concatenating generated strings into SQL:** It can change the query's syntax or introduce an injection vulnerability. Parameterized placeholders keep supplied values separate from SQL commands.

**Expected results:** `get_book_reviews(1,2)` returns two Dune reviews, newest first: Chani (`4.5`, “Classic sci-fi that still holds up.”) and Jessica (`4.5`, “Complex politics, unmatched depth.”). `get_book_reviews(999)` returns `found=False`, `reviews=[]`, `returned_count=0`. For an existing book without reviews, `found=True` and `reviews=[]`.

### F2 — Tool schema and questions

1. **What the model sees:** Tool name `get_book_reviews`, its description, the `book_id` parameter (required, strict positive integer) and optional `limit` (default 3; allowed 1–10). Once executed, it also sees the tool's JSON output.
2. **What it does not see:** The Python source code, the SQL connection's internal state or the review rows before successful execution, unless separately included in context.
3. **Validation versus prediction:** The model's argument proposal is a fallible decision. Pydantic and Python runtime checks provide deterministic acceptance/rejection for the input that is actually received.

### F3 — Extended dispatcher

1. **If Qwen requests `order_book`:** The extended executor returns a `blocked` result because only `search_books`, `get_book_rating` and `get_book_reviews` are registered. No order function is called.
2. **Why a system prompt is insufficient:** Natural-language directions are guidance, not an enforceable access-control boundary. A model may misinterpret instructions, produce unexpected output or follow adversarial review text. The Python allowlist enforces the boundary.

### F4 — Deterministic extension checks

The supplied `f_check_extension.py` asserts that Dune returns two reviews for `limit=2`, all three for `limit=10`, and that a missing ID is differentiated. It also checks invalid `limit=100`, blocked `order_book` and unchanged `orders` count. Passing these checks would establish those backend properties only; it would not establish Qwen's selection behavior.

### F5 — Actual Qwen use and untrusted review text

**Observed saved `F_TRACE`:** Qwen first called `search_books(query="Dune", limit=1)`, received catalog ID 1, then called `get_book_reviews(book_id=1, limit=2)`. Both executor responses had status `ok`. In model turn 3 it returned exactly two reviews:

- Chani: “Classic sci-fi that still holds up.” (4.5)
- Jessica: “Complex politics, unmatched depth.” (4.5)

The loop ended with `final_answer`, **three model turns and two tool calls**.

1. **Did Qwen request the new tool?** Yes, `get_book_reviews` was the second observed request.
2. **Did it find Dune's ID first?** Yes, it searched for Dune and took `id=1` from the first result.
3. **Were the reviews copied accurately?** Yes. Both quotes and reviewer-score pairings match the saved SQLite tool result.
4. **Could review content inject instructions?** Yes; review text is untrusted data and must never override system rules or authorize purchases. The Python read-only allowlist remains an independent guard.

### F6 — Student extension challenge: complete reference solution

The notebook asks students to implement **one** of two options. Below are solutions to both; apply one at a time. Both are read-only and must be integrated into the existing scripts before running the real model.

#### Option 1 — Optional minimum review rating

Extend `get_book_reviews` in `f_review_api.py` with a `min_rating` argument. The filtering applies to **individual reviews**, not the book's overall average. Validate it before constructing SQL:

```python
# Add min_rating=None to get_book_reviews(...).
if min_rating is not None:
    if isinstance(min_rating, bool) or not isinstance(min_rating, (int, float)):
        raise ValueError("min_rating must be numeric")
    if not 0 <= min_rating <= 5:
        raise ValueError("min_rating must be between 0 and 5")

sql = ("SELECT rating, reviewer_name, review_text "
       "FROM reviews WHERE book_id = ?")
params = [book_id]
if min_rating is not None:
    sql += " AND rating >= ?"
    params.append(float(min_rating))
sql += " ORDER BY id DESC LIMIT ?"
params.append(limit)
rows = conn.execute(sql, params).fetchall()
```

Update the corresponding schema and wrapper:

```python
from typing import Optional

class BookReviewsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    book_id: int = Field(..., strict=True, gt=0)
    limit: int = Field(3, ge=1, le=10)
    min_rating: Optional[float] = Field(None, ge=0, le=5)

@tool("get_book_reviews", args_schema=BookReviewsArgs)
def get_book_reviews_tool(
    book_id: int, limit: int = 3, min_rating: Optional[float] = None
) -> str:
    """Read up to limit reviews of a book, optionally filtering review scores."""
    return json.dumps(get_book_reviews(book_id, limit, min_rating), ensure_ascii=False)
```

**Deterministic test:** `get_book_reviews(1, limit=10, min_rating=4.75)` should return only the Dune review rated `5.0` (Paul), as the other two are rated `4.5`. `min_rating=-1` and `min_rating=6` must fail validation. The execution trace must still show zero change in the order count.

#### Option 2 — New `count_book_reviews(book_id)` tool

Add a new module, or add the function to `f_review_api.py`:

```python
def count_book_reviews(book_id: int) -> dict:
    """Return the number of reviews for an existing catalog ID."""
    if type(book_id) is not int or book_id <= 0:
        raise ValueError("book_id must be a positive integer")
    import sqlite3
    from contextlib import closing
    from books_api import DB_PATH
    with closing(sqlite3.connect(DB_PATH)) as conn:
        book = conn.execute(
            "SELECT 1 FROM books WHERE id = ?", (book_id,)
        ).fetchone()
        if book is None:
            return {"book_id": book_id, "found": False, "review_count": 0}
        count = conn.execute(
            "SELECT COUNT(*) FROM reviews WHERE book_id = ?", (book_id,)
        ).fetchone()[0]
    return {"book_id": book_id, "found": True, "review_count": count}
```

In `f_extended_tools.py`, add a wrapper and register it:

```python
class CountBookReviewsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    book_id: int = Field(..., strict=True, gt=0)

@tool("count_book_reviews", args_schema=CountBookReviewsArgs)
def count_book_reviews_tool(book_id: int) -> str:
    """Count the customer reviews of a book by its numeric catalog ID."""
    return json.dumps(count_book_reviews(book_id), ensure_ascii=False)

# After importing count_book_reviews and defining count_book_reviews_tool:
F_TOOLS = {
    **READ_ONLY_TOOLS,
    get_book_reviews_tool.name: get_book_reviews_tool,
    count_book_reviews_tool.name: count_book_reviews_tool,
}
# Rebuild F_TOOL_SCHEMAS from F_TOOLS as already done in the notebook.
```

**Deterministic test:**

```python
assert count_book_reviews(1) == {
    "book_id": 1, "found": True, "review_count": 3
}
assert count_book_reviews(999) == {
    "book_id": 999, "found": False, "review_count": 0
}
assert execute_f_calls([
    {"name": "count_book_reviews", "arguments": {"book_id": 1}}
])[0]["status"] == "ok"
```

Update the previous assertion requiring exactly three registered tools and any hard-coded tool lists, because this version intentionally adds a **fourth** tool. Then ask the real model, “How many customer reviews does Dune have?” An appropriate request is `count_book_reviews` with `{"book_id":1}`, but **the model's actual request is not available here**. The real tool result should show `review_count=3`. Check `orders` count before and after the test; equality shows that this particular execution did not change the count. `count_book_reviews` contains only `SELECT` queries and no order operation.

**Reflection — selection or implementation problem?** If direct `.invoke(...)` succeeds with correct values, but Qwen chooses another tool or malformed arguments, investigate tool choice, schema interpretation or parsing. If both manual invocation and model invocation fail with the same validated inputs, investigate function logic, registration or database access. Compare raw requests with dispatcher outcomes before assigning responsibility.

## Part G — Edge cases and reflection

### G1 — Predictions for all five cases

| Case | Expected call(s) and arguments | Expected grounded response |
|---|---|---|
| G1: sci-fi bestseller, `< $16`, rating `>4.5` | `search_books(genre="sci-fi", max_price=16, is_bestseller=True)`; `get_book_rating(book_id=1)` | *Dune* is `$14.99`, bestseller, average `4.67` (3 reviews) |
| G2: two reviews of Dune | `get_book_reviews(book_id=1, limit=2)`, optionally first `search_books(query="Dune")` | Two actual review records, e.g. Chani and Jessica in descending review-ID order |
| G3: rating for book ID 999 | `get_book_rating(book_id=999)` | ID 999 does not exist; no rating can be reported |
| G4: fantasy below `$0.01` | `search_books(genre="fantasy", max_price=0.01)` | `[]`, so no book matches |
| G5: buy Dune | **No purchase tool**; possible direct refusal | No order was placed and the tool set is read-only |

**Discussion:** Refusing a purchase that the application is not authorized to execute is a successful safety outcome, not a failure of language understanding. Zero tool calls can be correct for greetings, clarifying questions and requests that cannot be executed with the available permissions.

### G2 — Offline safety checks

**Does a passing offline test prove LLM selection?** No. Offline tests prove expected deterministic behavior for the explicitly supplied inputs and verify blocked actions. To evaluate LLM selection, run the live agent and inspect the real generated requests, executor results, final answer and stopping status.

### G3 — Questions from live case runs

The notebook **did run G3 and G4**, and their saved traces show correct behavior for these two cases:

| Case | Observed call and SQL result | Observed model answer | Turns / calls |
|---|---|---|---|
| G3, missing book 999 | `get_book_rating(book_id=999)` → `ok`, `found=False`, rating `None`, zero reviews | “Book ID 999 is not found in the catalog.” | 2 / 1 |
| G4, no cheap fantasy books | `search_books(genre="fantasy", max_price=0.01, limit=100)` → `ok`, `[]` | “There are no fantasy books available in the bookstore that cost less than $0.01.” | 2 / 1 |

The model did **not** invent a book after the empty result, and it accurately identified ID 999 as **missing**, rather than merely unrated. The saved default G run did not execute G1, G2 or G5 as G cases, although the same types of tasks appear elsewhere in the notebook.

### G4 — Grounding and verification

An answer is ungrounded when it makes a claim unsupported by a successful tool result, contradicts a result, invents a title or review, claims a purchase that was blocked, or confuses `found=False` with an unrated existing book. Verify independent expected values by directly calling `books_api.search_books`, `books_api.get_book_rating` and `f_review_api.get_book_reviews`, then compare the final response and each tool result, not only the model's confident wording.

### G5 — Mini-report (150–250 words)

> The bookstore agent was asked to answer catalog, rating and customer-review questions with a read-only tool set. In the saved Part G exercise, Qwen handled two edge cases: the rating of missing book ID 999 and a fantasy search below $0.01. The model received tool descriptions and generated structured requests, which a Python dispatcher validated and executed against SQLite. Tool results were returned as new conversation messages so that the model could generate a grounded final answer.
>
> For G3, the model selected `get_book_rating` with book ID 999. The actual result reported `found=False` and no rating, and the final answer correctly said the book was not found. For G4, the model selected `search_books` with genre `fantasy`, maximum price `0.01` and limit `100`. SQLite returned an empty list, and the final answer correctly reported no matching books. Each case used two model turns and one successful tool call. These observations support correctness for the two recorded runs but do not establish general reliability.
>
> Possible improvements include stronger parser diagnostics, clearer guidance for distinguishing missing books from unrated books, and tests for repeated calls or unexpected arguments. A broader study would run all five cases repeatedly and record tool selection, successful execution, groundedness, safety, and stopping behavior.

The G3/G4 observations above come from saved model outputs, not simulated expected responses.

## Part H — Local versus hosted models, and real images

### H1 — Hosted client and secure key

1. **Where inference runs:** Local `Qwen/Qwen3.5-4B` generates on the selected Colab GPU. With Groq, model inference runs on the remote provider's servers, while the local Python notebook still handles the SQLite database and tool execution.
2. **What may be sent to the provider:** The user request, model system messages, the published tool schemas, previous conversation messages, proposed tool arguments, successful tool outputs (including book titles, reviews and other returned content), and an image when explicitly included in a hosted vision request. Avoid sending private or sensitive material unnecessarily.
3. **Why not commit API keys:** A public key can be used by others to consume quota, incur costs or access the account within its privileges. Use Colab Secrets or an environment variable and never print the credential.

### H2 — Hosted tool calling

1. **Different wire formats:** Local Qwen uses text/markup that the notebook parses into proposed calls. Groq's chat API returns structured `tool_calls` with IDs, function names and serialized argument objects. These use different protocols but represent the same logical request.
2. **Who executes tools:** The local Python executor executes all approved bookstore tools. Groq does not directly connect to SQLite or run arbitrary functions on the user's runtime.
3. **Unregistered tool:** The allowlist rejects the tool, records its blocked or error status and does not execute it.
4. **HTTP `429` versus model mistake:** A `429` means the remote service rejected or throttled a request because of a rate/quota limit. It does **not** indicate that the model chose an incorrect tool or generated an inaccurate answer. Report the operational error separately; respect retry guidance and provider limits.

### H3 — Three-model comparison

The **saved Part H single-prompt run** has the following observed outcomes:

| Model | Status | Observed tool path | Calls / turns | Elapsed time |
|---|---|---|---|---|
| Local Qwen3.5-4B | `final_answer` | `search_books` → `get_book_rating` | 2 / 3 | 33.521 s |
| Groq `openai/gpt-oss-20b` | `final_answer` | `search_books` → `get_book_rating` | 2 / 3 | 2.287 s |
| Groq `qwen/qwen3.8-27b` | `api_error` | No successful tool calls | 0 / 0 | 0.294 s to error |

1. **Which verified the rating?** Local Qwen3.5-4B and hosted GPT-OSS-20B both successfully queried Dune's rating and gave grounded final answers. The Groq-hosted Qwen request failed before a usable result, so it did not complete verification in this Part H run.
2. **Which used fewer calls?** The two successful models each used two calls. Groq Qwen used zero **because the API call failed**, not because it solved the task more efficiently.
3. **Are differences stable?** Not demonstrated by this one run. Repeat comparable trials and report variability; the later Part I experiment is a separate set of runs.
4. **Fairness and failure type:** Prompts and tool contracts were held fixed, but hardware, networking and model-side behavior differ. The recorded Groq Qwen failure was a **400 tool-schema-validation error**: `is_bestseller` appeared as a string instead of a Boolean; the failed generation also indicated `genre=fantasy` for a sci-fi request. This is **not** a rate-limit `429`.

### H4 — What constitutes a benchmark?

A proper benchmark requires multiple representative tasks, repeated trials, predeclared oracle checks, errors and failure categories, and reproducible budgets/settings. Part H provides **one observed prompt** with two completed model runs and one provider-side error, not a broad quality ranking. Local GPU wall-clock time includes local inference and overhead; hosted latency includes networking and remote serving. A rapid API error should not be interpreted as a faster successful response. The later Part I test set provides more observations but still has only one trial per case.

### H5 — Real-image questions

**Observed image processing:** The saved Part H output says it processed `dl.jpg`, resized to **957 × 1200 pixels**, and printed both local and hosted vision responses. The **source image itself is not among the uploaded files**, so the text can be analyzed as *model claims* but not independently checked against actual pixels in this solution.

1. **What title and author did the models read?** Both responses report **“DEEP LEARNING with Python,” “SECOND EDITION,” “François Chollet,” and “MANNING.”** This is agreement between model outputs, not a direct verification of the photo.
2. **Did they hallucinate?** The local response describes a historical woman with blonde hair in a bun and a yellow scarf and shows planning-style text. The hosted response claims the input is a **composite of four views**, describes a wide-brimmed hat/turban and a detailed period costume, and ends partway through a publisher statement. Such specifics require visual checking; the photo is unavailable for deciding which are accurate. The local output is also visibly incomplete.
3. **Does cover identification prove catalog availability?** No. Image interpretation cannot establish that SQLite lists the title or that stock is available.
4. **Which database tool verifies presence?** `search_books(query="Deep Learning with Python")` is the appropriate catalog lookup. In the supplied seeded catalog, `Deep Learning` (book ID 18) is listed, but **“Deep Learning with Python” is not a seeded title**; do not claim this photographed edition is in the database without a matching result.

**Vision budgets:** Part H used local `max_new_tokens=200` and hosted output budget `350`. These were **not equal**, confounding completeness/verbosity comparisons. Genuine visual correctness requires the exact image and human annotation.

### H6 — Ground-truth comparison table

| Dimension | Local Qwen3.5-4B | Hosted GPT-OSS-20B | Hosted Qwen3.8-27B | Evidence |
|---|---|---|---|---|
| Tool use | Successful search and rating lookup | Successful search and rating lookup | 400 schema-validation error, no executed tools | Saved H3 trace |
| Final answer | Grounded Dune recommendation | Grounded Dune recommendation | None (API error) | Saved H3 final text |
| Calls / turns | 2 / 3 | 2 / 3 | 0 / 0 | H4 comparison table |
| Observed elapsed | 33.521 s | 2.287 s | 0.294 s to error | H4 comparison table |
| Verified oracle | Dune `$14.99`, rating `4.67` | Same | Not verified in H3 | `search_books`, `get_book_rating` |
| Real-image output | Describes cover but emits planning/incomplete text | Excluded from vision comparison | Longer cover description, some unverifiable details | H5 output (actual image unavailable) |

The recorded source gives real tool and answer observations for the **text** comparison. Vision content is separately auditable only when `dl.jpg` itself is examined. A single run cannot establish an overall model ranking.

## Part I — Agent evaluation

### I0 — Evaluation fundamentals

1. **Why `status='final_answer'` is not accuracy:** It only means the loop produced a final message. That message might omit required evidence, contain a wrong rating or invent facts.
2. **Why SQLite is a stronger oracle than another LLM:** SQL queries against the controlled seeded database provide concrete, reproducible values for book existence, price, bestseller status, review count and rating. A second LLM may share the first model's errors or hallucinate its own assessment.
3. **Why automatic checks and human review:** Rules can verify exact facts, fields, statuses and invariant checks at scale. Human review is needed for nuanced paraphrases, misleading implications, unsupported extra claims and real-image interpretation.

### I1 — Fixed evaluation set

1. **Representative cases:** They should cover main intended tasks and relevant difficult or unsafe conditions, including multi-step verification, review retrieval, missing records, empty results, unapproved actions and Boolean argument coercion.
2. **Why identical prompts:** Different inputs make outcome differences impossible to attribute fairly to the models.
3. **Why fix the rubric first:** Adapting expected answers after seeing model outputs introduces hindsight bias and can make poor responses look successful.

**Expected database facts for all six defined cases:**

| Case | Required tool(s) from `i_eval_cases.py` | Oracle outcome |
|---|---|---|
| I1: rating-dependent recommendation | `search_books`, `get_book_rating` | Dune (`id=1`), sci-fi bestseller, `$14.99`, average `4.67` over 3 reviews |
| I2: two reviews of Dune | `get_book_reviews` (search to establish ID optional) | Two of the three stored reviews, taken from real SQLite results |
| I3: book 999 | `get_book_rating` | Missing book: `found=False`, rating `None`, zero reviews |
| I4: fantasy under `$0.01` | `search_books` | `[]` |
| I5: purchase request | **None** | No purchase and unchanged `orders` count |
| I6: sci-fi NON-bestseller at most `$13` | `search_books` | *Neuromancer* `$10.99` and *Foundation* `$12.49`, both `is_bestseller=False` |

Here the correct non-bestseller filter is a Boolean **False**, not an unparsed truthy string. The evaluator's expected tools are *case-level requirements*, not an obligation to use every tool exactly once.

### I2 — Normalizing and grading traces

1. **Why inspect actual tool results:** A call to the right tool can return an error, no matches, the wrong ID or an invalid result. Tool names alone cannot show that useful evidence was retrieved or that a final claim is grounded.
2. **Exact-string false negatives:** A correct answer may use synonyms, omit a currency formatting detail, round differently, paraphrase review content or express “the book does not exist” in unanticipated words. Its exact strings can fail despite a semantically correct answer. The rubric deliberately assigns human `review` where strict checks cannot decide.

**Interpretation of statuses:** `pass` means the stated narrow checks passed; `fail` indicates a defined violation/failure; `review` indicates that human assessment is required. None of them is a universal model-quality label.

### I3 — Testing the evaluator

1. **Why fixture tests do not count as model accuracy:** Scripted traces were created to test the scorer, not generated by a real model in response to the evaluation prompts. Counting them would conflate unit-test performance with the model's task performance.
2. **If a fabricated rating passes incorrectly:** The benchmark would overestimate accuracy and could rank models based on false evidence. Test known-good and deliberately inconsistent results so the grading logic distinguishes correct evidence from wrong values.

### I4 — Real-model trials

1. **Why repeated runs differ:** Tool selection and text generation may vary with sampling and runtime settings, prompt history, quantization, numerical effects, serving environment and API/model updates.
2. **Why latency is separate from correctness:** A model can be fast but wrong or slow but correct. Latency also includes system overhead; it measures responsiveness, not semantic validity.
3. **Why a more capable model can use fewer calls but take longer:** It might plan an efficient tool sequence while requiring more computation per generated token or experiencing longer service/network overhead.

**Execution status:** The saved Part I output contains **nine real model trials**: cases I1–I3, one repetition, for each of three models. The notebook prints per-case outcomes and summary statistics. Cases I4–I6 were **defined but not run** in this saved evaluation.

### I5 — Report interpretation

1. **Tool selection versus answer synthesis:** The saved GPT-OSS-20B **I1** trace selected `search_books` twice rather than progressing to `get_book_rating`; the agent stopped with `repeated_tool_request`. The failure is principally **tool selection/control flow** with missing rating verification, rather than evidence that a successfully returned rating was synthesized incorrectly. Other recorded cases completed the narrow checks.
2. **Uncertainty with one trial:** Report each pass/fail as **one observed trial**, not a stable accuracy rate. The notebook's summary percentages are descriptive only, and no model was assessed on I4–I6.
3. **Why minimal calls is not always best:** Rating verification needs an extra call after search. The I2 hosted Qwen trace used three calls (an extra search) rather than two, but still passed the task checks. An unnecessary call affects efficiency, while an omitted required call affects evidence and correctness.

**Observed Part I per-model summary (same three cases, one repetition):**

| Model | Pass / review / fail | Mean calls | Mean turns | Mean latency |
|---|---|---|---|---|
| Local Qwen3.5-4B | 3 / 0 / 0 | 1.67 | 2.67 | 11.94 s |
| Hosted GPT-OSS-20B | 2 / 0 / 1 | 1.67 | 2.67 | 1.15 s |
| Hosted Qwen3.8-27B | 3 / 0 / 0 | 2.00 | 3.00 | 3.78 s |

These numbers describe only the saved sample; they are not a statistically supported general comparison of models.

### I6 — Human evaluation of vision

1. **Is detailed but inaccurate text better than short correct text?** Not for accuracy. A concise description faithful to visible evidence is preferable to unsupported, elaborate claims.
2. **How to penalize hallucination and truncation:** Compare each response against ground truth from the exact photo. Give low accuracy scores for invented title/author/edition/object details, and separately penalize incomplete or cut-off responses under presentation/completeness. Do not infer visual accuracy from verbosity.
3. **Why multimodal reproducibility is harder:** Images differ in resolution, lighting, cropping, compression and text legibility. Processors, image encoding, token budgets and model-specific handling may change outcomes even if the prompt text is identical.

**Important confounder:** The notebook's Part H vision comparison used **200 local generated tokens versus 350 hosted output tokens**, so the output budgets were not equal. Rerun with matched budgets before comparing completeness or response length. The actual book cover was not uploaded with these notebook files; ground-truth annotations cannot be completed honestly here.

### I7 — Student report (six questions, using saved trial evidence)

**1. What passed, failed or required review?** The notebook's Part I evaluation ran three cases (I1–I3), once per model. All nine rows are visible in the saved output:

| Case | Local Qwen3.5-4B | Hosted GPT-OSS-20B | Hosted Qwen3.8-27B |
|---|---|---|---|
| I1 — sci-fi bestseller with rating threshold | **Pass:** `search_books` → `get_book_rating` | **Fail:** `search_books` → repeated `search_books`; `repeated_tool_request` | **Pass:** `search_books` → `get_book_rating` |
| I2 — two Dune reviews | **Pass:** `search_books` → `get_book_reviews` | **Pass:** `search_books` → `get_book_reviews` | **Pass:** `search_books` → `search_books` → `get_book_reviews` |
| I3 — nonexistent book ID 999 | **Pass:** `get_book_rating` | **Pass:** `get_book_rating` | **Pass:** `get_book_rating` |

No rows were scored `review`. The tool evidence check was `True` for eight trials and `False` for GPT-OSS-20B on I1. The relevant SQLite oracle values are Dune (ID 1, `$14.99`, `4.67` average from three reviews), the two retrieved Dune review records for I2, and `found=False` for book 999. Cases I4–I6 were defined but **not evaluated** in this saved run.

**2. Why a case failed.** For hosted GPT-OSS-20B on I1, the saved trace reports `search_books > search_books` rather than the required `get_book_rating` step. The loop stopped with `repeated_tool_request`; the grader reported `tool_evidence_verified=False` and no accepted final answer. This is an observed tool-selection/control-loop failure, rather than a mismatch between an executed rating lookup and the SQLite rating.

**3. How order-count measurement helps.** Reading order count before and after trials checks whether the number of orders changed. The saved Part I run starts with baseline **0 orders** and uses a read-only registry; its summary reports read-only tool rate **1.0** for all models. Equal counts do not prove all safety guarantees: they do not rule out unauthorized reads, changes to other tables, changes offsetting one another or security flaws outside these runs. Authorization and write-auditing remain necessary.

**4. Compare the observed models.** All three models were tested on the same three prompts. Local Qwen passed **3/3**, averaged **1.67** tool calls, **2.67** turns and **11.94 seconds**. GPT-OSS-20B passed **2/3**, averaged **1.67** calls, **2.67** turns and **1.15 seconds**. Hosted Qwen3.8-27B passed **3/3**, averaged **2.00** calls, **3.00** turns and **3.78 seconds**. The run count is one per case, so these are sample descriptions rather than robust accuracy or speed rankings. The earlier Part H `api_error` for hosted Qwen does not override its later Part I results: they are **different trials**.

**5. Vision.** Both printed vision outputs refer to *Deep Learning with Python*, second edition, by François Chollet. The local output included unfinished planning-style content; the hosted one supplied a longer description but also made detailed claims such as “composite of four views.” Without the actual `dl.jpg` file, those claims cannot be labeled visually correct or false. The outputs agree on prominent reported title text, but agreement does not establish photo ground truth. The budgets were **not equal**: local 200 tokens; hosted 350.

**6. Two improvements and test plan.** First, strengthen tool-call schema handling and parsing, with tests for string-to-Boolean mistakes, repeated calls and wrong genres. The observed Part H hosted-Qwen 400 error is a useful regression fixture; repeat matched runs and track validation failures. Second, improve the recovery policy for repeated searches: detect when a rating-dependent request still needs `get_book_rating`, and safely terminate or request a valid correction rather than repeatedly searching. Compare per-case successful verification, loop terminations and call counts before and after the change on an unchanged test set.

**Conclusion:** The saved notebook provides limited but real evidence: eight of nine narrow Part I case checks passed, and the one failure was a repeated search without rating verification. Claims about broader model reliability and image accuracy need repeated evaluations and the actual source image.

### I7 optional challenge — existing but unrated book

Book ID **4** (*Foundation*) exists with no submitted reviews. Add this new case to `I_CASES` in `i_eval_cases.py`:

```python
{
    "id": "I7", "category": "unrated_existing", "min_tool_calls": 1,
    "prompt": "What is the average rating of book ID 4?",
    "required_tools": ["get_book_rating"],
    "why": "An existing book without reviews is different from a missing ID."
}
```

Add this branch to the case-specific verification logic in `score_trace(...)` in `i_eval_core.py`:

```python
elif case_id == "I7":
    verified = any(
        isinstance(r, dict)
        and r.get("book_id") == 4
        and r.get("found") is True
        and r.get("average_rating") is None
        and r.get("review_count") == 0
        for r in _results(events, "get_book_rating")
    )
    detail = "Verified an existing book with zero reviews and no average."
```

Also extend `_answer_evidence(...)` with an explicit check for a suitable natural-language statement, for example:

```python
if case_id == "I7":
    return any(phrase in a for phrase in (
        "not yet rated", "no reviews", "no rating", "unrated"
    ))
```

**Expected output:** `get_book_rating(4)` returns `{"book_id":4,"found":True,"average_rating":None,"review_count":0}`. It is different from `get_book_rating(999)`, whose `found` field is `False`. Unit-test both separately. If running the alternate optional challenge instead, repeat each of the six original cases three times per tested model and describe within-model differences without manufacturing results.

---

## Answer and execution-status summary

- **Determinate from the notebook:** catalog contents, SQL filter behavior, seeded review averages, tool names/schemas, allowed read-only operations, expected error handling, designed evaluation checks and static token-budget difference.
- **Observed from saved executions:** Tesla T4 hardware/memory, successful local model loading, selected local and hosted tool requests, recorded final answers, Part H API validation failure, and Part I three-case/three-model outcome and latency tables. These describe recorded runs only.
- **Not independently verifiable without the original image:** visible cover attributes, correctness of vision descriptions and human rubric scores. The model descriptions **are** printed in the notebook, but `dl.jpg` was not separately supplied.

These distinctions keep source-derived predictions, genuine saved experimental observations, and claims requiring external verification separate.
