## Part 1: Demo 1 (Colab Walkthrough Using `Tiny-2.1`)

Run this after **Theory 1** to show every stage from Module 1 on the trained `Tiny-2.1` model (`366,528` parameters).

### Cell A — Load Model, Test Answers & Display Summary
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import train2_tf as t2

# Load pre-trained Tiny-2.1 (or your trained model_tf_40m.ckpt)
model = t2.load_model("/content/Tiny-2.1/dos/TINY.BIN")

# 1. Run sample Q&A predictions
questions = [
    "what is 2+2?",
    "what is the capital of france?",
    "what is the capital of mars?",  # out-of-distribution example
]
for q, a in t2.answer_many(model, questions).items():
    print(f"Q: {q}\nA: {a}\n")

# 2. Build model with a dummy input so Keras displays layer parameter counts
_ = model(tf.zeros((1, 10), dtype=tf.int32))
model.summary()
```

### Cell B — Trace Tokenization, Shift/Mask, and Embeddings on `"2+2?\n4\n"`
```python
ops = tf.keras.ops

q, a = "2+2?", "4"
full_text = q + "\n" + a + "\n"
ids = t2.encode(full_text)
mask = [0] * (len(q) + 1) + [1] * (len(ids) - len(q) - 1)

print("Characters:", [repr(c)[1:-1] for c in full_text])
print("Token IDs :", ids)
print("Inputs x  :", ids[:-1])
print("Targets y :", ids[1:])
print("Loss mask :", mask[1:])  # 0 = question (ignored), 1 = answer (graded)

# Look up token + position embeddings
idx = tf.constant([ids], dtype=tf.int32)  # (1, 7)
T = tf.shape(idx)[1]
tok_emb = ops.take(model.emb, idx, axis=0)
pos_emb = ops.take(model.pos, tf.range(T), axis=0)
x = tok_emb + pos_emb

print("Token emb shape:", tok_emb.shape)
print("Pos emb shape  :", pos_emb.shape)
print("Input X shape  :", x.shape)
```

### Cell C — Extract & Plot All 4 Trained Causal Attention Heads (Block 0)
*(This replaces the random-tensor heatmap in `w6_prep1.py` with the real weights from `Tiny-2.1` so students see all 4 heads working on a real question.)*

```python
prompt = "what is 2+2?\n"
p_ids = tf.constant([t2.encode(prompt)], dtype=tf.int32)
T_p = tf.shape(p_ids)[1]
chars = [repr(c)[1:-1] for c in prompt]

x_p = ops.take(model.emb, p_ids, axis=0) + ops.take(model.pos, tf.range(T_p), axis=0)
blk = model.blocks[0]
h = blk.n1(x_p)

sp = lambda t: tf.transpose(tf.reshape(t, (1, T_p, 4, 24)), (0, 2, 1, 3))
qh, kh = sp(blk.q(h)), sp(blk.k(h))
scores = tf.linalg.matmul(qh, kh, transpose_b=True) / (24 ** 0.5)
causal_mask = tf.linalg.band_part(tf.ones((T_p, T_p)), -1, 0)
attn = tf.nn.softmax(tf.where(causal_mask == 0, -1e9, scores), axis=-1)  # (1, 4, T_p, T_p)

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for head_idx, ax in enumerate(axes.flat):
    im = ax.imshow(attn[0, head_idx].numpy(), cmap="viridis", vmin=0.0)
    ax.set_title(f"Block 0 — Head {head_idx}")
    ax.set_xticks(range(len(chars))); ax.set_xticklabels(chars, fontsize=8)
    ax.set_yticks(range(len(chars))); ax.set_yticklabels(chars, fontsize=8)
    ax.set_xlabel("Key (looked at)")
    ax.set_ylabel("Query (looking)")
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
plt.suptitle(f"Causal Multi-Head Attention on '{prompt.strip()}'")
plt.show()
```

---

## Part 2: Visualization Specifications (In the Style of `ConvolutionDemo.tsx`)

Each specification below maps directly to your `.viz-content` / `.matrix-workspace` / `.explain-value` / `.viz-controls` pattern so you can implement them as interactive React components.

---

### Visualization 1: Tokenization, Next-Token Shift & Loss Mask
* **Concept covered:** Sections 1 & 2 (*Tokenization*, *Next-Token Prediction & Loss Masking*).
* **Props / Input data:** `text: string` (default `"2+2?\n4\n"`), split into `question` (`"2+2?"`) and `answer` (`"4"`).
* **Stepper state (`step`):** `0` to `T - 2` (`6` steps total for `"2+2?\n4\n"`). At step `active`:
  * **Workspace cards (`.matrix-workspace`):**
    1. **Card 1 — `Input Token IDs (x)`** (`1×6` grid): Characters `['2', '+', '2', '?', '\n', '4']` with their IDs `[20, 13, 20, 33, 1, 22]` shown below/inside cells. Cells `0..active` use `highlight` (prefix visible to the model); cells `> active` use `pending`.
    2. **Symbol:** `→`
    3. **Card 2 — `Target Token IDs (y)`** (`1×6` grid): Shifted characters `['+', '2', '?', '\n', '4', '\n']` with IDs `[13, 20, 33, 1, 22, 1]`. Cell `active` uses `active` (if mask = `1`) or `pending`/muted (if mask = `0`); cells `> active` show `'·'`.
    4. **Card 3 — `Loss Mask`** (`1×6` grid): `[0, 0, 0, 0, 1, 1]`. Cell `active` uses `kernel` when `1` (graded) or `pending` when `0` (ignored).
* **`.explain-value` box:**
  * `CURRENT COMPUTATION`: `Input prefix: "2+2?\n" → Predict next char: "4" (ID 22) | Loss mask = 1 (Graded)` (or `Loss mask = 0 (Question token: not graded)` for steps `0..3`).
  * `tiny-note`: `"Rule: ID(c) = ASCII(c) - 30 (ID 1 = newline / stop token)"`.
* **Controls:** `← Previous token`, `Next token →`, `Reset`, `active + 1 / 6`.

---

### Visualization 2a: Token Embedding ONLY
* **Concept covered:** Section 3 (*Token Embeddings and Positional Embeddings*).
* **Props / Input data:**
  * `tokens: string[]` = `['2', '+', '2', '?']` (`T = 4`, IDs `[20, 13, 20, 33]`)
  * Toy `embTable` (`4×3`, e.g. `ID 20 -> [1.0, 0.0, 0.2]`, `ID 13 -> [0.0, 1.0, 0.1]`, `ID 33 -> [0.3, 0.4, 0.9]`)
  * Toy `posTable` (`4×3`, e.g. `pos 0 -> [0.1, 0.2, 0.0]`, `pos 1 -> [0.2, -0.1, 0.1]`, `pos 2 -> [0.4, -0.1, 0.3]`, `pos 3 -> [-0.2, 0.3, 0.2]`)
* **Stepper state (`step`):** `0` to `3` (one row/token per step).
* **Workspace cards (`.matrix-workspace`):**
  1. **Card 1 — `Token Lookup emb[ID]`** (`4×3` grid, rows labeled `'2' (20)`, `'+' (13)`, `'2' (20)`, `'?' (33)`): Row `active` has `highlight`.
  2. **Symbol:** `+`
  3. **Card 2 — `Position Table pos[t]`** (`4×3` grid, rows labeled `pos 0..3`): Row `active` has `kernel`.
  4. **Symbol:** `=`
  5. **Card 3 — `Input Matrix X`** (`4×3` grid): Rows `<= active` show computed sums; row `active` has `active`; rows `> active` show `'·'` (`pending`).
* **`.explain-value` box:**
  * `CURRENT COMPUTATION`: e.g. at step `2` (`t=2`, `'2'`): `[1.0, 0.0, 0.2] + [0.4, -0.1, 0.3] = [1.4, -0.1, 0.5]`.
  * `tiny-note`: Notice visually that row `0` (`'2'`) and row `2` (`'2'`) started with identical `emb` rows, but now have different rows in `X`!
* **Controls:** `← Previous token`, `Next token →`, `Reset`, `active + 1 / 4`.

---


---

### Visualization 2b:  Position Embedding (Addition) ONLY
* **Concept covered:** Section 3 (*Token Embeddings and Positional Embeddings*).
* **Props / Input data:**
  * `tokens: string[]` = `['2', '+', '2', '?']` (`T = 4`, IDs `[20, 13, 20, 33]`)
  * Toy `embTable` (`4×3`, e.g. `ID 20 -> [1.0, 0.0, 0.2]`, `ID 13 -> [0.0, 1.0, 0.1]`, `ID 33 -> [0.3, 0.4, 0.9]`)
  * Toy `posTable` (`4×3`, e.g. `pos 0 -> [0.1, 0.2, 0.0]`, `pos 1 -> [0.2, -0.1, 0.1]`, `pos 2 -> [0.4, -0.1, 0.3]`, `pos 3 -> [-0.2, 0.3, 0.2]`)
* **Stepper state (`step`):** `0` to `3` (one row/token per step).
* **Workspace cards (`.matrix-workspace`):**
  1. **Card 1 — `Token Lookup emb[ID]`** (`4×3` grid, rows labeled `'2' (20)`, `'+' (13)`, `'2' (20)`, `'?' (33)`): Row `active` has `highlight`.
  2. **Symbol:** `+`
  3. **Card 2 — `Position Table pos[t]`** (`4×3` grid, rows labeled `pos 0..3`): Row `active` has `kernel`.
  4. **Symbol:** `=`
  5. **Card 3 — `Input Matrix X`** (`4×3` grid): Rows `<= active` show computed sums; row `active` has `active`; rows `> active` show `'·'` (`pending`).
* **`.explain-value` box:**
  * `CURRENT COMPUTATION`: e.g. at step `2` (`t=2`, `'2'`): `[1.0, 0.0, 0.2] + [0.4, -0.1, 0.3] = [1.4, -0.1, 0.5]`.
  * `tiny-note`: Notice visually that row `0` (`'2'`) and row `2` (`'2'`) started with identical `emb` rows, but now have different rows in `X`!
* **Controls:** `← Previous token`, `Next token →`, `Reset`, `active + 1 / 4`.

---

### Visualization 3: Self-Attention Without Weights (Scores → Softmax → Context)
* **Concept covered:** Section 4 (*Self-Attention Without Trainable Weights*).
* **Props / Input data:** `X: number[][]` (`4×3` or `6×3` from `"Your journey starts with one step"` or `"2+2?"`), `labels: string[]`.
* **Two-stage mode toggle (or sequential step progression):**
  * **Tab / Mode 1: `1. Scores (X · Xᵀ)`** (direct parallel to `ConvolutionDemo.tsx`!)
    * **Stepper (`step`):** `0` to `T * T - 1`, where `r = Math.floor(active / T)` (query row) and `c = active % T` (key row).
    * **Workspace cards:**
      1. `Queries X` (`T×3`): row `r` has `highlight`.
      2. Symbol: `·`
      3. `Keys Xᵀ` (`3×T`): column `c` has `kernel`.
      4. Symbol: `=`
      5. `Scores` (`T×T`): cells `<= active` filled, cell `(r, c)` has `active`, future cells `'·'` (`pending`).
    * **`.explain-value`:** `x[r] · x[c] = (0.43 × 0.55) + (0.15 × 0.87) + (0.89 × 0.66) = 0.95`.
  * **Tab / Mode 2: `2. Softmax → Context (Weights · X)`**
    * **Stepper (`step`):** `0` to `T - 1` (one query row `r` per step).
    * **Workspace cards:**
      1. `Weights[r]` (`1×T` softmax row): `kernel` styling, showing probabilities summing to `1.00`.
      2. Symbol: `×`
      3. `Inputs X` (`T×3`): all rows `highlight` (intensity or badge showing weight %).
      4. Symbol: `→`
      5. `Context Matrix Z` (`T×3`): rows `<= active` filled (`active` on row `r`), rows `> active` `'·'`.
    * **`.explain-value`:** `0.14×x₀ + 0.24×x₁ + 0.23×x₂ + ... = [0.44, 0.65, 0.57]`.

---

### Visualization 4: Trainable $Q, K, V$ + Scaled Dot-Product + Causal Mask Toggle
* **Concept covered:** Sections 5 & 6 (*Self-Attention with $Q, K, V$*, *Scaling by $\sqrt{d_k}$*, and *Causal Masking*).
* **Props / Input data:** Small `X` (`4×2` or `4×4`) and toy `Wq, Wk, Wv` (`2×2` or `4×2`), producing precomputed `Q, K, V` matrices (`4×2`, $d_k = 2$).
* **Interactive controls:**
  * **Toggle switch:** `Causal Mask: OFF` / `Causal Mask: ON`
  * **Stepper (`step`):** `0` to `T - 1` (selecting active **Query token row `r`**).
* **Workspace cards (`.matrix-workspace`):**
  1. **Card 1 — `Q` (`4×2`):** Row `r` highlighted (`highlight`).
  2. **Symbol:** `× Kᵀ / √2 →`
  3. **Card 2 — `Scaled Scores + Mask` (`4×4`):** Row `r` shows raw scores for `c <= r` (`highlight`), and for `c > r` shows either the raw score (when Mask `OFF`) or `−∞` (`pending` / crossed-out cell when Mask `ON`).
  4. **Symbol:** `softmax →`
  5. **Card 3 — `Attention Weights` (`4×4`):** Color-coded probability cells (darker background for higher probability). When Mask `ON`, cells `(r, c)` with `c > r` display `0.00` in muted `pending` style.
  6. **Symbol:** `× V →`
  7. **Card 4 — `Context[r]` (`1×2`):** Resulting blended `Value` vector for token `r` (`active`).
* **`.explain-value` box:**
  * Line 1 (`Scaled Scores`): `Row r before mask: [1.20, 2.00, 3.50, 0.80]`
  * Line 2 (`After Mask`): When ON at `r=1`, `[1.20, 2.00, -∞, -∞]` → `Softmax: [0.31, 0.69, 0.00, 0.00] (sum = 1.00)`.
  * `tiny-note`: `"Causal rule: query i can only attend to keys j <= i (scaled by 1 / √d_k)"`.

---

### Visualization 5: Multi-Head Attention Split & Concatenation (`H = 4` Heads)
* **Concept covered:** Section 7 (*Multi-Head Attention* in `Tiny-2.1`).
* **Props / Input data:** Real or toy slice for `"what is 2+2?\n"` (`T = 13` or shortened `T = 6` `"2+2?\n4"`), with precomputed `4×(T×T)` attention weight matrices (e.g., extracted directly from Cell C of Demo 1!) and slice labels showing `D = 96`, `H = 4`, `d_k = 24`.
* **Interactive controls:**
  * **Head selector buttons:** `Head 0`, `Head 1`, `Head 2`, `Head 3` (or `All Heads` overview).
  * **Token stepper (`step`):** `0` to `T - 1` (selecting which query character is currently predicting/attending).
* **Workspace cards (`.matrix-workspace`):**
  1. **Card 1 — `Full Vector Split (D=96)`:** A horizontal 4-segment bar `[0..23 | 24..47 | 48..71 | 72..95]`, highlighting the active head's 24-dim slice (`kernel`).
  2. **Card 2 — `Head h Attention Matrix (T×T)`:** Displays the causal attention grid for the selected `Head h` (with row `active` highlighted, upper triangle masked as `0.00`, and cells shaded by weight).
  3. **Card 3 — `Concatenated Output [H0 | H1 | H2 | H3] → Wo`:** Shows the 4 head context vectors rejoining into `(T×96)` before passing through `Wo`.
* **`.explain-value` box:**
  * `CURRENT COMPUTATION`: `"Query token '\n' (pos 4) in Head 1 attends most to: '2' (0.41), '+' (0.33), '2' (0.18)"` (dynamically reading the top-2/3 attended keys in row `active` of `Head h`!).
  * `tiny-note`: `"Tiny-2.1 runs 4 heads of size d_k = 24 in parallel (4 × 24 = 96), then mixes them with linear layer Wo."`.

---

### Suggested Conversion Order into React (`.tsx`)
If you are building these alongside your `ConvolutionDemo.tsx`, this order reuses the most code:
1. Build **Viz 3 (Stage 1)** first—it is almost an exact structural twin of `ConvolutionDemo.tsx` (`Matrix` + `highlight` + `kernel` + `active` + `pending` + `terms.join(' + ')`).
2. Add the **Mask toggle** and **Softmax row** to turn it into **Viz 4**.
3. Reuse the `1×T` row grid for **Viz 1** and row-wise `Matrix` addition for **Viz 2**.
4. Reuse the `T×T` grid from Viz 4 inside a 4-tab wrapper for **Viz 5** (feeding real JSON weights exported from Demo 1 Cell C!).