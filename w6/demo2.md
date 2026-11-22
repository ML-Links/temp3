# Module 2 — Demo 2: Full Transformer Trace, Text Generation, Training Loss, and Compression

## Part 1: Standalone Colab Demo (PyTorch + GitHub Repository)

The cells below run standalone in Google Colab using the `WollyMonjac/Tiny-2.1` PyTorch repository.

### Cell 1 — Clone the Repository and Enter the Training Directory
```python
!git clone https://github.com/WollyMonjac/Tiny-2.1.git
%cd /content/Tiny-2.1/training
```

### Cell 2 — Imports and Load the Pre-Trained `TINY.BIN` (`TNY2`) Model
```python
import math, os, random, struct, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import train2

def load_v2_bin(path):
    """Load a version-2 ('TNY2') 8-bit weight file into train2.Model."""
    b = open(path, "rb").read()
    assert b[:4] == b"TNY2", "Expected a TNY2 weight file"
    D, L, H, HID = struct.unpack_from("<4h", b, 4)
    off = [12]

    def table(r, c):
        w = np.empty((r, c), dtype=np.float32)
        for i in range(r):
            s = struct.unpack_from("<f", b, off[0])[0]
            w[i] = np.frombuffer(b, np.int8, c, off[0] + 4).astype(np.float32) * s
            off[0] += 4 + c
        return torch.tensor(w)

    def gains():
        g = np.frombuffer(b, "<f4", D, off[0]).copy()
        off[0] += 4 * D
        return torch.tensor(g)

    m = train2.Model(train2.Cfg(D, L, H, HID))
    with torch.no_grad():
        m.emb.weight.copy_(table(train2.VOCAB, D))
        m.pos.weight.copy_(table(train2.CTX, D))
        for blk in m.blocks:
            blk.n1.g.copy_(gains())
            for lin in (blk.q, blk.k, blk.v, blk.o):
                lin.weight.copy_(table(D, D))
            blk.n2.g.copy_(gains())
            blk.f1.weight.copy_(table(HID, D))
            blk.f2.weight.copy_(table(D, HID))
        m.nf.g.copy_(gains())
    assert off[0] == len(b), "size mismatch"
    return m

model = load_v2_bin("/content/Tiny-2.1/dos/TINY.BIN")
model.eval()
print("Loaded:", model.c, "| Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
```

### Cell 3 — Trace One Complete Transformer Block (`RMSNorm`, Residuals, `ReLU` MLP)
This cell traces `"2+2?\n"` through `Block 0`, prints tensor shapes and RMS norms, measures how many `ReLU` neurons in the MLP are active (`> 0`), and plots the parameter distribution alongside the MLP neuron activations.

```python
prompt = "2+2?\n"
ids = torch.tensor([train2.encode(prompt)], dtype=torch.long)
B, T = ids.shape

with torch.no_grad():
    x_in = model.emb(ids) + model.pos(torch.arange(T))
    blk = model.blocks[0]

    # 1. RMSNorm before attention
    h1 = blk.n1(x_in)
    raw_rms_after = (h1 / blk.n1.g).pow(2).mean(-1).sqrt()
    print("Input RMS (before n1):", np.round(x_in.pow(2).mean(-1).sqrt()[0].numpy(), 3))
    print("Normalized RMS (h1/g):", np.round(raw_rms_after[0].numpy(), 3))

    # 2. Multi-head causal attention + first residual connection
    sp = lambda t: t.view(B, T, blk.c.H, blk.c.D // blk.c.H).transpose(1, 2)
    a = F.scaled_dot_product_attention(sp(blk.q(h1)), sp(blk.k(h1)), sp(blk.v(h1)), is_causal=True)
    attn_out = blk.o(a.transpose(1, 2).reshape(B, T, blk.c.D))
    x_mid = x_in + attn_out

    # 3. Second RMSNorm + ReLU MLP (96 -> 256 -> 96) + second residual connection
    h2 = blk.n2(x_mid)
    mlp_hidden = F.relu(blk.f1(h2))   # 256 ReLU neurons per token
    mlp_out = blk.f2(mlp_hidden)
    x_out = x_mid + mlp_out

active_frac = (mlp_hidden[0, -1] > 0).float().mean().item()
print(f"MLP hidden shape: {tuple(mlp_hidden.shape)} | Active ReLU neurons at '\\n': {100*active_frac:.1f}%")
print(f"Block output shape: {tuple(x_out.shape)}")

# Plot 1: Parameter distribution vs. ReLU neuron sparsity
emb_p  = model.emb.weight.numel() + model.pos.weight.numel()
attn_p = sum(getattr(b, n).weight.numel() for b in model.blocks for n in ("q", "k", "v", "o"))
mlp_p  = sum(getattr(b, n).weight.numel() for b in model.blocks for n in ("f1", "f2"))
norm_p = sum(p.numel() for p in model.parameters() if p.dim() == 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(["Emb+Pos", "Attention\n(Q,K,V,O)", "MLP\n(f1, f2)", "RMSNorm\n(g)"],
            [emb_p, attn_p, mlp_p, norm_p], color=["#9ecae1", "#4292c6", "#08519c", "#31a354"])
axes[0].set_ylabel("Parameters")
axes[0].set_title("Parameter Distribution Across Tiny-2.1 (366,528 total)")

axes[1].hist(mlp_hidden[0, -1].numpy(), bins=25, color="#08519c", edgecolor="white")
axes[1].set_xlabel("Neuron activation value after ReLU")
axes[1].set_ylabel("Neuron count (out of 256)")
axes[1].set_title("Block 0 MLP Neuron Activations at '\\n'")
plt.tight_layout()
plt.show()
```

### Cell 4 — Stack All 4 Blocks, Apply Final Norm & Weight Tying, and Inspect Next-Token Probabilities
This cell passes `"what is 2+2?\n"` through all 4 blocks, applies `model.nf` and weight-tied multiplication `nf(x) @ emb.weight.t()`, verifies exact agreement with `model(p_ids)`, and plots the top-8 next-character probabilities.

```python
q_prompt = "what is 2+2?\n"
p_ids = torch.tensor([train2.encode(q_prompt)], dtype=torch.long)

with torch.no_grad():
    x = model.emb(p_ids) + model.pos(torch.arange(p_ids.shape[1]))
    for blk in model.blocks:
        x = blk(x)
    x_norm = model.nf(x)
    logits_manual = x_norm @ model.emb.weight.t()   # weight-tied output: (1, T, 97)
    logits_model = model(p_ids)
    print("Max diff vs model(p_ids):", (logits_manual - logits_model).abs().max().item())

    probs = F.softmax(logits_manual[0, -1], dim=-1)  # probabilities at last position
    top_vals, top_ids = torch.topk(probs, 8)

labels = [repr(train2.decode([int(i)]))[1:-1] if i > 0 else "<pad>" for i in top_ids]
vals = top_vals.numpy()

plt.figure(figsize=(7, 4))
plt.barh(labels[::-1], vals[::-1], color="#2b8cbe")
plt.xlabel("Probability")
plt.title(f"Top-8 Next-Character Predictions After {repr(q_prompt)}")
plt.xlim(0, 1.0)
for idx_bar, v in enumerate(vals[::-1]):
    plt.text(v + 0.015, idx_bar, f"{v:.3f}", va="center")
plt.tight_layout()
plt.show()
```

### Cell 5 — Autoregressive Text Generation Trace (Character by Character)
This cell runs the greedy decoding loop step-by-step on `"what is the capital of france?\n"`, logging every generated character and its probability until ID `1` (`\n`) stops generation.

```python
gen_prompt = "what is the capital of france?\n"
x = torch.tensor([train2.encode(gen_prompt)], dtype=torch.long)
generated = []

print(f"Prompt: {repr(gen_prompt)}")
with torch.no_grad():
    for step in range(40):
        logits = model(x)[:, -1, :]                  # (1, 97) at final position
        probs = F.softmax(logits, dim=-1)
        nxt = int(logits.argmax(dim=-1).item())
        prob = float(probs[0, nxt].item())
        ch_repr = repr(train2.decode([nxt]))[1:-1]
        print(f"  step {step+1:2d}: predicted {ch_repr:>4} (ID {nxt:2d}, prob={prob:.3f})")
        if nxt <= 1:
            break
        generated.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], dtype=torch.long)], dim=1)

print("Final decoded answer:", train2.decode(generated))
```

### Cell 6 — Short Training Loss Curve and 8-Bit Weight Quantization Check
This cell demonstrates initial cross-entropy loss (`ln(97) ≈ 4.575`) on a fresh untrained model, runs 120 training steps on `gen2.py` batches to plot the loss drop, and compares float32 vs. 8-bit `TINY.BIN` file size and rounding error.

```python
train2._POOL.clear()
rng = random.Random(0)
fresh_model = train2.Model(train2.Cfg())
opt = torch.optim.AdamW(fresh_model.parameters(), lr=2e-3, weight_decay=0.01)

losses = []
for step in range(120):
    xb, mb = train2.batch(rng, 32)
    loss = train2.loss_fn(fresh_model, xb, mb)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(fresh_model.parameters(), 1.0)
    opt.step()
    losses.append(loss.item())

print(f"Step 1 loss: {losses[0]:.3f} (random guess ln(97) = {math.log(97):.3f}) -> Step 120 loss: {losses[-1]:.3f}")

# Compare float32 checkpoint size vs 8-bit exported TINY.BIN size and rounding error
train2.save_model(model, "demo_float.pt")
bin_bytes = train2.export(model, "demo_int8.bin")
pt_bytes = os.path.getsize("demo_float.pt")
print(f"Float32 checkpoint: {pt_bytes:,} bytes | 8-bit TINY.BIN: {bin_bytes:,} bytes ({pt_bytes/bin_bytes:.2f}x smaller)")

q_model = train2.quantized_copy(model)
w_float = model.blocks[0].f1.weight.detach().numpy()
w_int8  = q_model.blocks[0].f1.weight.detach().numpy()
diff = (w_float - w_int8).ravel()
print(f"Max quantization error on Block 0 f1.weight: {np.abs(diff).max():.5f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(losses, color="#08519c")
axes[0].axhline(math.log(97), color="gray", linestyle="--", label="Random guess ln(97)=4.57")
axes[0].set_xlabel("Training step")
axes[0].set_ylabel("Cross-entropy loss")
axes[0].set_title("Early Training Loss Drop (120 Steps)")
axes[0].legend()

axes[1].hist(diff, bins=30, color="#31a354", edgecolor="white")
axes[1].set_xlabel("Weight rounding error (float32 - int8_restored)")
axes[1].set_ylabel("Weight count")
axes[1].set_title("8-Bit Row-Wise Quantization Error (Block 0 f1)")
plt.tight_layout()
plt.show()
```

---

## Part 2: Visualization Descriptions (For `ConvolutionDemo.tsx`-Style React Components)

Each specification below follows the `.viz-content`, `.matrix-workspace`, `.explain-value` (`CURRENT COMPUTATION` + `tiny-note`), and `.viz-controls` (`← Previous`, `Next →`, `Reset`, `step / total`) structure of `ConvolutionDemo.tsx`.

---

### Visualization 6: RMSNorm Stepper (`Norm`)
* **Concept covered:** Module 2, Section 1 (*Normalizing Activations*).
* **Props / Input data:**
  * `inputVector: number[]` (default `[1.0, 7.0, 1.0, 7.0]`)
  * `gain: number[]` (default `[1.0, 2.0, 1.0, 0.5]`)
* **Interactive controls:**
  * **Toggle switch:** `Scale input ×10` (`[1, 7, 1, 7]` ↔ `[10, 70, 10, 70]`).
  * **Stepper (`step`):** `0` to `3` (`4` sequential stages).
* **Workspace cards (`.matrix-workspace`):**
  1. **Card 1 — `Input x` (`1×4`):** Displays current vector; cells use `highlight`.
  2. **Card 2 — `Squared x²` (`1×4`):** Filled at `step >= 0` (`active` at `step === 0`, `'·'` otherwise), accompanied by displayed scalar `Mean` and `RMS`.
  3. **Card 3 — `Normalized x / RMS` (`1×4`):** Filled at `step >= 2` (`active` at `step === 2`). Notice this card remains `[0.2, 1.4, 0.2, 1.4]` even when `Scale input ×10` is toggled ON.
  4. **Symbol:** `⊙`
  5. **Card 4 — `Gain g` (`1×4`):** Styled with `kernel`.
  6. **Symbol:** `=`
  7. **Card 5 — `Output` (`1×4`):** Filled at `step === 3` (`active`), `'·'` (`pending`) before `step === 3`.
* **`.explain-value` box:**
  * Step 0: `Squares: [1², 7², 1², 7²] = [1, 49, 1, 49]`
  * Step 1: `Mean = (1 + 49 + 1 + 49) / 4 = 25.0 → RMS = √25.0 = 5.0`
  * Step 2: `x / RMS = [1/5, 7/5, 1/5, 7/5] = [0.20, 1.40, 0.20, 1.40]`
  * Step 3: `Output = [0.20×1.0, 1.40×2.0, 0.20×1.0, 1.40×0.5] = [0.20, 2.80, 0.20, 0.70]`
  * `tiny-note`: `"Toggling ×10 leaves x / RMS unchanged, preventing activation drift across layers."`

---

### Visualization 7: Residual Shortcut vs. Plain Chain
* **Concept covered:** Module 2, Section 2 (*Shortcut / Residual Connections*).
* **Props / Input data:**
  * `initialValue: number` (`1.0`)
  * `deltas: number[]` (`[+0.05, -0.02, +0.08, +0.04]` for 4 layers)
  * `decayFactors: number[]` (`[0.5, 0.5, 0.5, 0.5]` for non-residual comparison)
* **Interactive controls:**
  * **Toggle switch:** `Residual Shortcut: ON (x + delta)` / `OFF (x × factor)`.
  * **Stepper (`step`):** `0` to `3` (representing `Block 0` through `Block 3`).
* **Workspace cards (`.matrix-workspace`):**
  * Horizontal chain of 5 `1×1` matrix cards (`Input`, `Block 0`, `Block 1`, `Block 2`, `Block 3`).
  * Cards `<= active + 1` show computed signal values (`active` on the current block, `highlight` on earlier blocks); unreached blocks show `'·'` (`pending`).
  * When `Residual Shortcut: ON`, a second `1×4` card labeled `Sublayer Delta` highlights the small delta added at `active` (`kernel`), visually separating the preserved base signal from the incremental delta.
* **`.explain-value` box:**
  * Residual ON (e.g. Block 2): `x_out = x_in + delta[2] = 1.03 + 0.08 = 1.11`
  * Residual OFF (e.g. Block 2): `x_out = x_in × 0.5 = 0.25 × 0.5 = 0.125`
  * `tiny-note`: `"With shortcuts, signals and gradients travel across 4 blocks without shrinking toward zero."`

---

### Visualization 8: Feed-Forward MLP (`f1 → ReLU → f2`)
* **Concept covered:** Module 2, Section 3 (*Feed-Forward Network / ReLU Neurons*).
* **Props / Input data:**
  * `h: number[]` (`1×2`, e.g. `[1.0, -1.0]`)
  * `W1: number[][]` (`4×2`, 4 hidden neurons)
  * `W2: number[][]` (`2×4`, output projection back to 2D)
* **Stepper state (`step`):** `0` to `5` (`6` steps total):
  * Steps `0..3`: Computing each hidden neuron `0..3` (`h · W1[i]`, followed by `ReLU`).
  * Steps `4..5`: Computing output dimensions `0` and `1` (`relu_hidden · W2[j]`).
* **Workspace cards (`.matrix-workspace`):**
  1. **Card 1 — `Input h` (`1×2`):** Highlighted during steps `0..3`.
  2. **Card 2 — `W1ᵀ` (`2×4`):** Column `active` styled as `kernel` during steps `0..3`.
  3. **Card 3 — `Hidden Pre-ReLU` (`1×4`):** Fills neuron-by-neuron; cell `active` highlighted.
  4. **Card 4 — `After ReLU` (`1×4`):** Positive activations stay highlighted; negative values turn to `0` (`pending` style to indicate inactive neurons).
  5. **Card 5 — `W2ᵀ` (`4×2`) & `MLP Output` (`1×2`):** Active during steps `4..5`, filling the final `1×2` vector.
* **`.explain-value` box:**
  * Steps `0..3` (e.g. Neuron 1): `Neuron 1: (1.0 × -1.0) + (-1.0 × 2.0) = -3.0 → ReLU(-3.0) = 0.0 (inactive)`
  * Steps `4..5`: Explicit dot product across the 4 post-ReLU values and `W2` row, e.g. `(2.0 × 0.3) + (0.0 × 0.5) + (0.5 × 0.4) + (0.0 × -0.2) = 0.80`.
  * `tiny-note`: `"In Tiny-2.1, this layer expands 96 → 256 ReLU neurons → 96 independently at each token position."`

---

### Visualization 9: Weight-Tied Logits and Autoregressive Generation Loop
* **Concept covered:** Module 2, Sections 5 & 6 (*Weight Tying* and *Autoregressive Text Generation*).
* **Props / Input data:**
  * `initialPrompt: string` (`"2+2?\n"`)
  * `stepsData`: Precomputed array of generation steps, where each step holds:
    * `currentIds`: sequence so far
    * `lastVec`: `1×3` toy slice representing `nf(x)[-1]`
    * `candChars`: top 5 candidate tokens (`['4', '5', '2', '\n', ' ']`) with their `emb` rows (`5×3`) and resulting `logits` + `softmax` probabilities
    * `chosenChar`: predicted token (`'4'` on step 0, `'\n'` on step 1)
* **Stepper state (`step`):** `0` to `stepsData.length - 1` (`Step 1 / 2`: predict `'4'`; `Step 2 / 2`: predict `'\n'` → Stop).
* **Workspace cards (`.matrix-workspace`):**
  1. **Card 1 — `Prompt Sequence` (`1×T`):** Shows characters accumulated so far; newly appended token marked `active`; final token (the position making the next prediction) marked `highlight`.
  2. **Card 2 — `Final Vector nf(x)[-1]` (`1×3`):** Styled `highlight`.
  3. **Symbol:** `×`
  4. **Card 3 — `embᵀ (Top-5 Candidates)` (`3×5`):** Columns labeled `'4'`, `'5'`, `'2'`, `'\n'`, `' '`. Winning column styled `kernel`.
  5. **Symbol:** `→`
  6. **Card 4 — `Probabilities` (`1×5`):** Softmax percentages for the 5 candidates; highest entry marked `active`.
* **`.explain-value` box:**
  * `CURRENT COMPUTATION`: `nf(x)[-1] · emb['4'] = (0.8×0.9) + (0.6×0.7) + (0.2×0.5) = 1.24 (highest logit) → Softmax prob = 0.88 → Append '4'` (and on the final step: `Predicted ID = 1 ('\n') → Stop generation`).
  * `tiny-note`: `"Weight tying reuses the input embedding matrix emb in reverse to score all 97 vocabulary characters."`

---

### Visualization 10: 8-Bit Row-Wise Quantization (`TINY.BIN` Compression)
* **Concept covered:** Module 2, Section 7 (*8-Bit Quantization*).
* **Props / Input data:**
  * `rowFloat: number[]` (`1×4`, default `[0.00, 0.50, -1.00, 0.25]`)
* **Stepper state (`step`):** `0` to `3` (one weight column `c` per step).
* **Workspace cards (`.matrix-workspace`):**
  1. **Card 1 — `Original Float32 Row` (`1×4`, 16 bytes):** Column `active` highlighted (`highlight`).
  2. **Card 2 — `Scale s` (`1×1`, 4 bytes):** Fixed scalar `max(|row|) / 127 = 1.00 / 127 = 0.007874` (`kernel`).
  3. **Card 3 — `Stored Int8` (`1×4`, 4 bytes):** Fills column-by-column (`[0, 64, -127, 32]`; `active` on current column, `'·'` for `> active`).
  4. **Card 4 — `Restored Float` (`1×4`):** Shows `int8 × s` (`[0.000, 0.504, -1.000, 0.252]`).
  5. **Card 5 — `Rounding Error` (`1×4`):** Shows `|original - restored|` (`[0.000, 0.004, 0.000, 0.002]`).
* **`.explain-value` box:**
  * Example at `step = 1`: `w[1] = 0.50 → round(0.50 / 0.007874) = 64 (1-byte int8) → restored = 64 × 0.007874 = 0.504 (error = 0.004)`
  * `tiny-note`: `"For 96 weights per row in Tiny-2.1: 384 bytes (float32) shrinks to 4 + 96 = 100 bytes (3.84× smaller)."`