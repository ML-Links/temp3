# Teaching flow and additional demo recommendations

These are proposed teaching designs, not additional required content or completed new demos. Durations are estimates to adapt to the actual class length and students' calculus/Python background.

## Suggested sequence

| Stage | Approximate time | Student outcome |
|---|---:|---|
| Brief recap | 5 minutes | Identify inputs, target, parameters, prediction, and loss |
| Demo 1: one weight | 10–15 minutes | Predict gradient sign and trace one complete update |
| Activity 2: two weights | 15–20 minutes | Compute and interpret separate partial derivatives |
| Debrief and bias extension | 5–10 minutes | Use a shared error and simultaneous gradients for three parameters |
| Activity 3: binary classification | 20–25 minutes | Trace sigmoid/BCE and explain the sign for either label |
| SGD extension | 8–12 minutes | Explain why batch choice changes updates while the chain rule stays the same |
| Architecture comparison | 5–8 minutes | Explain why a deep MLP is also a DNN |
| Exit check | 3–5 minutes | Distinguish gradients, optimizer, batch size, and depth |

This is roughly 70–100 minutes before breaks and setup. For a shorter class, move bias experiments, detailed BCE algebra, learning-rate stability derivations, and the architecture comparison to follow-up. Do not attempt every code listing and plot live.

After class, assign the core theory as consolidation. Give a navigation map: Sections 3–4 for derivatives, 7 for BCE, 9–11 for updates/learning rate, 12–13 for SGD, 14 for depth, and 15 for autodiff. Mark `4-a`, the cleaned `4-b`, and `4-c` as optional focused readings. If all four theory files must be compulsory, specify distinct reading sections to avoid repeating the same derivations.

## Make Activity 2 active

Place this before the worked calculations:

> Work individually for 4 minutes, then compare with a partner. Use `x1=2`, `x2=1`, `y=6`, `w1=3`, `w2=1`, and learning rate `0.1`. Calculate the prediction, error, loss, both partial derivatives, both updated weights, and the new loss. State which values are held constant in each partial derivative. Reveal the solution only after attempting all steps.

Completion check: `7`, `1`, `0.5`, gradients `(2,1)`, new weights `(2.8,0.9)`, new loss `0.125`.

Transfer task:

> Restart at the original weights, but change `x2` to `-1`. Predict the signs of both gradients, then calculate one update. Explain why the two weights now move in different directions.

Answer key: prediction `5`, error `-1`, loss `0.5`, gradients `(-2,+1)`, new weights `(3.2,0.9)`, new prediction `5.5`, loss `0.125`. This checks interpretation rather than copying the original calculation.

Optional bias task: restart at `(w1,w2,b)=(3,1,1)` and calculate all three gradients before updating anything. Ask why one observation cannot determine a unique three-parameter solution.

## Make Activity 3 active

Keep the first natural-log BCE derivation guided. Then ask:

> Starting from zero weights and bias with `x=(2,1)`, calculate one update for `y=1`. Restart and repeat for `y=0`. Submit a two-column table of logit, probability, loss, gradients, updated parameters, and new probability. Explain why the loss initially agrees but the gradient signs differ.

Checks: both initial losses are approximately `0.693147`; gradients are `(-1,-0.5,-0.5)` versus `(1,0.5,0.5)`; updated probabilities are approximately `0.574443` versus `0.425557`.

Before showing the learning-rate plot, ask learners to predict which curve is likely to oscillate. Afterward, require one observation about the whole curve and one reason the final loss is insufficient. Explicitly label that experiment full-batch GD and correct its finite-optimum discussion as described in the review.

## Should there be an SGD demo?

**Yes: this is the more valuable addition.** The current code repeatedly uses one fixed observation or the full six-example dataset. Neither implementation shows selecting a new random example or mini-batch across updates. A short extension closes that gap without introducing another model or loss.

Use Activity 3's six observations, sigmoid, and BCE. For the comparison, reset the same initial parameters for every run and use plain SGD with no momentum. Compare batch sizes **6, 2, and 1**. Shuffle a common seeded order once per epoch and use that order for both smaller-batch runs. Recompute predictions and gradients at the current parameters after every update; average gradients within the current batch.

Display:

- Batch indices, batch size, and update count for one epoch.
- A parameter trajectory or compact update table.
- **Full-dataset mean BCE** at epoch boundaries for all methods. Comparing each run's current batch loss would mix different objectives on the plot.
- A second view against update count if useful, labelled clearly: equal epochs mean comparable data passes; equal updates do not mean equal data exposure or wall-clock cost.

For six examples, expect **1, 3, and 6 updates per epoch**. Ask students to predict these counts before running. Keep the loss convention, initial parameters, and learning rate fixed for the first illustration, then explain that learning rates may need separate tuning for a performance comparison. More frequent updates do not establish that one method is faster in elapsed time.

Use two seeds or a second shuffle to illustrate variability; do not promise that a small dataset will always produce dramatic noisy curves. Include a checkpoint that compares batch gradients at the **same** parameter values before showing evolving training trajectories.

The existing four-gradient example in `4-c` is a good 2-minute warm-up: all methods start at `w=10`, yet give `9.9`, `9.7`, or `10.2`. Clarify that those frozen gradients cannot be reused for later updates.

## Should there be an MLP/DNN demo using a Keras visualizer?

**A short visual comparison is worthwhile; a separate long training demo is unnecessary for the terminology objective.** MLP and DNN are overlapping descriptions. Avoid presenting them as rival algorithms or implying that a deeper model must perform better.

Show two fully connected models on the same two-feature input:

| Model | Architecture | Trainable parameters including biases | Description |
|---|---|---:|---|
| A | `2 → 4 → 1`, nonlinear hidden activation, sigmoid output | `3*4 + 5*1 = 17` | Shallow MLP |
| B | `2 → 4 → 4 → 4 → 1`, nonlinear hidden activations, sigmoid output | `12 + 20 + 20 + 5 = 57` | Deep MLP; also a DNN |

The arrow diagrams count feature dimensions; the input is not a trainable Dense layer. State that these are illustrative shallow/deep examples, not a universal numerical definition of “deep.” Show a deep CNN or Transformer schematic only as a contrasting architecture: it can be a DNN without being an MLP.

Ask learners to identify hidden layers, count parameters, and explain what happens if all hidden activations are removed. Expected answer: successive affine maps collapse to one affine map, so depth alone does not add the intended nonlinear representation power.

[Keras's official `plot_model` utility](https://keras.io/api/utils/model_plotting_utils/) can generate an architecture image with shapes, layer names, and activation labels. It is useful for connecting model code to layer structure. It is an architecture diagram, not an animation of gradient flow or an SGD trajectory. If “Keras visualiser” means a separate third-party package, the exact package was not specified; do not make that package a lesson dependency without checking its capabilities and setup. Prepare exported diagrams in advance and provide them in the handout.

If the aim expands to demonstrate **why nonlinear hidden units help**, add a small XOR comparison between a sigmoid neuron and an MLP. A shallow nonlinear MLP can already solve XOR, so XOR does not prove that a deep network is necessary. A trained shallow/deep comparison also changes parameter count and optimization difficulty; it cannot isolate depth by itself.

## Release checklist

1. Correct the quasi-separation claim, remove the duplicate chapter, and align example values.
2. Remove instructor scripts and editing history; resolve missing companion references.
3. Add attempt-first tasks, expected outputs, prerequisites, and required/optional labels.
4. Provide a runnable environment and visible plot exports; verify Markdown math, details blocks, and Mermaid in the actual student platform.
5. Check slide links and numbering against the distributed deck, then run the NumPy and framework examples in the course environment.
