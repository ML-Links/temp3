# Detailed review of the seven files

Line numbers refer to the reviewed source files. Priority: **High** = correct before student release; **Medium** = materially improves learning or clarity; **Low** = polish or optional extension.

## 0 — How neural networks learn: post-class theory

**Overall:** Accurate and unusually careful about common misconceptions. The half-squared-error convention, natural-log BCE, gradient sign, learning-rate stability interval, finite differences, and separation of autodiff from optimization are explained correctly. Retain the distinction between training loss and generalization.

- **High — Sections 16.2–16.3 do not recap the actual activities.** Section 16.2 uses `(x1,x2,y)=(2,3,10)` and parameters `(1,2,1)`; Activity 2 uses `(2,1,6)`, initially `(3,1)` without bias and then `(3,1,1)` with bias. Section 16.3 starts at `(0.2,-0.4,0)`; Activity 3 starts at `(0,0,0)`. Both classification setups happen to give the same initial logit and subsequent probabilities, but different parameter values. Replace these summaries with the activity values, or explicitly label them as additional examples. For the bias activity, the actual result is `(2.6,0.8,0.8)`, loss `2 → 0.32`; for classification it is `(0.1,0.05,0.05)`, loss approximately `0.693147 → 0.554355`.
- **Medium — Reading load and duplication.** At roughly 1,000 lines, this chapter already covers almost everything in `4-a` and `4-c`. Make this the required consolidation reading and mark those files as optional deep dives. A section map would let readers return to their difficulty without rereading the whole chapter.
- **Medium — Order of definitions.** Section 3 differentiates a squared-error loss before Section 5 defines loss and Section 6 explains this particular choice. Add one sentence before the derivative example defining the penalty, with a forward link to Section 6. Introduce the basic chain-rule prerequisite there too.
- **Medium — Architecture bridge, Section 14.** Explicitly state that stacking affine layers without nonlinear activations collapses to a single affine transformation. This explains why adding layers alone does not create the intended expressive advantage.
- **Medium — Final source section, line 985 onward.** Two named files do not exist under those exact names: `4-a-ordinary_...` and `4-c-SGD.md` omit `misc-`. Use actual relative Markdown links. Replace the editing explanation about “supplied note sets” and “source-material integration” with a short student reading guide.
- **Low — Gradient terminology, Section 8.** “A gradient may be positive, negative, or zero” is appropriate for the scalar case but ambiguous for a vector. Prefer “A scalar derivative or an individual gradient component may be positive, negative, or zero.”
- **Low — Update counts.** Some plot loops record `steps+1` states and also perform an extra unused update after the final recorded state. The plotted values are correct; guard the update with `if t < steps` to make the code match the stated count.

## 1 — Single-neuron gradient-descent demo

**Overall:** Strong worked demo. The initial update, repeated-update table, derivative, optimum, and convergence condition `0 < eta < 0.5` are correct. The code keeps loss and gradients tied to the current weight.

- **Medium — Too much for one live walkthrough.** Use Sections 1–5 and the minimal simulation in class; make the longer hill discussion, multiple plot implementations, and alternative losses optional follow-up. Otherwise repetition can obscure the short training cycle being taught.
- **Medium — Add prediction pauses.** Before revealing the gradient ask “Should the weight increase or decrease?” Before the update ask for the new weight and loss. These give the instructor evidence that students follow the sign and scale rather than simply copy arithmetic.
- **Medium — First runnable-code instructions.** State that the arithmetic simulation uses basic Python, while plots need NumPy and Matplotlib. Provide a prepared notebook or environment and show the plots as images as well as code; collapsed code is not a visible visualization.
- **Low — Table and print alignment.** Section 5's table uses completed-update counts and the simulation prints the state before each update. Both are valid and align at step 0, but explicitly state “Printed step k is the state after k completed updates.” The final update is not printed by that loop.
- **Low — Closing sentence.** “Update the parameters in directions that reduce the loss” should say “aim to reduce the loss,” or retain the sufficiently-small-step qualification. Earlier sections correctly explain that a large step can increase loss.

## 2 — Two-weight partial-derivative activity

**Overall:** Numerically and conceptually sound. The gradients `(2,1)`, first update `(2.8,0.9)`, loss `0.125`, limiting pair `(2.6,0.8)`, and specific stability interval `0 < eta < 0.4` are correct. The bias calculation and its loss `0.32` are also correct. The note about nonunique solutions and the caveat on the bias comparison are valuable.

- **High — Activity identity.** The header calls this a “Worked lesson” and its objectives say “demonstration.” Almost every result is supplied before students need to produce it. Keep the explanations as solutions, but put an attempt-first task block before Sections 2–5; move the worked answers into reveal blocks or a separate solution section.
- **Medium — Define what to submit.** Ask for prediction, error, loss, both partial derivatives, updated weights, and a sentence explaining why the other weight is held fixed. Include a modified-input transfer question rather than only reproducing the worked numbers.
- **Medium — Clarify simultaneous updates.** State directly that both gradients must be evaluated at the same old parameter vector. The implementation is correct because `error` is computed before either assignment, even where assignments occur on successive lines. Learners may otherwise recompute the second gradient after updating the first weight.
- **Medium — Manage optional material.** Make the bias extension and learning-rate recurrence a second stage or extension. Deriving the stability condition while also introducing partial derivatives can overload students without prior multivariable calculus.
- **Low — Architecture language.** “No activation function” can be clarified as “identity/linear output, with no nonlinear activation.” This makes the connection to the activation language in the next activity smoother.

## 3 — Binary-classification activity

**Overall:** The sigmoid/BCE chain rule, both label cases, update, and iteration table are correct. The warning about zero initialization in hidden networks, stable BCE calculation, thresholding, and the full-batch/SGD distinction are helpful.

- **High — No finite optimum for the six-example dataset, Section 9.** The statement after Plot 5 suggests that the conflicting labels resolve the finite-optimum issue. They do not for this dataset. Consider changing the parameters by `t*(1,1,-2)` for any positive `t`. The logits of `(0,0)`, `(1,0)`, and `(0,1)` decrease by `2t`, `t`, and `t`; all three labels are 0, so their losses strictly decrease at every finite parameter value. The logits of `(1,1)`, `(2,0)`, and `(0,2)` stay unchanged because `x1+x2=2`. Their losses therefore stay unchanged. Thus there is always a direction of strict improvement, so no finite parameter vector minimizes the unregularized mean BCE. This is quasi-separation, despite the absence of a perfectly separating boundary.

  Minimal replacement: “No single linear boundary can classify all six examples perfectly. This experiment demonstrates learning-rate stability, but it does not guarantee a finite unregularized optimum: some parameters can keep growing while loss decreases.” Keep the existing experiment; its oscillation conclusion is valid. If a finite optimum is a teaching requirement, add an explicit positive L2 penalty on **all three parameters**, include its gradients, and rerun/relabel the experiment. That changes the objective and should not be introduced silently.
- **High — Missing companion reading.** Lines 34, 131, 165, and 906 refer to an unlinked companion, heart-disease context, and base-2 tables. No such file is among the seven supplied files; the provided theory instead uses abstract examples and natural logs throughout. If another reading is intended, name and link it. Otherwise remove these references and keep the standalone statement “All logarithms here are natural logarithms.” Its underlying gradient explanation is correct.
- **High — Activity identity.** The title says “Demo 3” and the opening says “Worked lesson,” although this is assigned as an activity. Rename it “Activity 3” and give learners a task before the solutions. Reserve the full sigmoid/BCE derivation as a guided section rather than requiring every student to discover it unaided.
- **Medium — State this is logistic regression.** This sigmoid neuron has no hidden layers and is a binary logistic-regression model. Naming it prevents learners from thinking this is already an MLP or a deep network.
- **Medium — Learning objective scope.** “A binary classifier produces a probability” is too broad: some classifiers output scores or only decisions. Say “This binary classifier produces an estimated probability using sigmoid.”
- **Medium — Step counts in plots.** Plots 3–5 include an unused update after the last recorded point. Recorded histories and the “after 80 updates” result are correct; guard the final update for clarity.
- **Medium — Numeric interpretation verified.** Independently reproduced losses after 80 updates are approximately `0.627827`, `0.445239`, `0.359038`, and `0.641674` for learning rates `0.05`, `0.5`, `2`, and `20`. The last run has 50 loss increases and peaks near `5.912705`. Keep the instruction to inspect the whole curve.
- **Low — Closing source-alignment paragraph.** Remove editorial commentary about what was extended or converted. Retain a normal reference if the actual source is available.

## 4-a — Derivatives and autodiff reading

**Overall:** Technically careful and broadly consistent with official framework explanations. The finite-difference evaluation counts, distinction between `jax.grad` and `jax.jit`, PyTorch accumulation, and TensorFlow tape/optimizer separation are appropriate. The three framework examples use the same correct arithmetic.

- **High — Instructor/editor content remains.** Replace “Teaching point” at lines 59, 127, and 210 with “Key idea.” Replace “What to emphasize” at lines 359, 391, and 422 with “What to notice.” Rename “Lecture phrasing” at line 463 to “Putting the ideas together.” Rename “Summary for students” at line 486 to “Summary.” Remove “Suggested explanation to say aloud” at line 506 and its script, or convert it into a concise explanation directly addressed to the reader. Remove the entire “Corrections to the original draft” section at line 512; it exposes editing history and problems that no longer exist in the released text.
- **Medium — Reduce compulsory scope.** Mark Section 9 (LLM memory/distributed training) as optional enrichment. FSDP, ZeRO, optimizer state, and checkpointing are a substantial jump from partial derivatives and are not prerequisites for the activities.
- **Medium — Framework choice.** Use the framework taught in this course for the main runnable example; retain the other two as optional comparisons. Students should not infer that three installations are required.
- **Low — Qualification.** Some common activations are nondifferentiable at isolated points; frameworks use defined derivative conventions there. A brief optional note is sufficient. Avoid expanding the introductory chapter into a full treatment of nonsmooth optimization.

Checked against [PyTorch autograd tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html), [TensorFlow autodiff guide](https://www.tensorflow.org/guide/autodiff), and [JAX automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html). These sources support the framework interface distinctions; runtime execution of those snippets was not performed.

## 4-b — MLP and DNN reading

**Overall:** The first 54 lines correctly distinguish architecture from depth. It explicitly avoids the misleading claims that MLPs cannot learn features or that every DNN must use a different layer type.

- **High — Appended duplicate, line 58 onward.** Remove the SGD chapter from this file and link to `4-c`. It is unrelated to the title and repeats the separate reading almost in full.
- **Medium — Put the relationship first.** Lead with “MLP describes an architecture; deep describes its depth. A deep MLP is both an MLP and a DNN.” Two architecture drawings will teach this more directly than six sections comparing overlapping categories.
- **Medium — Include nonlinear activations explicitly.** The definition should mention nonlinear hidden activations and explain why a stack of purely affine layers is still affine.
- **Medium — Avoid guaranteed complexity claims.** “Allowing the network to learn increasingly complex representations” should be “can support hierarchical representations.” More layers do not guarantee better learned representations or accuracy.
- **Low — Clarify layer counting.** Count hidden layers separately from the input and output. State that the examples illustrate shallow/deep usage without defining a universal cutoff.

## 4-c — SGD reading

**Overall:** Correct update equations and numerical comparisons. It appropriately distinguishes full-batch, one-example SGD, and mini-batch usage, and avoids promising automatic speedups.

- **Medium — Update size, line 182.** “10,000 smaller updates” is unjustified: smaller batches do not necessarily produce smaller parameter steps. Use “10,000 parameter updates, each based on 100 observations.” The same issue appears in the appended copy in `4-b`.
- **Medium — Randomness, line 268.** The mismatch between batch and full-data gradients is not itself what creates randomness. Random sampling or shuffling makes the selected batch, and therefore its gradient, random. Replace with: “Random batch selection makes the gradient estimate vary between updates; an individual estimate can differ from the full-data gradient.”
- **Medium — Abrupt ending.** Section 7 is titled “Why Approximate Gradients Can Be Useful” but ends with limitations of full-batch GD. Complete it with a balanced paragraph: cheaper updates and useful hardware throughput; variable estimates and possible instability; noise does not guarantee escape from bad regions. Add two checks on batch size and update counts.
- **Medium — Sampling convention.** Distinguish random sampling with replacement from a shuffled pass without replacement. One epoch as “every observation once” assumes the latter. The existing count table already makes that assumption, which is good.
- **Low — Nondivisible datasets.** Add the main theory's ceiling/floor qualification for the last batch. Example: 1,003 examples with batch size 100 gives 11 updates when the final batch is kept.
- **Low — Slide references.** References to slides 90–93 are fine if students receive that exact deck. Add a deck link/title; their numbering could not be verified from the supplied material.

## Student-facing voice across the set

There is no blanket problem with using **you**, **we**, or **students**. Direct instructions such as “Calculate the new loss” and collaborative explanations such as “we calculate a partial derivative” are suitable for student materials. The issue is whether the text addresses the learner or tells an instructor how to teach.

Most prose uses a consistent explanatory voice. The clear audience leaks are concentrated in `4-a`, together with source-editing history in the theory and Activity 3. “Summary for students” is mildly distancing rather than technically wrong; “What to emphasize” and “Suggested explanation to say aloud” are much stronger instructor signals.

Use one consistent naming system: **Demo 1**, **Activity 2**, **Activity 3**, **Core theory**, and **Optional reading A/B/C**. Keep learning-rate notation consistent within each file; `alpha` and `eta` are both valid and the theory already explains the equivalence. Use clickable links between prerequisites, activities, and supporting sections instead of filename references alone.
