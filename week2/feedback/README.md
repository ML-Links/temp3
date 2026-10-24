# Lecture review

Reviewed all seven supplied Markdown files on 24 October 2026. This is the original review. The seven teaching files were subsequently revised in place at your request; see [Applied revisions](03-applied-revisions.md). Findings below describe the version originally reviewed.

The mathematical progression is strong: one weight → multiple weights → sigmoid and BCE, with a consistent distinction between calculating gradients and updating parameters. The material is substantially usable, but needs a few corrections and better separation between reference reading, demonstrations, and student activities.

## Highest priorities

1. **Correct Activity 3's finite-optimum discussion.** The six-example dataset is quasi-separated: despite not being perfectly linearly classifiable, its unregularized logistic loss has no finite minimizer. The learning-rate experiment itself remains useful. See the proof and replacement wording in the detailed review.
2. **Remove the appended SGD chapter from `4-b`.** Its architecture explanation ends at line 54; another full chapter begins at line 58 and duplicates `4-c`.
3. **Align the theory's demonstration summaries with the actual activities.** Sections 16.2 and 16.3 use different initial values. These are valid calculations but misleading summaries of the supplied activities.
4. **Remove instructor and editing language from `4-a`.** In particular, “What to emphasize,” “Suggested explanation to say aloud,” and “Corrections to the original draft” do not belong in the student release.
5. **Make files 2 and 3 actual activities.** Both currently expose almost all calculations as worked lessons. Add short attempt-before-reveal tasks with deliverables and completion criteria.

## Additional demos

Interpreting “SDG” in the request as **SGD**: add a short multi-example SGD comparison, preferably as an extension of Activity 3. The present material explains SGD but never actually runs stochastic or mini-batch updates.

For **MLP and DNN**, a short architecture comparison is sufficient; a separate full training demo is optional. Show a shallow MLP and a deep MLP and ask why both are MLPs while the second is also a DNN. A Keras diagram can support this, but does not demonstrate gradient noise or training dynamics.

## Feedback files

- [Detailed review](01-detailed-review.md): accuracy, teaching design, language, and file-specific changes.
- [Flow and demo recommendations](02-flow-and-demo-recommendations.md): suggested lesson sequence, concrete activity prompts, and proposed SGD/architecture demonstrations.
- [Validation results](validation-results.json): syntax checks and independently reproduced numerical results.

## Verification limits

All 22 Python code blocks passed Python syntax checks. Four blocks using basic Python were executed successfully. The four learning-rate runs in Activity 3 were independently reproduced using standard-library arithmetic. NumPy is unavailable in this environment, so the NumPy/Matplotlib and PyTorch/TensorFlow/JAX snippets were reviewed but not executed. Plot rendering, Mermaid rendering, and the cited lecture slides/unsupplied companion reading were not verified. Official framework documentation was consulted; links appear alongside the relevant feedback.
