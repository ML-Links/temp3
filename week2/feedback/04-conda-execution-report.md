# Conda execution report

Executed all 22 Python blocks independently using the existing conda environment `G:\AI-ML-Teacher\AI\DH-ML\.conda\dhml-lab-test`, without changing that environment. Matplotlib used the noninteractive Agg backend; each `show()` saved the generated figures instead of opening a window.

## Results

- **18 passed** runtime execution.
- **0 failed** after their imports succeeded.
- **4 blocked** by missing dependencies: core theory block 4 (PyTorch), and reading 4-a blocks 1–3 (PyTorch, TensorFlow, JAX).

The main theory/demo/activity files contain 19 blocks, of which 18 ran successfully. Reading 4-a contains another 3 framework blocks. Missing dependencies are not evidence that the blocked examples are incorrect; their runtime behavior remains unverified.

## Environment setup limitation

Attempting to create the authorized isolated conda environment encountered a refused connection through the session's configured network proxy and a denied write to conda's external Terms of Service cache. An offline attempt also encountered a Windows permission error in conda's virtual-package discovery. No installed environment was modified. This session does not allow sandbox approval escalation, so user authorization alone cannot enable dependency downloads here.

## Evidence

- [Machine-readable results, captured output, warnings, and plot filenames](python-execution/results.json).
- [Reusable execution runner](test_python_blocks.py).
- Plot PNGs are in `feedback/python-execution`.

The runner executes each block in a fresh namespace, checks for runtime exceptions, and saves figures. It does not automatically prove all explanatory or mathematical claims, or verify rendering in the student Markdown platform.

To finish the four framework checks from an unrestricted terminal, create the intended environment with Python 3.11 and install NumPy, Matplotlib, CPU PyTorch, TensorFlow, and JAX; then run this runner through `conda run --prefix .conda/lecture-test python feedback/test_python_blocks.py`.
