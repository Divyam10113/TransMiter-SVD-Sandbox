# TransMiter SVD Alignment Sandbox

A minimal PyTorch implementation verifying the core mathematical claims of the AAAI paper: *Transferable Model-agnostic Vision-Language Model Adaptation for Efficient Weak-to-Strong Generalization* (TransMiter).

### Objective
This repository isolates and tests the **Forward-only Adapter Transfer** mechanism (Section 4.2). TransMiter claims to bypass expensive backpropagation when transferring knowledge to a strong VLM by utilizing a closed-form Orthogonal Procrustes solution. This sandbox independently verifies that tensor geometry.

### The Architecture
1. **Prediction-Based Adapter:** A standard `[LayerNorm -> Linear (Expansion) -> GELU -> Linear (Compression)]` residual block used to extract the knowledge gap.
2. **Procrustes Alignment:** Solves for the optimal transition matrix $\hat{W}$ without gradient descent using Singular Value Decomposition:
   $$M = H_{strong}^T \cdot H_{weak}$$
   $$U, \Sigma, V^T = \text{SVD}(M)$$
   $$\hat{W} = U \cdot V^T$$

### Execution
Running `python transmiter.py` simulates the extraction on synthetic logits (`[Batch: 32, Dim: 1024]`), proving the mathematical distance drops instantly upon applying the orthogonal matrix.

```text
Initial Distance between models: 1.9843
Distance after SVD Alignment: 0.0167
Alignment successful. Zero backpropagation required.
