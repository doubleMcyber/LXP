# ARCHITECTURE_SPEC.md: Continuous Latent Collaboration Framework

## 1. System-Level Tensor Flow
The objective of this architecture is to transition multi-agent communication from a discrete token space into a continuous, high-dimensional vector space. The tensor flow across the agent boundary is strictly continuous until the final solver agent decodes the action.

Let $\mathcal{A}_1$ be the sender agent (Planner/Reasoner) and $\mathcal{A}_2$ be the receiver agent (Critic/Solver).
1. $\mathcal{A}_1$ encodes the initial prompt sequence into input embeddings $E = [e_1, e_2, \dots, e_T] \in \mathbb{R}^{B \times T \times d_{model}}$.
2. $\mathcal{A}_1$'s reasoning module $M_\phi$ integrates over the continuous latent space to generate a compressed thought sequence $H_K \in \mathbb{R}^{B \times K \times d_{model}}$.
3. The `AlignmentAdapter` applies an orthogonal linear transformation $Q$ to map $H_K$ from the output manifold of $\mathcal{A}_1$ to the valid input manifold of $\mathcal{A}_2$, yielding $E_{aligned} = H_K Q$.
4. $\mathcal{A}_2$'s `LatentWorkingMemory` ingests the layer-wise Key-Value (KV) cache generated during $\mathcal{A}_1$'s latent loop, concatenating it to its own attention tensors to preserve the complete computational history losslessly.

---

## 2. Core Class Boundaries & Interfaces

### 2.1. `LatentCompressor` (The Sender Module)
Replaces the standard language model head (`lm_head`). Instead of projecting to a vocabulary size $\mathbb{R}^{|V|}$ and applying an argmax/softmax, this module retains the tensor in the hidden dimension $\mathbb{R}^{d_{model}}$ and feeds it auto-regressively—or continuously—back into the model.

* **Continuous Dynamics (Neural ODE Formulation):** To bypass the 80-step latent degradation plateau, the reasoning trajectory is modeled as a continuous vector field. Rather than discrete jumps $h_{t+1} = f(h_t)$, the compressor defines the derivative of the hidden state with respect to computational "time":
    $$\frac{dh}{dt} = f_\phi(h(t), t)$$
    where $f_\phi$ represents the transformer block. An ODE solver (e.g., Runge-Kutta 4) integrates this over the desired reasoning depth $K$.
* **Input Tensor:** $E \in \mathbb{R}^{B \times T \times d_{model}}$
* **Output Tensor:** $H_K \in \mathbb{R}^{B \times K \times d_{model}}$ (The compressed continuous thought).

### 2.2. `AlignmentAdapter` (The Geometric Bridge)
If $\mathcal{A}_1$ and $\mathcal{A}_2$ are heterogeneous (e.g., Qwen and LLaMA), their latent vector spaces possess different topologies. To align them without distorting the internal geometry of the thought vector, we strictly bypass standard dense feed-forward networks (which warp the space) and implement an **Orthogonal Procrustes** solver.

* **Mathematical Formulation:** Let $S \in \mathbb{R}^{N \times d}$ be a set of anchor states from $\mathcal{A}_1$ and $R \in \mathbb{R}^{N \times d}$ be the target states in $\mathcal{A}_2$. We seek an orthogonal matrix $Q \in \mathbb{R}^{d \times d}$ (where $Q^T Q = I$) that minimizes the Frobenius norm $||SQ - R||_F$.
* **Implementation Computation:**
    1. Compute the cross-covariance matrix: $C = S^T R$
    2. Perform Singular Value Decomposition (SVD): $U \Sigma V^T = C$
    3. Construct the orthogonal mapping: $Q = U V^T$
* **Forward Pass:** $E_{aligned} = H_K \times Q$
* **Tensor Shapes:** Input $H_K \in \mathbb{R}^{B \times K \times d_{sender}}$. 
    Output $E_{aligned} \in \mathbb{R}^{B \times K \times d_{receiver}}$. 
    *(Note: If $d_{sender} \neq d_{receiver}$, a unitary padding or strictly isometric projection matrix must precede $Q$).*

### 2.3. `LatentWorkingMemory` (The Cross-Agent Cache)
To ensure lossless information transfer, the internal state of the attention mechanism is transferred. This acts as the shared working memory across the multi-agent system.

* **Data Structure:** A tuple of length $L$ (number of transformer layers), where each element contains the Key and Value tensors.
    $$\mathcal{M}_{\mathcal{A}_1} = \left( \left(K^{(1)}, V^{(1)}\right), \dots, \left(K^{(L)}, V^{(L)}\right) \right)$$
* **Tensor Shapes:** Each $K^{(l)}$ and $V^{(l)}$ is of shape $\mathbb{R}^{B \times H_{num} \times T_{cache} \times d_{head}}$.
* **Injection Mechanism:** During the initialization of $\mathcal{A}_2$'s forward pass, $\mathcal{M}_{\mathcal{A}_1}$ is passed via the standard HuggingFace `past_key_values` argument. 
* **Attention Concatenation:** For any new query $Q_{\mathcal{A}_2}$ in $\mathcal{A}_2$, the attention is computed over the union of the history:
    $$Attention(Q_{\mathcal{A}_2}, [K_{\mathcal{A}_1}; K_{\mathcal{A}_2}], [V_{\mathcal{A}_1}; V_{\mathcal{A}_2}])$$
* **State-Space Model (SSM) Contingency:** For near-infinite latent horizons, the $O(N^2)$ attention mechanism mapping $[K_{\mathcal{A}_1}; K_{\mathcal{A}_2}]$ will eventually exhaust VRAM. The architecture should isolate this component so that the Transformer KV cache can be hot-swapped for an SSM (e.g., Mamba) recurrent hidden state matrix $h_t \in \mathbb{R}^{B \times D \times N}$ in subsequent phases.