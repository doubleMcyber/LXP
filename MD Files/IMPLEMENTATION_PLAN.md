# IMPLEMENTATION_PLAN.md: Phased Build for the Continuous Latent MAS

## Introduction
This document outlines the step-by-step implementation strategy for the hybrid Latent Multi-Agent System. To manage complexity and ensure stability, the build is divided into four strict phases. Each phase must be fully implemented, tested, and validated against its specific success criteria before progressing to the next. 

[Image of a software development implementation roadmap]


---

## Phase 1: Zero-Shot Latent Loop & KV Cache Transfer
**Objective:** Establish the baseline infrastructure for passing hidden states and attention caches between two identical models (homogeneous setup) without any compression or complex alignment.

**Action Items:**
1. **Model Initialization:** Load two instances of `Qwen2.5-7B-Base` (Agent A and Agent B) using HuggingFace Transformers.
2. **Hidden State Interception:** Modify the generation loop of Agent A to bypass the `lm_head`. Extract the final transformer layer's continuous hidden states $H \in \mathbb{R}^{T \times d}$.
3. **KV Cache Extraction:** Hook into the forward pass of Agent A to capture the layer-wise `past_key_values` tensor tuple.
4. **Basic Alignment:** Implement the ridge regression closed-form matrix ($W_a$) to project the output state back to the input embedding space.
5. **Cross-Agent Injection:** Pass the aligned embedding $e = hW_a$ and the extracted KV cache into the forward pass of Agent B. 

**Validation Gate 1:** Agent B must successfully generate coherent text conditional on Agent A's latent thoughts without encountering tensor dimension mismatches or CUDA out-of-memory errors.

---

## Phase 2: Compression Training Pipeline (The Reasoner)
**Objective:** Train a reasoning module ($M_\phi$) to compress long latent reasoning trajectories into a strict, predefined budget (e.g., 8 to 16 latent steps) while maintaining task utility.

**Action Items:**
1. **Dataset Preparation:** Load the ALFWorld and MATH trajectory datasets. Extract expert Chain-of-Thought (CoT) text plans to serve as the baseline for the preference and geometric losses.
2. **Freeze the Actor:** Lock all parameters of the downstream actor model ($A_\theta$).
3. **Loss Function Implementation:** * Code the Cross-Entropy task utility loss ($\mathcal{L}_{task}$).
    * Code the Uncertainty-Weighted Agreement loss using KL divergence ($\mathcal{L}_{pref}$).
    * Code the Latent Direction Alignment loss using step-averaged cosine similarity ($\mathcal{L}_{geom}$).
4. **Training Loop:** Set up a mixed-precision (bfloat16) training loop using DeepSpeed ZeRO-2 to optimize $M_\phi$ over the composite loss function.

**Validation Gate 2:** The trained reasoning agent must compress a 128-step reasoning trajectory into 16 steps, and the frozen actor must maintain at least 90% of its uncompressed baseline accuracy on the MATH validation set.

---

## Phase 3: Advanced Geometric Alignment (Orthogonal Procrustes)
**Objective:** Enable heterogeneous agent communication (e.g., Qwen-7B to LLaMA3-8B) by replacing the heuristic ridge regression with a strict geometric rotation.

**Action Items:**
1. **Dimension Padding:** If the sender and receiver have different hidden dimensions (e.g., 3584 vs 4096), implement an isometric padding or truncation utility.
2. **SVD Solver:** Write a PyTorch utility block to compute the Singular Value Decomposition ($U \Sigma V^T$) of the cross-covariance matrix between the sender and receiver's representation spaces.
3. **Orthogonal Mapping:** Construct the Orthogonal Procrustes matrix $Q = U V^T$.
4. **Adapter Integration:** Replace the ridge regression step in the inference pipeline with the matrix multiplication $E_{aligned} = H_K \times Q$.

**Validation Gate 3:** The system must successfully pass a latent thought from Qwen2.5-7B to LLaMA3.1-8B, resulting in correct downstream task execution without destroying the latent geometry.

---

## Phase 4: Continuous Dynamical Systems (Scaling the Horizon)
**Objective:** Replace the discrete autoregressive loop with a continuous ODE solver to shatter the 80-step latent degradation plateau.

**Action Items:**
1. **ODE Integration:** Import `torchdiffeq`. Wrap the transformer block of the reasoning agent in a differential equation class ($\frac{dh}{dt} = f_\phi(h(t), t)$).
2. **Continuous Generation:** Swap the discrete `for` loop generation with `odeint`, using a stable solver like `rk4` to integrate the thought trajectory over a continuous time horizon.
3. **State-Space Contingency (Optional):** If KV cache memory scales quadratically and causes VRAM exhaustion, replace the standard attention layers in the working memory transfer with a Mamba-style recurrent SSM module.

**Validation Gate 4:** The agent must successfully execute a simulated reasoning trajectory equivalent to 500+ steps without experiencing catastrophic representation drift or crashing VRAM.