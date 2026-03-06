# PROJECT_OVERVIEW.md: Hybrid Continuous Latent Multi-Agent System (HL-MAS)

## 1. Executive Summary
The Hybrid Continuous Latent Multi-Agent System (HL-MAS) is an advanced AI framework that transitions multi-agent collaboration from discrete natural language tokens to continuous, high-dimensional vector spaces. By synthesizing the aggressive latent reasoning compression of the *Interlat* architecture with the lossless, training-free Key-Value (KV) cache transfer mechanism of *LatentMAS*, this system fundamentally increases inter-agent communication bandwidth. The ultimate objective is to enable near-infinite step continuous reasoning and zero-shot geometric alignment between heterogeneous large language models without relying on explicit text decoding.

## 2. Core Technology Stack
* **Deep Learning Framework:** PyTorch (Core autograd, tensor operations, and custom SVD solvers for Orthogonal Procrustes alignment).
* **Continuous Dynamics:** `torchdiffeq` (For Neural ODE continuous solver integration to bypass discrete step limits).
* **Model Orchestration:** HuggingFace `transformers` (For loading base models, intercepting `lm_head` outputs, and manipulating `past_key_values`).
* **Training Backend:** DeepSpeed ZeRO-2 (For memory-efficient mixed-precision bfloat16 training of the reasoning module).
* **Inference Engine:** vLLM (For optimized KV prefix caching and high-throughput evaluation).
* **Target Base Models:** Qwen2.5-7B-Base (Primary Sender/Reasoner) and LLaMA3.1-8B-Base (Heterogeneous Receiver/Actor).

## 3. Target Benchmarks
The system will be evaluated on high-complexity, multi-step tasks where traditional Chain-of-Thought (CoT) text generation struggles due to sequence length and early token collapse:
* **Interactive & Embodied Planning:** ALFWorld (evaluated on success rates within a 20-step environment budget).
* **Complex Mathematical Reasoning:** MATH (specifically Level-5 difficulty) and GSM8K.

## 4. Success Criteria & Definition of Done
The build is considered successful when the following empirical metrics are achieved:
1. **Accuracy Gain:** The hybrid system must outperform both single-model CoT and standard text-based MAS baselines on the MATH Level-5 benchmark.
2. **Latent Compression:** The trained reasoning module must successfully compress a 128-step latent trajectory into $\le 16$ continuous steps while retaining at least 95% of the uncompressed baseline accuracy.
3. **Latency Reduction:** End-to-end communication and execution latency must be reduced by at least 4x compared to traditional auto-regressive text-based MAS.
4. **Heterogeneous Alignment:** The system must successfully execute a cross-family latent transfer (e.g., Qwen-7B $\rightarrow$ LLaMA3-8B) using the Orthogonal Procrustes geometric alignment without crashing or experiencing catastrophic representation drift.