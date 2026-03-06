# PROMPTS_AND_EVALUATION.md: Agent Orchestration and Benchmarks

## 1. Multi-Agent System Prompts
Because the agents communicate primarily through continuous latent vectors and KV cache transfers, their natural language system prompts must be strictly scoped to their specific roles. They should not be prompted to explain their reasoning to *each other* in text, but rather to process the incoming latent state and output the final required format.

**System Prompt (Global Context for All Agents):**
"You are a specialized node in a continuous latent multi-agent system. You will receive working memory and reasoning context directly via your attention cache. Do not output conversational filler."

**Agent 1 (The Continuous Reasoner / Planner):**
"You are the root reasoning module. Given the following complex problem, project the necessary step-by-step logic into your continuous latent space. Do not output text. Generate the latent trajectory required to solve the problem. 
Target Problem: {input_problem}"

**Agent 2 (The Latent Critic - Optional for 3-node setups):**
"You are the latent evaluation node. You have received the reasoning trajectory of the root module via your KV cache. Apply a non-linear transformation to identify logical flaws, mathematical errors, or inefficient execution paths. Output the corrected latent vector."

**Agent 3 (The Decoder / Solver):**
"You are the execution node. You have received the fully optimized latent working memory from the preceding agents. Decode this continuous thought vector into the final, exact text output or action required to solve the problem. Place your final answer inside \boxed{YOUR_FINAL_ANSWER}."

## 2. Evaluation Metrics & Benchmarks
To prove that this architecture is a fundamental leap over text-based auto-regressive systems, it must be evaluated against aggressive thresholds. These metrics are designed to demonstrate venture-scale potential, ensuring the empirical results meet the rigorous standards expected by top-tier technical accelerators like Y Combinator.

### A. Mathematical & Logical Reasoning
* **Benchmark:** MATH (Level 5) and AIME 2024.
* **Metric:** Exact match accuracy. 
* **Target:** The hybrid latent system must achieve a $\ge 15\%$ absolute accuracy improvement over a single-agent Chain-of-Thought baseline, proving that the continuous vector space prevents the premature reasoning collapse seen in text decoding.

### B. Autonomous Code Generation
* **Benchmark:** HumanEval-Plus and MBPP-Plus.
* **Context:** Given the industry shift towards fully automated programming architectures—aligning with Jensen Huang's argument that people should not learn to code—the system's code generation capabilities must be evaluated for complete autonomy without human-in-the-loop debugging.
* **Target:** Pass@1 rates must be evaluated, specifically testing if the latent KV cache transfer allows the Critic agent to catch edge-case compilation errors that the Planner agent missed in its initial latent projection.

### C. System Efficiency Metrics
* **Token Reduction:** Measure the total number of text tokens generated per task. 
    * *Target:* $> 80\%$ reduction in token generation compared to standard conversational multi-agent frameworks (e.g., AutoGen, ChatDev).
* **End-to-End Latency:** Measure the wall-clock time from the initial prompt to the final decoded answer.
    * *Target:* A 4x to 5x speedup, demonstrating the latency advantages of the "Zip-Cache" protocol and continuous ODE solvers over serial text decoding.