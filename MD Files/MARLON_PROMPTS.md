Task 1: Multi-Layer Manifold Alignment


Updated Prompt 1.1: Multi-Layer Procrustes Logic
  > "Update src/utils/alignment.py to support aligning multiple layers simultaneously.
  > 1. Modify compute_orthogonal_mapping to accept a list of tensors (hidden states from multiple layers) instead of a single tensor.
  > 2. Implement 'Concatenated Procrustes': Concatenate the hidden states along the feature dimension before computing the SVD.
  > 3. Ensure the function remains compatible with the existing single-layer calls by checking if the input is a list or a single tensor.
  > Completed: Manifold Alignment math in the utility layer.
  > Codebase Impact: Centralizes the manifold mapping logic in the correct utility module.
  > Verification: Run a test script importing from src.utils.alignment to verify SVD rank increases with multi-layer inputs."



  Prompt 1.2: Multi-Layer Handoff Execution
  > "Update run_hybrid_pipeline in latent_pipeline.py to use multi-layer alignment.
  > 1. Modify the anchor-point generation to extract hidden states from a set of 'Reasoning Layers' (e.g., the middle 25% of the model).
  > 2. Update the apply_orthogonal_mapping call to use the new multi-layer mapping Q.
  > 3. Constraint: Keep the function signature of run_hybrid_pipeline the same; use internal variables for the layer indices.
  > Completed: Multi-layer pipeline integration.
  > Codebase Impact: Dramatically reduces 'Representational Drift' because the Actor now receives a signal aligned with its internal reasoning path,
  not just its output.
  > Verification: Run the 'Entropy' test; verify that Agent B's first 5 tokens are more deterministic/consistent than the zero-shot baseline."

  ---

  Task 2: Adaptive Neural ODE Dynamics (The "Thought Depth")


Updated Prompt 2.1: Input Complexity Estimator (with Hydra)
  > "Add a 'Complexity Scorer' to the reasoning pipeline.
  > 1. In latent_pipeline.py, implement estimate_problem_complexity(prompt).
  > 2. Add a new configuration section to configs/main.yaml under dynamics: called complexity_thresholds.
  > 3. Use these config values to return a scaling factor between 0.5 and 2.0.
  > Completed: Config-driven reasoning intensity controller.
  > Codebase Impact: Allows researchers to tune the 'thought-depth' sensitivity without touching the code.
  > Verification: Run python latent_pipeline.py dynamics.complexity_thresholds.math=1.5 and verify the scaling factor changes."



  Prompt 2.2: Dynamic ODE Integration
  > "Modify the Neural ODE loop in latent_pipeline.py to use the complexity score.
  > 1. Scale the LATENT_STEPS or the time_space end-point by the complexity score.
  > 2. If complexity is high, increase the integration steps; if low, decrease them to save latency.
  > 3. Update TransformerBlockDynamics to accept a complexity_factor if needed.
  > Completed: Adaptive Reasoning Depth.
  > Codebase Impact: This is a major research contribution (Dynamic Latent Reasoning). It optimizes the 'Bandwidth vs. Accuracy' tradeoff in real-time.
  > Verification: Measure the continuous_integration_time for a simple prompt vs. a complex one. The complex prompt should take longer and have more
  'steps' in the trajectory."

  ---

  Task 3: Compression Scaling Laws (The "Shannon Limit")


  Prompt 3.1: Latent Step Sweep Script
  > "Create a standalone script sweep_compression.py to identify the 'Optimal Compression Ratio.'
  > 1. Loop through LATENT_STEPS values of [1, 2, 4, 8, 16, 32, 64].
  > 2. For each value, run the evaluate_hybrid_mas.py logic on 10 samples of GSM8K.
  > 3. Record the accuracy and total latency for each step count.
  > Completed: Scaling Law Data Collector.
  > Codebase Impact: Provides the data for the 'S-Curve' graph in your paper, proving the efficiency of your protocol.
  > Verification: Ensure the script generates a scaling_results.csv with clear performance drop-offs at very low LATENT_STEPS (e.g., 1 or 2)."

  ---


  Phase 2 "Final Goal" (The Paper Contribution):
  By the end of these prompts, you will have implemented "Adaptive Manifold Alignment." This isn't just a MAS; it's a system that:
   1. Aligns the "Reasoning Manifolds" of two different models (Task 1).
   2. Adjusts its "Thinking Depth" based on problem difficulty (Task 2).
   3. Quantifies the Efficiency of latent vs. text communication (Task 3).


  Hand the Noah prompts to your partner. You now have the keys to Phase 2. Prompt your model for Task 1.1 to begin.
  ---


  Task 4: Uncertainty-Aware Handoff (The "Confidence" Filter)


 Updated Prompt 4.1: Entropy-Weighted Loss (New Location)
  > "Implement the 'Uncertainty-Weighted Agreement' ($\mathcal{L}_{pref}$) in `train_compressor.py`.
  > 1. Locate the LatentCompressorLoss class (move it to src/models/losses.py first if not already done).
  > 2. Calculate the Entropy of the Actor's output distribution.
  > 3. Implement the weighting $w_t = 1 / (\text{entropy} + \epsilon)$ and apply it to the KL-Divergence loss.
  > Completed: Intelligence-Weighted Loss in the core models package.
  > Codebase Impact: Professionalizes the training objective to focus on 'high-information' tokens.
  > Verification: Monitor WandB; ensure l_pref correlates with tokens where the model is highly confident (e.g., the first token of a formula)."



  Prompt 4.2: Dynamic Handoff Gating
  > "Implement a 'Confidence Gate' in run_hybrid_pipeline.
  > 1. After the ODE reasoning but before the handoff, calculate the 'Uncertainty' of the final latent state.
  > 2. If the uncertainty exceeds a threshold (meaning the Reasoner is 'confused'), fallback to a few extra discrete reasoning steps before the
  handoff.
  > Completed: Stability Gating.
  > Codebase Impact: Prevents 'Garbage-In, Garbage-Out' by ensuring the Actor only receives latent signals that meet a minimum quality standard.
  > Verification: Trigger a fallback by providing a nonsense prompt (e.g., 'asdfghjkl') and verify the system executes more steps than a standard
  prompt."

  ---

  Task 5: Latent Trajectory Visualization (The "Paper Artifacts")


  Prompt 5.1: PCA Trajectory Plotting
  > "Create a visualization utility visualize_thoughts.py.
  > 1. Use PCA (Principal Component Analysis) or t-SNE to project the high-dimensional latent trajectory (the output of the ODE) into 2D space.
  > 2. Plot the trajectory of a 'Correct' math solution vs. an 'Incorrect' one.
  > 3. Highlight the 'Handoff Point' where the Reasoner passes the signal to the Actor.
  > Completed: The 'Visual Evidence' generator.
  > Codebase Impact: Provides the primary qualitative figure for the research paper, showing how the model's 'train of thought' physically moves
  through latent space.
  > Verification: Run the script and ensure a .png plot is generated showing two distinct paths for two different prompts."

  ---

  Task 6: Tokenizer Mismatch Bridge (The "Semantic Translator")


  Prompt 6.1: Cross-Model Vocabulary Anchor Mapping
  > "Improve the Procrustes alignment by using 'Semantic Anchors.'
  > 1. Instead of aligning on a single prompt, create a list of 100 'Fundamental Concepts' (e.g., 'number', 'add', 'logic', 'if', 'then').
  > 2. Generate the hidden states for these 100 anchors in both Qwen and EXAONE.
  > 3. Compute a global 'Universal Alignment Matrix' $Q_{global}$ using these anchors.
  > Completed: Semantic Manifold Alignment.
  > Codebase Impact: Significantly reduces the 'gibberish' output seen in zero-shot tests because the alignment is now anchored to shared human
  concepts, not just a single prompt.
  > Verification: Run latent_pipeline.py using $Q_{global}$ and compare the decoded text coherence to the single-prompt $Q$ used in the smoke test."
