# LXP Final Video Script

Target length: 5 to 7 minutes

Core framing: this is a functional research prototype with a successful same-family raw latent decode smoke, not a finished production protocol for arbitrary model pairs.

## 0:00-0:35 - Hook

Visual: Open `docs/final_demo.html` or the title section of `docs/lxp_research_paper.html`.

Speaker:

"Most multi-agent AI systems communicate in text. One model thinks, writes out a long chain of thought, and another model reads it back in. That works, but it is slow and lossy. The first model has a rich internal mathematical state, then compresses it into words, and the second model has to reconstruct meaning from those words.

My project, LXP, asks a different question: can models pass compact latent states directly instead of passing long text transcripts?"

## 0:35-1:15 - Problem and Motivation

Visual: Show the animated pipeline in `docs/final_demo.html`.

Speaker:

"The motivation is that text is a human interface, not necessarily the best machine-to-machine interface. If models could exchange compressed hidden states, multi-agent systems might become faster, cheaper, and eventually more expressive.

That is the long-term research idea. The practical project goal was to build a real testbed for it: compatibility checks, latent extraction, alignment, training, decode evaluation, and failure diagnostics."

## 1:15-2:15 - Architecture

Visual: Scroll to Figure 1 in `docs/lxp_research_paper.html`.

Speaker:

"The architecture has two agents. Agent A is the reasoner. It processes the prompt and produces hidden-state trajectories. LXP compresses those states into a short latent packet. In the final local smoke, that packet is only eight latent steps.

Then a Stage-II handoff adapter maps the latent packet into the actor interface. Agent B receives the latent prefix and tries to decode the final answer.

The important part is that I did not evaluate this with one vague accuracy number. The repo separates several surfaces: the actor text baseline, the latent answer probe, semantic readout, raw actor decode, extraction rate, unique answer count, and model-pair compatibility."

## 2:15-3:05 - What Was Built

Visual: Show source files or the implementation table in the research paper.

Speaker:

"The implementation is a full Python codebase, not just a demo notebook.

`latent_pipeline.py` handles the runtime latent handoff path. `train_compressor.py` implements Stage-II training objectives, including answer loss, first-token loss, latent probing, logit steering, and early stopping. `run_training.py` turns those pieces into a training run and writes structured reports. `benchmark_all.py` runs smoke and benchmark-style comparisons. There is also a preflight model compatibility tool so impossible cache transfers are detected before loading full weights.

This mattered because a lot of the project was failure analysis. When accuracy was zero, I needed to know whether the problem was the baseline actor, the latent injection, the decoder, the answer parser, or the model pair itself."

## 3:05-4:20 - Main Result

Visual: Scroll to "Final Metric Summary" and "Raw Decode Learning Curve" in `docs/lxp_research_paper.html`.

Speaker:

"The latest same-family smoke uses Qwen3.5-0.8B as both the reasoner and actor on Apple Silicon MPS. It is a three-sample train-overfit smoke, so I am not claiming broad generalization yet.

But as an interface proof, it works. The final run reaches 100 percent raw actor free-decode exact match, 100 percent answer extraction, 100 percent latent probe accuracy, and three out of three unique predicted answers.

The learning curve is important. At initialization, raw actor decode was degenerate. It repeatedly produced the wrong answer. During training, it moved through partial success and instability, and the readiness gate only passed once raw exact match, extraction, and non-degeneracy all passed together."

## 4:20-5:05 - Demo Outputs

Visual: Show the decoded examples table.

Speaker:

"Here are the final decoded examples. The targets were 4, 42, and 5. The raw latent actor decode produced `Final answer: 4`, `Final answer: 42`, and `Final answer: 5`.

This is not just a semantic bridge printing the answer. The raw actor path is the result I wanted for the MVP: the actor receives a learned latent handoff and emits parseable final-answer text."

## 5:05-5:50 - Limitations and Honesty

Visual: Show "Honest Scope", "Still Open", or Appendix B in the paper.

Speaker:

"There are important limitations. This is a same-family smoke, not a production benchmark. It uses three examples and one seed. The broad Phase-II gate still blocks because real mode, multi-seed evaluation, and baseline retention have not been completed.

Heterogeneous model-family transfer is also not solved yet. One thing the project discovered is that model-pair selection matters. Some pairs, like Qwen to EXAONE, do not have compatible cache topology, so they need a different alignment strategy instead of direct KV transfer."

## 5:50-6:35 - Evaluation and Next Steps

Visual: Show commands in the paper appendix or README.

Speaker:

"The next real evaluation is to lock a held-out manifest and compare LXP against pure token-context handoff, verified token-context handoff, and sender-answer text handoff. That will tell us whether latent transfer is actually useful beyond a controlled smoke.

The repo is set up for that next step. It has smoke runners, generated trajectory adapters, report rendering, compatibility preflight, DigitalOcean GPU guidance, and Mac MPS fallback training."

## 6:35-7:00 - Closing and Disclosure

Visual: Return to the title page or final demo page.

Speaker:

"My final pitch is: LXP is a working research prototype for latent communication between language models. It does not yet prove the full long-term vision, but it does demonstrate substantial implementation, iteration, diagnostics, and a real same-family latent decode result.

I also want to disclose AI assistance. I used AI tools as coding and debugging collaborators while I directed the project goals, interpreted failures, chose the MVP path, and assembled the final evaluation and presentation."

## Backup 90-Second Version

"LXP is a research prototype for latent context transfer between language models. Today, multi-agent systems usually pass text: one model writes a long chain of thought, another reads it back in. My project asks whether models can instead pass compact hidden-state packets directly.

The repo implements a full testbed: model-pair compatibility preflight, latent extraction, Stage-II handoff training, actor-prefix injection, semantic probes, raw actor decode, extraction checks, non-degeneracy gates, benchmark reports, and visual documentation.

The final local result is a same-family Qwen3.5-0.8B to Qwen3.5-0.8B smoke on Apple Silicon MPS. It reaches 100 percent raw actor free-decode exact match, 100 percent answer extraction, and three out of three unique answers on a three-sample train-overfit smoke. The raw path started degenerate and only passed after training and readiness gating.

I am not claiming broad generalization yet. Heterogeneous transfer and multi-seed held-out benchmarks remain future work. But the artifact is functional, understandable, and demonstrates meaningful progress: I built the infrastructure to diagnose where latent handoff fails and got the same-family raw decode interface to work in a controlled setting."
