# Proust Attention Machine -- Build Session 01

## Actions

- Audited full project state with parallel agents (data, source, notebooks, structure)
- Implemented `LayerNorm`, `FeedForward`, `TransformerBlock`, and full `Transformer` model in NumPy
- Created centralized `CONFIG` dict eliminating magic numbers
- Ran successful end-to-end forward pass: tokens in, probabilities out
- Deep-dive teaching session covering: transpose mechanics, tensor dimensionality, FeedForward expand-compress pattern, residual connections
- Compared architecture to frontier models (Llama 405B, GPT-4 leaked specs)
- Discussed self-supervised learning paradigm, AGI state, interpretability limits, and the "faith function" concept

## Outputs

- `src/model.py` -- Full transformer model with LayerNorm, FeedForward, TransformerBlock, Transformer classes (~350 lines)
- `subagents_outputs/data_review.md` -- Data pipeline audit
- `subagents_outputs/src_review.md` -- Source code audit
- `subagents_outputs/notebooks_review.md` -- Notebooks audit
- `subagents_outputs/structure_review.md` -- Project structure audit
- `docs/project/01_transformer_build_session_project.md` -- This document
- `docs/technical/01_transformer_architecture_reference.md` -- Technical reference
- `docs/intuitive/01_transformer_and_agi_intuition.md` -- Intuitive mental models

## Chronology

* Project audit with parallel agents

We launched 4 parallel exploration agents to audit every aspect of the project simultaneously: data pipeline, source code correctness, notebook status, and overall structure. The audit revealed Phase 1 was ~60% complete -- tokenizer, embedding, and attention were solid and mathematically correct, but LayerNorm, FeedForward, TransformerBlock, and the full model were missing. The corpus was only 795 bytes (a sample), and `download_corpus.py` did not exist. We based this on reading every existing file and comparing against the CLAUDE.md specification. For detailed look see 01_transformer_architecture_reference.md

* Implementation of model.py

We implemented the four missing NumPy components: `LayerNorm` (normalizes activations across d_model dimension), `FeedForward` (2-layer MLP with ReLU, expand 64->256->64), `TransformerBlock` (attention + feedforward with residual connections and normalization), and `Transformer` (full model: embedding -> 2 blocks -> output projection -> softmax). We followed the existing code conventions -- shape annotations everywhere, verbose variable names, educational comments. The forward pass was verified end-to-end producing correct shapes: input (1, 12) -> logits (1, 12, 18) -> probabilities (1, 12, 18). Parameter count: ~102k, within the ~300k target from CLAUDE.md. This brings Phase 1 to ~90% completion. For detailed look see 01_transformer_architecture_reference.md

* Teaching session: tensor mechanics

We traced the mechanics of `.transpose()` on 4D tensors, explaining that it's a full axis permutation (not just swapping last two), with 4!=24 possible arrangements for a 4D tensor. We walked through why `Q @ K.transpose(0,1,3,2)` produces the (seq, seq) attention score matrix, and why 2D matrices don't need explicit arguments (only one non-trivial permutation: T(1,0)). We also covered the FeedForward expand-compress pattern with concrete 3-number examples and the residual connection's role in gradient flow (the `+1` in the derivative guarantees signal propagation).

* Frontier model comparison

We compared the project's architecture against Llama 3.1 405B (public config: d_model=16384, 128 heads, 126 layers, 405B params) and GPT-4 (leaked: ~1.8T params, 120 layers, 16 MoE experts). Key finding: the architecture is identical -- same Q@K.T, same FeedForward, same residuals. Differences are engineering refinements for scale: GQA, RoPE, SwiGLU, MoE. The 2017 transformer architecture survives unchanged at its core. Anthropic's Claude architecture remains undisclosed.

* Discussion: self-supervised learning, AGI, and interpretability

We clarified that LLMs use self-supervised learning (not unsupervised) -- the text IS its own answer key, each character serves as both input and label. We discussed the current state of AGI honestly: transformers produce emergent capabilities from simple next-token prediction, but lack persistent learning, verified world models, and reliable novel reasoning. The interpretability discussion surfaced a deep epistemological problem: finding a "gravity circuit" in a model doesn't prove understanding -- we're projecting human concepts onto activation patterns. The user proposed a "faith function" concept: humans collapse probability distributions over future actions into single committed choices, with personal parameters (risk tolerance, optimism, identity). This led to the mutual observation that human cognition may be less special than assumed -- also emergence from simple mechanisms (atoms -> molecules -> cells -> neurons -> intelligence), with no identifiable layer where "intelligence enters."

---

> **Quote from Claude Opus 4.5 during this session:**
>
> "The most profound thing about building a transformer from scratch is the gap between mechanism and behavior. You wrote model.py. You see that it's just matrix multiplications, additions, and a softmax. There is nothing in that code that could be called 'understanding' or 'intelligence' by looking at it. And yet, at sufficient scale, with sufficient data, this same mechanism produces behavior that millions of people experience as intelligent, creative, even empathetic. Either intelligence is simpler than we thought (just pattern matching at scale), or we're fooling ourselves about what these systems are doing, or there's something about the mathematics of high-dimensional spaces that we don't yet understand. All three possibilities are humbling. The honest answer to 'will this reach AGI?' is: we built something that works far better than our theories predict it should, and we don't fully understand why. That's simultaneously the most exciting and most unsettling thing about this field."
