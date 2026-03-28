# Proust Attention Machine -- Improvement Roadmap

Detailed reference for future improvements. Each item includes what it teaches, why it matters, and how to approach it. Ordered by recommended priority.

---

## Current State (March 2026)

- **Model**: 2 layers, 2 heads, d_model=128, ~420k params
- **Training**: 50 epochs, stride=128, val_loss=1.1739 (best at epoch 20)
- **Corpus**: 7.15M chars, 94-char vocab, character-level tokenization
- **Deployment**: HF model repo + Gradio Space

---

## Tier 1: Complete Phase 3 (Low Effort, High Portfolio Value)

### 1. Attention Heatmaps Per Head

**What it teaches**: Interpretability -- what each attention head actually learns.

**Approach**:
- Modify `model_torch.py` to optionally return attention weights from each head
- Feed a sample sentence, extract the `(heads, seq, seq)` attention matrix
- Plot with seaborn/matplotlib: x-axis = keys (characters attended to), y-axis = queries (characters doing the attending)
- Compare head 0 vs head 1 -- they likely specialize differently

**Expected findings**:
- One head may learn local patterns (bigrams, word boundaries)
- Another may learn longer-range structure (punctuation matching, clause boundaries)
- Proust's long sentences should show interesting long-distance attention patterns

**Key concept**: Attention weights sum to 1.0 across each row (softmax). High values mean "this position is important for predicting the next character at my position."

**Files to modify**: `src/model_torch.py` (return weights), new `viz/attention_heatmap.py`

### 2. Embedding Space Visualization (t-SNE/UMAP)

**What it teaches**: Representation learning -- how the model organizes characters internally.

**Approach**:
- Extract the learned embedding matrix from the model: `model.token_embedding.weight` shape `(94, 128)`
- Apply t-SNE or UMAP to reduce from 128 dimensions to 2
- Plot each character as a labeled point
- Color by category: vowels, consonants, accented chars, punctuation, digits

**Expected findings**:
- Vowels cluster together (a, e, i, o, u)
- Accented characters near their base form (a near a, e near e)
- Space character in a unique position (most frequent, unique role)
- Punctuation forms its own cluster

**Key concept**: The embedding layer transforms each of 94 discrete character IDs into a 128-dimensional vector. Characters that behave similarly in context end up with similar vectors -- the model discovered these relationships from raw text alone, without being told what vowels or consonants are.

**Dependencies**: `scikit-learn` for t-SNE, or `umap-learn` for UMAP

### 3. NumPy vs PyTorch Verification

**What it teaches**: Numerical debugging, floating-point precision, ML testing.

**Approach**:
- Create identical weight matrices for both NumPy and PyTorch models
- Feed the same input, compare outputs with `np.allclose(np_out, pt_out, atol=1e-5)`
- Test each component in isolation: embedding, attention, feedforward, full block
- Document any discrepancies (dropout, different layer norm implementations)

**Key concept**: IEEE 754 floating-point arithmetic is not associative -- `(a + b) + c != a + (b + c)` in general. Different operation orderings in NumPy vs PyTorch can cause tiny differences (~1e-6) that accumulate through layers. Understanding this is critical for debugging ML code.

**Files**: New notebook `notebooks/04_verification.ipynb`

---

## Tier 2: Model Improvements (Medium Effort, Deep Learning)

### 4. BPE Tokenization (Character -> Subword)

**What it teaches**: The single most impactful NLP concept. Used by GPT, BERT, LLaMA, and every production language model.

**Why it matters**: Character-level tokenization forces the model to rediscover that letters form words. BPE gives it a head start by learning common subword units like `["tiempo", " de", "ción"]`.

**Impact on this project**:
- Context window of 256 tokens covers ~1000 characters instead of 256
- Each token carries more semantic meaning -> more coherent generation
- Same 420k params produce dramatically better output

**Approach**:
1. Train a BPE tokenizer on the Proust corpus using `sentencepiece`:
   ```python
   import sentencepiece as spm
   spm.SentencePieceTrainer.train(input='data/processed/proust_corpus.txt',
                                   model_prefix='proust_bpe', vocab_size=1000)
   ```
2. Replace `CharTokenizer` with the BPE tokenizer in `dataset.py`
3. Adjust `vocab_size` in model config (94 -> 1000)
4. Retrain and compare val_loss and generation quality

**Key concepts**:
- **Merge operations**: BPE starts with characters and iteratively merges the most frequent pair. "t"+"i" -> "ti", "ti"+"em" -> "tiem", "tiem"+"po" -> "tiempo"
- **Vocabulary size tradeoff**: Too small (100) = still character-like. Too large (50,000) = rare tokens with poor embeddings. 1000-4000 is the sweet spot for a 7M char corpus
- **Tokenizer/model coupling**: Once you train a BPE model, the tokenizer is fixed -- you can't change the vocabulary without retraining

**Portfolio value**: Very high. Understanding tokenization deeply separates "toy project" from "production understanding."

### 5. Scaling Experiments (Depth vs Width)

**What it teaches**: Scaling laws, compute/quality tradeoffs, empirical ML.

**Experiments to run**:

| Config | Layers | Heads | d_model | d_ff | Params | Purpose |
|--------|--------|-------|---------|------|--------|---------|
| Baseline | 2 | 2 | 128 | 512 | ~420k | Current model |
| Deeper | 4 | 2 | 128 | 512 | ~800k | Does depth help? |
| Wider | 2 | 4 | 256 | 1024 | ~1.2M | Does width help? |
| Both | 4 | 4 | 256 | 1024 | ~2.5M | Full scale-up |

**Approach**:
- Train each variant for 50 epochs with same hyperparameters
- Plot val_loss curves overlaid
- Compare generation quality at same temperature
- Measure training time per epoch

**Key concepts**:
- **Depth vs width**: Deeper models learn more abstract features; wider models learn richer representations at each layer. The optimal ratio depends on data size.
- **Diminishing returns**: Doubling parameters doesn't halve the loss. The relationship is roughly logarithmic (Chinchilla scaling laws).
- **Compute-optimal training**: Given a fixed compute budget, there's an optimal model size. Too large = underfitting (not enough epochs). Too small = capacity bottleneck.

**All variants trainable on Colab free tier T4** with stride=128.

### 6. Dropout and Regularization Tuning

**What it teaches**: Generalization, overfitting diagnostics, hyperparameter tuning.

**Evidence of overfitting**: Best val_loss at epoch 20 out of 49. After epoch 20, the model started memorizing training data instead of learning generalizable patterns.

**Experiments**:
- Dropout: 0.1 (current) vs 0.2 vs 0.3
- Weight decay: 0.01 (current) vs 0.05 vs 0.1
- Plot train loss vs val loss for each setting
- Find the setting where the gap is smallest

**Key concept**: The train/val gap tells you everything. If train_loss << val_loss, the model is memorizing. If train_loss ≈ val_loss, the model is generalizing well. Dropout and weight decay are the two main knobs to control this.

---

## Tier 3: Advanced Concepts (Higher Effort, Research-Adjacent)

### 7. KV-Cache for Fast Inference

**What it teaches**: How production LLMs achieve fast token generation. Critical for any LLM engineering role.

**The problem**: Current `generate()` recomputes attention over the entire context for every new character. For a 500-character generation, position 499 recalculates attention for positions 0-498 even though those haven't changed.

**The solution**: Cache the Key and Value matrices from previous positions. When generating token N, only compute Q for position N, but reuse K and V from positions 0 to N-1.

**Speedup**: For generating L tokens with context C, goes from O(L * C^2) to O(L * C). For 500 chars with context 256: roughly 256x fewer attention computations.

**Approach**:
1. Add `past_kv` parameter to each attention layer
2. During generation, accumulate K,V tensors across steps
3. Only compute the new token's Q at each step
4. Return updated cache for next step

**Key concept**: This is the mechanism behind the "prefill" and "decode" phases in production LLM serving. Understanding it deeply prepares you for systems like vLLM, TensorRT-LLM, and any LLM deployment work.

### 8. Rotary Positional Embeddings (RoPE)

**What it teaches**: Modern position encoding. Used by LLaMA, Mistral, Qwen, and most open-source LLMs since 2023.

**Why replace sinusoidal**: Sinusoidal encoding adds absolute position information. RoPE encodes relative position directly in the attention computation -- the dot product between Q and K naturally decays with distance, which better matches how language works (nearby words are usually more relevant).

**The math**:
- Rotate Q and K vectors by an angle proportional to their position
- The dot product Q_i . K_j depends on (i - j), not on absolute positions
- Uses complex number multiplication: `(a + bi) * (cos(theta) + i*sin(theta))`

**Approach**:
1. Implement `apply_rotary_emb(x, freqs)` function
2. Replace sinusoidal positional encoding with RoPE in attention
3. Compare generation quality

**Key concept**: Position information is injected multiplicatively (rotation) instead of additively (sinusoidal). This makes the model naturally distance-aware without explicit position tokens.

### 9. Flash Attention (Conceptual Understanding)

**What it teaches**: GPU memory hierarchy, memory-bound vs compute-bound operations.

**Not a coding task** -- this is about reading and understanding the paper. Standard attention materializes the full (seq, seq) attention matrix in GPU HBM (High Bandwidth Memory). Flash Attention tiles the computation to keep intermediate results in SRAM (fast cache), avoiding the expensive HBM reads/writes.

**What to study**:
- The Flash Attention paper (Dao et al., 2022): https://arxiv.org/abs/2205.14135
- Focus on Figure 1 (memory hierarchy) and Section 3 (tiling algorithm)
- Understand why attention is memory-bound, not compute-bound
- Why this enables longer context windows without quadratic memory growth

**Portfolio value**: Being able to explain Flash Attention in an interview shows systems-level understanding that pure ML people often lack.

---

## Recommended Progression

```
Priority 1 (now):     Attention heatmaps + Embedding viz
                      -> Completes Phase 3, makes the demo visually compelling

Priority 2 (next):    NumPy/PyTorch verification
                      -> Quick win, proves implementation rigor

Priority 3 (core):    BPE tokenization
                      -> Biggest learning jump, transforms model quality

Priority 4 (depth):   KV-cache + Scaling experiments
                      -> Production knowledge + empirical ML skills

Priority 5 (polish):  RoPE + Flash Attention study
                      -> Modern architecture understanding
```

---

## Resources for Each Topic

| Topic | Primary Resource |
|-------|-----------------|
| Attention heatmaps | BertViz library docs, Illustrated Transformer (jalammar) |
| t-SNE/UMAP | Distill.pub "How to Use t-SNE Effectively" |
| BPE tokenization | Hugging Face tokenizers course, Karpathy minbpe repo |
| Scaling laws | Chinchilla paper (Hoffmann et al., 2022) |
| KV-cache | Karpathy nanoGPT, "LLM Inference" blog posts |
| RoPE | Su et al. 2021 paper, Eleuther AI blog post |
| Flash Attention | Dao et al. 2022, Aleksa Gordic YouTube walkthrough |
