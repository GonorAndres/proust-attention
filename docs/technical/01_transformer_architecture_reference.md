# Transformer Architecture Reference

## Model Configuration

| Parameter | Symbol | Value | Notes |
|:----------|:-------|:------|:------|
| Embedding dimension | d_model | 64 | All layers maintain this width |
| Attention heads | n_heads | 2 | Parallel attention patterns |
| Dimension per head | d_k | 32 | d_model / n_heads |
| Transformer blocks | n_layers | 2 | Stacked sequentially |
| FeedForward hidden | d_ff | 256 | 4 * d_model (convention) |
| Context window | max_seq_len | 256 | Characters, not words |
| Vocab size | vocab_size | ~50 | Spanish character-level |
| Init scale | init_scale | 0.02 | Weight initialization std |
| Mask value | mask_value | -1e9 | Causal mask fill |

## Parameter Count

| Component | Formula | Count |
|:----------|:--------|------:|
| Token embedding | vocab_size * d_model | ~3,200 |
| W_Q per block | d_model * d_model | 4,096 |
| W_K per block | d_model * d_model | 4,096 |
| W_V per block | d_model * d_model | 4,096 |
| W_O per block | d_model * d_model | 4,096 |
| FF W_1 per block | d_model * d_ff | 16,384 |
| FF b_1 per block | d_ff | 256 |
| FF W_2 per block | d_ff * d_model | 16,384 |
| FF b_2 per block | d_model | 64 |
| LayerNorm (gamma+beta) per block | 2 * 2 * d_model | 256 |
| Final norm | 2 * d_model | 128 |
| Output projection | d_model * vocab_size | ~3,200 |
| **Total (2 blocks, vocab=50)** | | **~105,000** |

## Shape Flow

```
Input:                (batch, seq)
Embedding lookup:     (batch, seq, 64)
+ Positional enc:     (batch, seq, 64)

--- TransformerBlock x2 ---
  Attention:
    Q,K,V project:    (batch, seq, 64) @ (64, 64) = (batch, seq, 64)
    Split heads:       (batch, seq, 64) -> (batch, 2, seq, 32)
    Scores Q@K.T:      (batch, 2, seq, 32) @ (batch, 2, 32, seq) = (batch, 2, seq, seq)
    Scale:             / sqrt(32)
    Mask:              + triu(-1e9)
    Softmax:           (batch, 2, seq, seq)
    Weighted V:        (batch, 2, seq, seq) @ (batch, 2, seq, 32) = (batch, 2, seq, 32)
    Merge heads:       (batch, 2, seq, 32) -> (batch, seq, 64)
    Output proj:       (batch, seq, 64) @ (64, 64) = (batch, seq, 64)
  Residual + Norm:     x = LayerNorm(x + attention_out)

  FeedForward:
    Expand:            (batch, seq, 64) @ (64, 256) + bias = (batch, seq, 256)
    ReLU:              max(0, x)
    Compress:          (batch, seq, 256) @ (256, 64) + bias = (batch, seq, 64)
  Residual + Norm:     x = LayerNorm(x + ff_out)
--- End Block ---

Final norm:           (batch, seq, 64)
Output projection:    (batch, seq, 64) @ (64, vocab) = (batch, seq, vocab)
Softmax:              (batch, seq, vocab)
```

## Core Formulas

```
ATTENTION:
  Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k) + mask) @ V

SOFTMAX (numerically stable):
  softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

CAUSAL MASK:
  mask = triu(ones(seq, seq), k=1) * (-1e9)

LAYER NORM:
  LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
  mean, var computed across d_model axis (last axis)

FEEDFORWARD:
  FF(x) = ReLU(x @ W_1 + b_1) @ W_2 + b_2

RELU:
  ReLU(x) = max(0, x)

RESIDUAL:
  output = LayerNorm(x + sublayer(x))
  gradient: d(output)/d(x) = 1 + d(sublayer)/d(x)    (the +1 prevents vanishing)

TRANSFORMER BLOCK:
  x = LayerNorm(x + MultiHeadAttention(x))
  x = LayerNorm(x + FeedForward(x))
```

## Transpose Reference

```
2D: A.T = A.transpose(1, 0)          only option, no args needed
3D: x.transpose(a, b, c)             3! = 6 permutations, 5 non-trivial
4D: x.transpose(a, b, c, d)          4! = 24 permutations, 23 non-trivial

Key transposes in this model:
  split_heads:    (batch, seq, heads, d_k) -> .transpose(0, 2, 1, 3) -> (batch, heads, seq, d_k)
  merge_heads:    (batch, heads, seq, d_k) -> .transpose(0, 2, 1, 3) -> (batch, seq, heads, d_k)
  attention K.T:  (batch, heads, seq, d_k) -> .transpose(0, 1, 3, 2) -> (batch, heads, d_k, seq)
```

## Frontier Model Comparison

| Parameter | This model | Llama 3.1 405B | GPT-4 (leaked) |
|:----------|:-----------|:---------------|:---------------|
| d_model | 64 | 16,384 | undisclosed |
| n_heads | 2 | 128 (8 KV) | MQA |
| n_layers | 2 | 126 | 120 |
| d_ff | 256 | 53,248 | undisclosed |
| context | 256 | 131,072 | 128,000 |
| vocab_size | ~50 | 128,256 | ~100,000 |
| params | ~105K | 405B | ~1.8T |
| Architecture | Dense transformer | Dense + GQA + RoPE + SwiGLU | MoE (16 experts, 2 active) |

## Key Identities

```
d_k = d_model / n_heads
d_ff = 4 * d_model                              (convention from Vaswani et al.)
attention cost = O(seq^2 * d_model)              (quadratic in sequence length)
feedforward cost = O(seq * d_model * d_ff)       (linear in sequence length)
feedforward params = ~2/3 of total params        (the bulk of the model)
```

## File Map

| File | Classes / Functions | Status |
|:-----|:-------------------|:-------|
| `src/tokenizer.py` | `CharTokenizer` | Complete |
| `src/embedding.py` | `Embedding` | Complete |
| `src/attention.py` | `softmax`, `create_causal_mask`, `SingleHeadAttention`, `MultiHeadAttention` | Complete |
| `src/model.py` | `LayerNorm`, `relu`, `FeedForward`, `TransformerBlock`, `Transformer`, `CONFIG` | Complete |
| `src/train.py` | Training loop | Not started |
| `src/generate.py` | Text generation + sampling | Not started |
