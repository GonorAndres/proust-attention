# The sqrt(d) Scaling Factor: From Variance to Transformers

**A Bridge Document for the Proust Attention Machine**

---

## Abstract

This document answers one question: *why does sqrt(d) appear everywhere in transformers?*
We build the answer from first principles -- starting with the variance of a single random variable,
then sums, then dot products, and finally connecting to both the `1/sqrt(d_k)` scaling in attention
and the `sqrt(d_model)` scaling in embeddings. Every step uses concrete numerical examples
from our 64-dimensional Proust model.

---

## Notation and Conventions

| Symbol | Meaning | Our value |
|--------|---------|-----------|
| `d` | Generic dimension (number of elements in a vector) | varies |
| `d_model` | Embedding dimension | 64 |
| `d_k` | Per-head dimension (`d_model / n_heads`) | 32 |
| `n_heads` | Number of attention heads | 2 |
| `sigma` | Standard deviation of a random variable | varies |
| `sigma^2` | Variance of a random variable | varies |
| `E[X]` | Expected value (average) of random variable X | -- |
| `Var(X)` | Variance of random variable X | -- |
| `q, k` | Query and key vectors in attention | `(d_k,)` |

Vectors are written in bold (**v**), scalars in italics. Shape annotations appear in comments: `(batch, seq, d_model)`.

---

## 1. Prerequisite: What Is Variance?

Before we can understand sqrt(d), we need to be precise about variance.

> **[FORMAL]** Definition: Variance
>
> The **variance** of a random variable X measures how spread out its values are from the mean:
>
> ```
> Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
> ```
>
> The **standard deviation** is `sigma = sqrt(Var(X))`.
>
> A random variable `X ~ N(0, 1)` (standard normal) has:
> ```
> E[X] = 0,    Var(X) = 1,    sigma = 1
> ```
> This means values are typically between -2 and +2 (within 2 standard deviations).

> **[INTUITION]** What variance feels like
>
> Variance tells you **how big the numbers tend to be** (ignoring sign).
>
> - Variance = 1: numbers are typically around +/-1. Like a gentle wave.
> - Variance = 64: numbers are typically around +/-8 (sigma = sqrt(64) = 8). Much wilder.
> - Variance = 0.0004: numbers are typically around +/-0.02. Almost flat.
>
> When we say "we want variance = 1," we mean: *keep the numbers in a moderate range where nothing explodes or vanishes.*

> **[APPLICATION]** Variance in our Proust model
>
> Our embedding matrix is initialized with `np.random.randn(...) * 0.02`, which means:
> ```
> Each element ~ N(0, 0.02^2) = N(0, 0.0004)
> ```
> So each embedding element is typically +/-0.02. Very small on purpose -- to prevent
> activations from exploding through the network layers on the very first forward pass.

---

## 2. The Core Theorem: Variance of a Sum

This is the single most important result for understanding sqrt(d). Everything else follows from it.

> **[FORMAL]** Theorem: Variance of a sum of independent variables
>
> Let X_1, X_2, ..., X_d be **independent** random variables, each with mean 0 and variance sigma^2. Then:
>
> ```
> S = X_1 + X_2 + ... + X_d
> ```
>
> has:
> ```
> E[S] = 0
> Var(S) = d * sigma^2
> ```
>
> **Proof:** Since the variables are independent, variances add:
> ```
> Var(S) = Var(X_1) + Var(X_2) + ... + Var(X_d)
>        = sigma^2 + sigma^2 + ... + sigma^2
>        = d * sigma^2
> ```
> Standard deviation of the sum: `sigma_S = sqrt(d * sigma^2) = sqrt(d) * sigma`.

> **[INTUITION]** Why sums get bigger
>
> Imagine flipping a coin that says +1 or -1. After 1 flip, you could be at +/-1.
> After 4 flips? You might be at +2 or -2 (not +/-4). After 64 flips? Typically around +/-8.
>
> The pattern: after d flips, you're typically at **+/- sqrt(d)**.
>
> This is the **random walk**: each step is random, but the accumulation grows as sqrt(d), not as d. It's a fundamental law of probability.
>
> | Number of terms (d) | sqrt(d) | Typical magnitude of sum |
> |---------------------|---------|--------------------------|
> | 1                   | 1.0     | +/-1                     |
> | 4                   | 2.0     | +/-2                     |
> | 16                  | 4.0     | +/-4                     |
> | 64                  | 8.0     | +/-8                     |
> | 512                 | 22.6    | +/-23                    |
>
> The sum of d independent random numbers has standard deviation sqrt(d) times bigger than each individual number. This is why sqrt(d) is the natural correction factor.

> **[APPLICATION]** In our 64-dimensional model
>
> Each embedding vector has `d_model = 64` elements. If each element has sigma = 0.02, then any operation that sums all 64 elements produces a result with:
> ```
> sigma_sum = sqrt(64) * 0.02 = 8 * 0.02 = 0.16
> ```
> The sum is 8x bigger than each element. That factor of 8 is sqrt(d_model).

---

## 3. Dot Products: Where sqrt(d) Becomes Critical

The dot product is the most important operation in transformers. Attention scores, embedding lookups, and linear projections all reduce to dot products. And dot products suffer directly from the variance accumulation we just described.

> **[FORMAL]** Theorem: Variance of a dot product
>
> Let **q** = (q_1, ..., q_d) and **k** = (k_1, ..., k_d) be two random vectors where all elements are **independent** with mean 0 and variance sigma^2.
>
> The dot product `q . k = sum(q_i * k_i, i=1..d)` has:
>
> ```
> E[q . k] = 0
> Var(q . k) = d * sigma^4
> ```
>
> **Proof:**
>
> Each term of the sum is `Z_i = q_i * k_i`.
>
> *Step 1: Mean of each term.*
> Since q_i and k_i are independent and both have mean 0:
> ```
> E[q_i * k_i] = E[q_i] * E[k_i] = 0 * 0 = 0
> ```
>
> *Step 2: Variance of each term.*
> For independent zero-mean variables:
> ```
> Var(q_i * k_i) = E[q_i^2 * k_i^2] - (E[q_i * k_i])^2
>                = E[q_i^2] * E[k_i^2] - 0
>                = sigma^2 * sigma^2
>                = sigma^4
> ```
>
> *Step 3: Variance of the sum.*
> The d terms Z_1, ..., Z_d are independent, so:
> ```
> Var(sum of Z_i) = d * sigma^4
> ```
> Standard deviation: `sqrt(d) * sigma^2`.

> **[INTUITION]** What this means in plain language
>
> A dot product sums d terms. Each term is small (a product of two small numbers), but there are d of them. By the "variance of sums" theorem, the result is sqrt(d) times bigger than any single term.
>
> **Concrete example with d = 64 and sigma = 1:**
>
> Each term `q_i * k_i` has variance `1^4 = 1`.
> The sum of 64 such terms has variance `64 * 1 = 64`, so `sigma_dot = sqrt(64) = 8`.
>
> This means dot product values are typically around +/-8 -- not +/-1.
>
> **The critical question**: if we want the dot product to have variance 1 (so that downstream operations like softmax work well), how do we fix this?
>
> **Answer**: divide by sqrt(d):
> ```
> Var((q . k) / sqrt(d)) = Var(q . k) / d = (d * sigma^4) / d = sigma^4
> ```
> For sigma = 1: variance becomes 1. The scaling perfectly compensates.

> **[APPLICATION]** In our attention mechanism (`src/attention.py:224`)
>
> ```python
> scores = Q @ K.transpose(0, 1, 3, 2)   # (1, 2, 4, 32) @ (1, 2, 32, 4) = (1, 2, 4, 4)
> scores = scores / np.sqrt(self.d_k)     # divide by sqrt(32) = 5.66
> ```
>
> Here `d_k = 32` (per-head dimension). Each attention score is a dot product of 32-dimensional query and key vectors. Without dividing by `sqrt(32) ~ 5.66`, the scores would have standard deviation ~5.66 (assuming unit-variance elements), making softmax produce near-one-hot outputs.

---

## 4. Why Softmax Cares About Scale

> **[FORMAL]** The softmax function
>
> For a vector **z** = (z_1, ..., z_n):
> ```
> softmax(z_i) = exp(z_i) / sum(exp(z_j), j=1..n)
> ```
> The output is a probability distribution: all values are in [0, 1] and sum to 1.
>
> **Key property:** softmax is sensitive to the *magnitude* of its inputs.
> - Small inputs z ~ (0.1, 0.2, 0.1): output ~ (0.32, 0.36, 0.32) -- uniform, spread out.
> - Large inputs z ~ (1, 20, 1): output ~ (0.0, 1.0, 0.0) -- one-hot, all mass on one element.

> **[INTUITION]** Softmax as a "temperature dial"
>
> Think of softmax as a dial between two extremes:
>
> | Inputs near zero | | Inputs very large |
> |------------------|-|-------------------|
> | Uniform distribution | <---> | One-hot (argmax) |
> | "Consider everything equally" | | "Focus on one thing only" |
> | Gradients are healthy | | Gradients are near zero |
>
> When attention scores are too large (because we didn't scale), softmax becomes a hard argmax. The output says "attend 100% to position 3, 0% everywhere else." The gradient of this is essentially zero for all inputs except the maximum, so **the model cannot learn to adjust its attention pattern**.
>
> This is called **softmax saturation**: the function is "stuck" at the extremes.

> **[APPLICATION]** Numerical example with our d_k = 32
>
> Suppose **q** and **k** are 32-dimensional with elements ~ N(0, 1).
>
> **Without scaling:**
> ```
> score = q . k   =>   sigma_score = sqrt(32) ~ 5.66
> ```
> A typical score vector might be: [2.1, 15.3, -4.7, 8.2].
>
> `softmax([2.1, 15.3, -4.7, 8.2]) ~ [0.000, 0.999, 0.000, 0.001]`
>
> Almost one-hot. Gradient ~ 0 for all but one position.
>
> **With scaling by sqrt(32):**
> ```
> score = (q . k) / sqrt(32)   =>   sigma_score = 1
> ```
> A typical score vector might be: [0.4, 2.7, -0.8, 1.4].
>
> `softmax([0.4, 2.7, -0.8, 1.4]) ~ [0.06, 0.60, 0.02, 0.17]`
>
> Smooth distribution. All positions get meaningful gradient. The model can learn.

---

## 5. The Other sqrt(d): Embedding Scaling

Now we connect to the embedding layer. This is a *different* appearance of sqrt(d), but it comes from the same mathematical root.

> **[FORMAL]** Embedding scaling in the paper
>
> Section 3.4 of "Attention Is All You Need" (Vaswani et al., 2017):
>
> *"In the embedding layers, we multiply those weights by sqrt(d_model)."*
>
> The embedding forward pass is:
> ```
> output = E[token_ids] * sqrt(d_model)  +  PE[position]
>          \_________________________/      \____________/
>             scaled token embedding      positional encoding
> ```
> where E is the learned embedding matrix and PE is the fixed sinusoidal encoding.

> **[INTUITION]** Why embeddings need scaling: the magnitude mismatch
>
> Two signals are being **added** together in the embedding layer:
> 1. **Token embedding**: what character is this? (learned)
> 2. **Positional encoding**: where is it in the sequence? (fixed, sin/cos)
>
> For addition to be meaningful, both signals must have **comparable magnitudes**. If one is 50x bigger, the other is effectively invisible in the sum.
>
> The positional encoding uses sin and cos, which produce values in [-1, +1]. Each element has a typical absolute value around 0.7 (the RMS of a sine wave is 1/sqrt(2)).
>
> The token embedding is initialized small. How small depends on the scheme:
>
> | Initialization | Per-element sigma | Vector L2 norm (~sqrt(d) * sigma) |
> |----------------|-------------------|-----------------------------------|
> | Our NumPy (x 0.02) | 0.02 | sqrt(64) * 0.02 = 0.16 |
> | Paper (~N(0, 1/sqrt(d))) | 0.125 | sqrt(64) * 0.125 = 1.0 |
> | PyTorch nn.Embedding default | 1.0 | sqrt(64) * 1.0 = 8.0 |
> | Positional encoding | ~0.7 | sqrt(64) * 0.7 = 5.6 |
>
> Without scaling, our NumPy embeddings have norm 0.16 vs positional encoding norm 5.6. The ratio is 35:1 -- position completely drowns out content.
>
> After multiplying by sqrt(d_model) = sqrt(64) = 8:
> ```
> Scaled embedding norm = 8 * 0.16 = 1.28
> ```
> Now the ratio is 5.6 : 1.28 ~ 4.4 : 1. Not perfectly balanced, but the content signal is now **audible**. During training, the embedding weights grow and the balance improves further.

> **[APPLICATION]** In our code: `src/embedding.py:95-100`
>
> ```python
> # Step 1: Token embedding lookup
> token_emb = self.token_embedding[token_ids]       # (batch, seq, 64)
>
> # Scale embeddings by sqrt(d_model) as in original paper
> token_emb = token_emb * np.sqrt(self.d_model)     # multiply by 8.0
>
> # Step 2: Add positional encoding
> pos_emb = self.pos_encoding[:seq_len, :]           # (seq, 64)
> output = token_emb + pos_emb                       # (batch, seq, 64)
> ```
>
> Before our bug fix, the `np.sqrt(self.d_model)` line was missing. The PyTorch version (`src/model_torch.py:145`) always had it:
> ```python
> token_emb = token_emb * math.sqrt(self.d_model)   # multiply by 8.0
> ```

> **[DIMENSION CHECK]**
>
> | Stage | Shape | Per-element magnitude |
> |-------|-------|-----------------------|
> | Token IDs | `(batch, seq)` | integers in [0, 95] |
> | Raw embedding lookup | `(batch, seq, 64)` | ~0.02 |
> | After x sqrt(64) | `(batch, seq, 64)` | ~0.16 |
> | Positional encoding | `(seq, 64)` | ~0.7 |
> | Sum (embedding + positional) | `(batch, seq, 64)` | ~0.7-0.9 |

---

## 6. Unifying the Two Scalings

> **[FORMAL]** The two sqrt(d) scalings in a transformer
>
> 1. **Embedding layer**: multiply by sqrt(d_model)
>    *Purpose*: scale UP embeddings to match positional encoding magnitude.
>
> 2. **Attention layer**: divide by sqrt(d_k)
>    *Purpose*: scale DOWN dot products to prevent softmax saturation.
>
> Both arise from the same mathematical fact:
> ```
> ┌─────────────────────────────────────────────────┐
> │  Var(sum of d X_i's) = d * Var(X_i)             │
> └─────────────────────────────────────────────────┘
> ```
> When you sum d random terms, the result has standard deviation sqrt(d) times the standard deviation of each term. The correction factor is always sqrt(d).

> **[INTUITION]** One formula, two directions
>
> The sqrt(d) factor is like a "dimension tax" -- the price you pay for working in high dimensions.
>
> - In **embeddings**, we PAY the tax voluntarily: multiply by sqrt(d) to boost the signal up to where positional encodings live. We *want* the embedding to be bigger.
>
> - In **attention**, we UNDO the tax: divide by sqrt(d) to bring the dot product back down to a safe range for softmax. We *don't want* the score to be bigger.
>
> Same sqrt(d), opposite directions, same reason.

> **[APPLICATION]** Full signal flow through our Proust model
>
> Tracing the magnitude of a single value through the entire architecture:
>
> | Stage | Code location | Per-element sigma |
> |-------|---------------|-------------------|
> | Embedding init | `embedding.py:38` | 0.02 |
> | x sqrt(64) scaling | `embedding.py:100` | 0.02 * 8 = 0.16 |
> | + positional encoding | `embedding.py:109` | ~0.7 |
> | -> into attention | `attention.py:199` | ~0.7 |
> | Q, K projection (x W) | `attention.py:199` | sqrt(64) * 0.7 * 0.02 ~ 0.11 |
> | Dot product Q . K | `attention.py:221` | sqrt(32) * 0.11^2 ~ 0.07 |
> | / sqrt(32) scaling | `attention.py:224` | 0.07 / 5.66 ~ 0.01 |
> | Softmax | `attention.py:240` | probabilities in [0, 1] |
>
> At every stage, the magnitudes stay in a "reasonable" range. No explosions, no vanishing. The two sqrt(d) scalings are two of the key mechanisms that ensure this.

---

## 7. The Deeper Connection: L2 Norm and Dimension

> **[FORMAL]** L2 norm of a random vector
>
> For a d-dimensional vector **v** where each element v_i has mean 0 and variance sigma^2:
> ```
> ||v||^2 = sum(v_i^2, i=1..d)
> ```
> Taking the expected value:
> ```
> E[||v||^2] = sum(E[v_i^2], i=1..d) = d * sigma^2
> ```
> So:
> ```
> ||v|| ~ sqrt(d) * sigma
> ```
> The L2 norm of a random vector scales as sqrt(d) -- same factor, same reason.

> **[INTUITION]** High dimensions are strange
>
> In 2D, a vector with elements ~ N(0, 1) has length ~ sqrt(2) ~ 1.4.
> In 64D, the same distribution gives length ~ sqrt(64) = 8.
> In 768D (GPT-2), it's ~ sqrt(768) ~ 27.7.
>
> High-dimensional vectors are **much longer** than you'd expect from their individual elements. This is not because any single element is large -- it's because there are *many small contributions* that accumulate via the Pythagorean theorem.
>
> This is why sqrt(d) appears so often in machine learning:
> - Xavier/Glorot initialization: `sigma = 1/sqrt(d_in)`
> - Attention scaling: `/ sqrt(d_k)`
> - Embedding scaling: `* sqrt(d_model)`
> - Layer normalization: normalize the sqrt(d)-sized vector back to unit variance
>
> They are all compensating for the same geometric reality of high dimensions.

> **[APPLICATION]** Numerical check with our model
>
> Our embedding vectors: d = 64, sigma = 0.02:
> ```
> ||e|| ~ sqrt(64) * 0.02 = 0.16
> ```
> Our positional encoding vectors: d = 64, elements in [-1, 1] with RMS ~ 0.7:
> ```
> ||p|| ~ sqrt(64) * 0.7 = 5.6
> ```
> Ratio without scaling: 5.6 / 0.16 = 35.
>
> After x sqrt(64) scaling: 5.6 / (0.16 * 8) = 5.6 / 1.28 = 4.4.
>
> Much better -- and training will close the remaining gap.

---

## 8. Summary Table

| Where | Formula | Dimension | Why |
|-------|---------|-----------|-----|
| Embedding layer | `e * sqrt(d_model)` | d_model = 64 | Scale up to match positional encoding magnitude |
| Attention scores | `(q . k) / sqrt(d_k)` | d_k = 32 | Scale down to prevent softmax saturation |
| Xavier init | `sigma = 1 / sqrt(d_in)` | d_in = input dim | Keep layer output variance ~ input variance |
| LayerNorm | Normalize to mu=0, sigma=1 across d_model | d_model = 64 | Undo any accumulated magnitude drift |

All four are different manifestations of the same underlying principle: **when you sum or combine d independent numbers, the result grows by sqrt(d), and you must compensate.**

---

## 9. Key Identities and Properties

1. `Var(a*X) = a^2 * Var(X)` -- scaling by constant a
2. `Var(X + Y) = Var(X) + Var(Y)` -- if X, Y independent
3. `Var(X * Y) = Var(X) * Var(Y)` -- if X, Y independent, zero-mean
4. `Var(sum of d X_i's) = d * sigma^2` -- i.i.d. variables, the core theorem
5. `Var((1/sqrt(d)) * sum of d X_i's) = sigma^2` -- the fix: divide by sqrt(d)
6. `||v|| ~ sqrt(d) * sigma` -- L2 norm of a random d-dimensional vector

Identities 4 and 5 are the entire story. Everything else is a special case.

---

## 10. What This Document Does Not Cover (Yet)

- **Gradient flow**: how sqrt(d) affects the backward pass and gradient magnitudes (related, but requires chain rule derivations).
- **Pre-norm vs post-norm**: how the placement of LayerNorm interacts with these scaling choices.
- **RoPE and ALiBi**: modern positional encodings that avoid the additive approach entirely.
- **FlashAttention**: how fused kernels handle the scaling internally.
- **Non-unit-variance initializations**: what happens when the i.i.d. assumption breaks (e.g., after several layers of training when weights are no longer near initialization).

---

*"The real voyage of discovery consists not in seeking new landscapes, but in having new eyes."*

-- Marcel Proust, *La Prisonniere*
