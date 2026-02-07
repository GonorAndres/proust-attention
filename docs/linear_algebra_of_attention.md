# The Linear Algebra of Attention
### A Complete Mathematical Reference for the Proust Attention Machine

> This document maps every linear algebra operation in our transformer architecture, from character embedding to attention output. Each concept is presented in three layers: **formal definition** (blue), **geometric intuition** (green), and **transformer application** (orange). Read it to connect your knowledge of linear algebra to what the Proust machine actually computes.

---

## Table of Contents

1. [Notation and Conventions](#1-notation-and-conventions)
2. [Vectors and Vector Spaces](#2-vectors-and-vector-spaces)
3. [The Embedding Lookup](#3-the-embedding-lookup-indexing-into-a-matrix)
4. [Vector Addition: Combining Embeddings with Position](#4-vector-addition-combining-embeddings-with-position)
5. [Linear Transformations](#5-linear-transformations-the-heart-of-everything)
6. [The Q, K, V Projections](#6-the-q-k-v-projections)
7. [The Dot Product: Measuring Compatibility](#7-the-dot-product-measuring-compatibility)
8. [Scaling: Variance Control](#8-scaling-variance-control)
9. [The Causal Mask](#9-the-causal-mask-constraining-the-score-matrix)
10. [Softmax: From Scores to Probability Distribution](#10-softmax-from-scores-to-probability-distribution)
11. [Weighted Sum: Retrieving Information](#11-weighted-sum-retrieving-information)
12. [Reshape and Transpose: Multi-Head Mechanics](#12-reshape-and-transpose-multi-head-mechanics)
13. [The Output Projection](#13-the-output-projection-mixing-across-heads)
14. [The Complete Attention Formula](#14-the-complete-attention-formula)
15. [Summary: Every Operation at a Glance](#15-summary-every-operation-at-a-glance)
16. [Key Identities and Properties](#16-key-identities-and-properties)
17. [What This Document Does Not Cover (Yet)](#17-what-this-document-does-not-cover-yet)

---

## 1. Notation and Conventions

> **[FORMAL]** Notation Table

| Symbol | Meaning | Value in our model |
|---|---|---|
| V | Vocabulary size (number of unique characters) | ~80 |
| d | Model dimension (`d_model`) | 64 |
| d_k | Dimension per attention head | 32 |
| n_h | Number of attention heads | 2 |
| T | Sequence length (`seq_len`) | <= 256 |
| B | Batch size | variable |
| R^{m x n} | Set of all real matrices with m rows and n columns | --- |
| A^T | Transpose of matrix A | --- |
| AB | Matrix multiplication of A and B | --- |
| <x, y> | Dot product (inner product) of vectors x and y | --- |

> **[INTUITION]** How to Read Shapes

Throughout this document, we annotate shapes as `(B, T, d)` meaning:
- **First axis**: which sequence in the batch (B sequences)
- **Second axis**: which position in the sequence (T positions)
- **Third axis**: which feature/dimension (d numbers per position)

A matrix multiplication `(B, T, **m**) x (**m**, n) = (B, T, n)` requires the **inner dimensions** to match (both are m). The outer dimensions survive.

---

## 2. Vectors and Vector Spaces

> **[FORMAL]** Vector in R^n

A **vector** x in R^n is an ordered list of n real numbers:

```
x = (x_1, x_2, ..., x_n) in R^n
```

The set R^n forms a **vector space**: vectors can be added and scaled, and the results remain in R^n.

> **[INTUITION]** Vectors as Points in Space

A vector in R^64 is a point in 64-dimensional space. We can't visualize 64 dimensions, but the math works identically to 2D or 3D:
- Two vectors can be **close** (similar) or **far** (different)
- They can **point in the same direction** (aligned) or **perpendicular** (unrelated)
- We can **add** them (combine information) and **scale** them (amplify/diminish)

Every character in our model is represented as a point in R^64.

> **[APPLICATION]** Embedding Vectors

Each character in the vocabulary gets a vector e_i in R^64. After training, characters that behave similarly in Proust's text will have vectors that are close together in this 64-dimensional space. Vowels might cluster. Consonants might cluster. Punctuation in another region. At initialization, these vectors are random -- the structure emerges from training.

---

## 3. The Embedding Lookup: Indexing Into a Matrix

> **[FORMAL]** Embedding as Row Selection

Let E in R^{V x d} be the **embedding matrix**, where row i is the embedding vector for character i in the vocabulary.

Given a sequence of token IDs (t_1, t_2, ..., t_T) where each t_j in {0, 1, ..., V-1}, the embedding operation selects rows:

```
X_tok = [ e_{t_1}^T ]
        [ e_{t_2}^T ]    in R^{T x d}
        [   ...      ]
        [ e_{t_T}^T ]
```

This is equivalent to multiplication by a one-hot matrix: `X_tok = H * E` where H in {0,1}^{T x V} and H_{j,i} = 1 iff t_j = i.

> **[DIMENSION CHECK]**

```
    H          x     E         =    X_tok
(T x V)       x  (V x d)      =   (T x d)
```

Inner dimension V is consumed. Each of T positions gets a d-dimensional vector.

> **[INTUITION]** Lookup vs. Multiplication

Selecting row 5 from a matrix is the same as multiplying by a one-hot vector [0,0,0,0,0,1,0,...]. In code, we use direct indexing (`E[token_ids]`) because it's faster than constructing a one-hot matrix. But mathematically, embedding *is* a matrix multiplication -- which is why it's differentiable and learnable in gradient terms.

> **[APPLICATION]** In the Proust Machine

`token_embedding[token_ids]` on line 95 of `embedding.py`:
- Input: `(B, T)` integers -- each integer is a character ID
- Table: `(80, 64)` -- one row of 64 numbers per character
- Output: `(B, T, 64)` -- each character is now a 64-dim vector

---

## 4. Vector Addition: Combining Embeddings with Position

> **[FORMAL]** Vector Addition

For vectors x, y in R^n, their sum is defined element-wise:

```
x + y = (x_1 + y_1, x_2 + y_2, ..., x_n + y_n) in R^n
```

The result lies in the same space R^n. Dimensions must match exactly.

> **[FORMAL]** Positional Encoding

The positional encoding matrix P in R^{T_max x d} is defined by:

```
P(pos, 2i)   = sin(pos / 10000^{2i/d})
P(pos, 2i+1) = cos(pos / 10000^{2i/d})
```

for position pos in {0, ..., T_max - 1} and dimension index i in {0, ..., d/2 - 1}.

The combined input is: `X = X_tok + P[:T, :] in R^{T x d}`

> **[INTUITION]** Why Addition Works

Addition blends two signals into one vector. Imagine two radio stations broadcasting on different frequencies -- you can receive both simultaneously because their signals don't interfere destructively. Similarly, the learned token embedding and the fixed sin/cos positional encoding occupy different "frequency bands" in the 64 dimensions.

After addition, dimension 0 might be 80% character information and 20% position. Dimension 47 might be the reverse. The subsequent projection matrices (W_Q, W_K, W_V) can learn to separate or combine these signals as needed.

> **[DIMENSION CHECK]**

```
  X_tok      +    P       =      X
(T x d)      + (T x d)    =   (T x d)
```

Both operands must have identical shape. The output has the same shape. Addition does not change dimensionality.

---

## 5. Linear Transformations: The Heart of Everything

> **[FORMAL]** Linear Transformation

A **linear transformation** (or linear map) is a function f: R^n -> R^m that satisfies:

```
f(x + y) = f(x) + f(y)       (additivity)
f(alpha * x) = alpha * f(x)  (homogeneity)
```

Every linear transformation can be represented as multiplication by a matrix W in R^{n x m}:

```
f(x) = x * W    where x in R^{1 x n}, f(x) in R^{1 x m}
```

The matrix W completely defines the transformation. Its columns are the images of the standard basis vectors.

> **[INTUITION]** What a Matrix Does to a Vector

Multiplying a vector by a matrix does three things geometrically:
1. **Rotates** -- changes the direction the vector points
2. **Stretches/compresses** -- changes the magnitude along different axes
3. **Projects** -- can reduce or increase dimensionality

A (64, 32) matrix takes a 64-dim vector and produces a 32-dim vector. It "squeezes" the information from 64 dimensions into 32, keeping whatever the matrix has learned is most important. Information is necessarily lost (64 dimensions into 32 cannot be invertible), but training iterations ensures the *useful* information is preserved.

> **[FORMAL]** Matrix Multiplication: The Dimension Rule

For matrices A in R^{m x n} and B in R^{n x p}:

```
C = A * B in R^{m x p},    C_ij = sum_{k=1}^{n} A_ik * B_kj
```

**The critical rule**: the inner dimensions must match (n = n). The outer dimensions (m and p) survive to become the output shape.

```
    A          x       B       =       C
(m x **n**)   x  (**n** x p)   =    (m x p)
```

If the inner dimensions don't match, the operation is **undefined**.

> **[INTUITION]** Three Ways to See Matrix Multiplication

1. **Row-column dot products**: entry C_ij is the dot product of row i of A with column j of B.
2. **Transformation of rows**: each row of A is a vector that gets transformed by B.
3. **Linear combination of columns**: each column of C is a linear combination of columns of A, with weights from B.

In transformers, interpretation (2) is most useful: each position's vector (a row of X) is independently transformed by the weight matrix.

---

## 6. The Q, K, V Projections

> **[FORMAL]** Three Learned Projections

Given input X in R^{T x d} and three weight matrices:

```
W_Q in R^{d x d},    W_K in R^{d x d},    W_V in R^{d x d}
```

the Query, Key, and Value matrices are:

```
Q = X * W_Q in R^{T x d}
K = X * W_K in R^{T x d}
V = X * W_V in R^{T x d}
```

> **[DIMENSION CHECK]**

```
     X          x      W_Q       =       Q
(T x **d**)    x  (**d** x d)    =    (T x d)
```

Inner dimension d=64 is consumed. Each of T positions produces a new d-dimensional vector. The same input X is multiplied by three *different* matrices to produce three *different* outputs.

> **[INTUITION]** What Each Projection Does Geometrically

Each weight matrix rotates and stretches the 64-dimensional input into a new 64-dimensional space:
- **W_Q** rotates into "question space" -- dimensions optimized for expressing what information is needed
- **W_K** rotates into "key space" -- dimensions optimized for advertising what information is available
- **W_V** rotates into "value space" -- dimensions optimized for carrying useful content

The same input vector x_i is viewed through three different "lenses." Each lens extracts different aspects of the combined character+position information.

> **[APPLICATION]** In the Proust Machine

`attention.py` lines 199-201:

```python
Q = x @ self.W_Q   # (1, 13, 64) @ (64, 64) = (1, 13, 64)
K = x @ self.W_K   # (1, 13, 64) @ (64, 64) = (1, 13, 64)
V = x @ self.W_V   # (1, 13, 64) @ (64, 64) = (1, 13, 64)
```

Three separate applications of the same operation (matrix multiply) with three different learned matrices. The batch dimension B passes through unchanged -- each sequence in the batch is processed independently.

---

## 7. The Dot Product: Measuring Compatibility

> **[FORMAL]** Dot Product (Inner Product)

For vectors x, y in R^n:

```
<x, y> = x^T * y = sum_{i=1}^{n} x_i * y_i    (a scalar in R)
```

Geometric form:

```
<x, y> = ||x|| * ||y|| * cos(theta)
```

where theta is the angle between the vectors and ||x|| = sqrt(sum x_i^2) is the Euclidean norm.

> **[INTUITION]** What the Dot Product Means

The dot product combines magnitude and alignment into one number:

| Geometric Relationship | Dot Product |
|---|---|
| Same direction (theta = 0) | Large positive |
| Perpendicular (theta = 90) | Zero |
| Opposite direction (theta = 180) | Large negative |

The dot product does not inherently measure "meaning similarity." It measures geometric alignment. In the transformer, the Q and K projections *create* a space where alignment corresponds to usefulness -- this correspondence is "learned" by testing, not built in.

> **[FORMAL]** All Pairwise Dot Products via Matrix Multiplication

To compute the dot product between every pair of rows from Q in R^{T x d_k} and K in R^{T x d_k}:

```
S = Q * K^T in R^{T x T},    S_ij = <q_i, k_j> = sum_{l=1}^{d_k} Q_il * K_jl
```

Entry S_ij is the dot product of Query at position i with Key at position j: how compatible position i's question is with position j's advertisement.

> **[DIMENSION CHECK]**

```
     Q            x      K^T          =       S
(T x **d_k**)    x  (**d_k** x T)     =    (T x T)
```

The feature dimension d_k is consumed. What remains is a T x T matrix: every position scored against every other position. This is why attention has O(T^2) complexity -- the score matrix grows quadratically with sequence length.

> **[APPLICATION]** In the Proust Machine

`attention.py` line 221:

```python
scores = Q @ K.transpose(0, 1, 3, 2)
# (1, 2, 13, 32) @ (1, 2, 32, 13) = (1, 2, 13, 13)
```

The transpose swaps the last two dimensions of K, turning each head's (T, d_k) into (d_k, T). The batch and head dimensions pass through unchanged. The result is a (13, 13) compatibility matrix *per head*.

---

## 8. Scaling: Variance Control

> **[FORMAL]** Scaled Dot-Product

```
S_scaled = (Q * K^T) / sqrt(d_k)
```

If entries of Q and K have variance sigma^2, then entries of Q*K^T have variance d_k * sigma^4 (sum of d_k products). Dividing by sqrt(d_k) restores variance to sigma^4, independent of dimension.

> **[INTUITION]** Why sqrt(d_k)?

Without scaling, larger d_k means larger dot products, which means softmax saturates (outputs near 0 or 1), which means gradients vanish, which means the model stops learning. The sqrt(d_k) is not a hyperparameter to tune -- it's a mathematical correction derived from the statistics of random dot products.

In our model: sqrt(d_k) = sqrt(32) ~ 5.66. Every score is divided by 5.66.

---

## 9. The Causal Mask: Constraining the Score Matrix

> **[FORMAL]** Causal Mask

Define the mask matrix M in R^{T x T}:

```
M_ij = 0       if j <= i    (past and present)
M_ij = -inf    if j > i     (future)
```

Applied by addition: `S_masked = S_scaled + M`

> **[INTUITION]** Why Addition of -inf Works

Adding -inf to a score means exp(-inf) = 0 in the subsequent softmax. The future position gets exactly zero attention weight. This is more elegant than deleting entries -- the matrix stays the same shape, and the operation is differentiable everywhere except at the masked positions (where we don't need gradients anyway).

The mask is a lower-triangular matrix of zeros with -inf above the diagonal:

```
M = [  0    -inf  -inf  -inf ]
    [  0     0    -inf  -inf ]
    [  0     0     0    -inf ]
    [  0     0     0     0   ]
```

---

## 10. Softmax: From Scores to Probability Distribution

> **[FORMAL]** Softmax Function

For a vector z in R^n, the softmax function maps to the probability simplex:

```
softmax(z)_i = exp(z_i) / sum_{j=1}^{n} exp(z_j)
```

Properties:
- `softmax(z)_i > 0` for all i (always positive)
- `sum_i softmax(z)_i = 1` (sums to one)
- Monotone: if z_i > z_j then softmax(z)_i > softmax(z)_j (preserves order)
- `softmax(z + c) = softmax(z)` for any constant c (shift-invariant)

> **[FORMAL]** Numerically Stable Implementation

```
softmax(z)_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))
```

This is mathematically identical (by shift-invariance) but prevents overflow, since the largest exponent is exp(0) = 1.

> **[INTUITION]** Softmax as Maximum-Entropy Allocation

Softmax is not an arbitrary normalization. It is the **unique** probability distribution that:
1. Respects the ordering of scores (higher score = higher probability)
2. Maximizes entropy (makes the fewest assumptions beyond what scores dictate)

This is a theorem from information theory. Any other normalization would inject unjustified bias into how attention weight is allocated.

> **[APPLICATION]** Applied Row-wise to the Score Matrix

```
A = softmax( (Q * K^T) / sqrt(d_k) + M ) in R^{T x T}
```

Each row A_{i,:} is an independent probability distribution over context positions. Row i sums to 1 and represents: "how much attention does position i pay to each previous position."

```python
attention_weights = softmax(scores, axis=-1)
# (1, 2, 13, 13) -- each row of the last two dims sums to 1
```

---

## 11. Weighted Sum: Retrieving Information

> **[FORMAL]** Convex Combination of Vectors

A **convex combination** of vectors v_1, ..., v_T in R^{d_k} with weights alpha_1, ..., alpha_T >= 0 summing to 1 is:

```
o = sum_{j=1}^{T} alpha_j * v_j = alpha_1*v_1 + alpha_2*v_2 + ... + alpha_T*v_T in R^{d_k}
```

The result lies within the **convex hull** of the input vectors -- it cannot escape the region spanned by them. It is a "blend" that lives somewhere "in between" the input vectors, positioned by the weights.

> **[FORMAL]** As Matrix Multiplication

Computing the convex combination for *all* positions simultaneously:

```
O = A * V in R^{T x d_k}
```

Row i of O is the convex combination of all rows of V, weighted by row i of A:

```
o_i = sum_{j=1}^{T} A_ij * v_j
```

> **[DIMENSION CHECK]**

```
     A          x      V           =       O
(T x **T**)    x  (**T** x d_k)    =   (T x d_k)
```

The sequence dimension T is consumed. Each position's output is a d_k-dimensional vector that blends information from all attended positions.

> **[INTUITION]** Soft Selection

If the attention weights were one-hot (one entry is 1, rest are 0), this would be *hard* selection: just copy one Value vector. Softmax makes it *soft*: a weighted average. This softness is critical -- it makes the operation differentiable, which means gradients can flow through and the model can learn *which* positions to attend to, using gradient.

> **[APPLICATION]** In the Proust Machine

`attention.py` line 248:

```python
head_outputs = attention_weights @ V
# (1, 2, 13, 13) @ (1, 2, 13, 32) = (1, 2, 13, 32)
```

Each head independently computes its weighted retrieval. The batch and head dimensions pass through unchanged. Each position's 32-dim output encodes context gathered according to that head's attention pattern.

---

## 12. Reshape and Transpose: Multi-Head Mechanics

> **[FORMAL]** Tensor Reshape

A **reshape** reinterprets the memory layout without changing the data. For a matrix A in R^{T x d} where d = n_h * d_k:

```
reshape(A, [T, n_h, d_k]) in R^{T x n_h x d_k}
```

The total number of elements is unchanged: T * d = T * n_h * d_k. No computation occurs -- only the "shape label" on the data changes.

> **[FORMAL]** Transpose (Axis Permutation)

A **transpose** reorders the axes of a tensor. For a 3D tensor A in R^{T x n_h x d_k}:

```
transpose(A, [1, 0, 2]) in R^{n_h x T x d_k}
```

This moves the head axis to the front, so we can compute attention for all heads in a single batched matrix multiplication.

> **[DIMENSION CHECK]**

Split heads:
```
Q: (B, T, d) --reshape--> (B, T, n_h, d_k) --transpose--> (B, n_h, T, d_k)

In our model:
(1, 13, 64) --> (1, 13, 2, 32) --> (1, 2, 13, 32)
```

Merge heads (reverse):
```
O: (B, n_h, T, d_k) --transpose--> (B, T, n_h, d_k) --reshape--> (B, T, d)

(1, 2, 13, 32) --> (1, 13, 2, 32) --> (1, 13, 64)
```

> **[INTUITION]** Why Move Heads Before Sequence?

After transposing to (B, n_h, T, d_k), the last two dimensions are (T, d_k) -- exactly the shape needed for attention: T positions, each with d_k features. The batch and head dimensions sit in front, meaning NumPy/PyTorch can process all heads in all batches simultaneously with a single matrix multiplication call.

Without the transpose, we'd need a loop over heads. With it, one `@` does everything.

---

## 13. The Output Projection: Mixing Across Heads

> **[FORMAL]** Output Projection

After merging heads, the concatenated output O_cat in R^{T x d} (where the first d_k dimensions are head 1, the next d_k are head 2) is projected:

```
O_final = O_cat * W_O,    W_O in R^{d x d}
```

> **[DIMENSION CHECK]**

```
   O_cat        x      W_O       =    O_final
(T x **d**)    x  (**d** x d)    =    (T x d)
```

> **[INTUITION]** Why This Projection Is Necessary

Before W_O, the 64 dimensions are rigidly split: dims 0-31 are head 1, dims 32-63 are head 2. They can't interact. W_O is a full (64, 64) matrix that creates *linear combinations across all 64 dimensions*, allowing head 1's findings to modulate how head 2's findings are interpreted.

Without W_O, multi-head attention would just be two independent single-head attentions glued together. With W_O, the heads become a coordinated system.

---

## 14. The Complete Attention Formula

> **[FORMAL]** Single-Head Attention

```
Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) + M ) * V    in R^{T x d_k}
```

> **[FORMAL]** Multi-Head Attention

```
head_h = Attention(X * W_Q^(h), X * W_K^(h), X * W_V^(h))    in R^{T x d_k}

MultiHead(X) = Concat(head_1, ..., head_{n_h}) * W_O           in R^{T x d}
```

> **[FORMAL]** Complete Dimension Flow

```
X               in R^{T x d}                    input
Q = X * W_Q     in R^{T x d}                    projection
Q -> Q^(h)      in R^{n_h x T x d_k}            split heads
Q^(h) * K^(h)^T in R^{n_h x T x T}             score matrix
/ sqrt(d_k) + M in R^{n_h x T x T}             scale + mask
softmax(.)  = A in R^{n_h x T x T}             attention weights
A * V^(h)       in R^{n_h x T x d_k}           weighted retrieval
merge, * W_O    in R^{T x d}                    output
```

**Input shape equals output shape: R^{T x d} -> R^{T x d}.**

---

## 15. Summary: Every Operation at a Glance

| Operation | Linear Algebra | Dimensions | What It Does |
|---|---|---|---|
| Embedding lookup | Row selection (H*E) | (T,V) x (V,d) = (T,d) | Discrete symbols to continuous vectors |
| Position addition | Vector addition | (T,d) + (T,d) = (T,d) | Injects position information |
| Q, K, V projection | Matrix multiply | (T,d) x (d,d) = (T,d) | Rotates into Q/K/V spaces |
| Head split | Reshape + transpose | (T,d) -> (n_h,T,d_k) | Separates heads for parallel attention |
| Score computation | Q*K^T | (T,d_k) x (d_k,T) = (T,T) | All pairwise compatibilities |
| Scaling | Scalar division | (T,T) / sqrt(d_k) = (T,T) | Variance normalization |
| Masking | Matrix addition | (T,T) + (T,T) = (T,T) | Blocks future positions |
| Softmax | R^T -> simplex (per row) | (T,T) -> (T,T) | Scores to probability distributions |
| Weighted sum | A*V | (T,T) x (T,d_k) = (T,d_k) | Context-weighted information retrieval |
| Head merge | Transpose + reshape | (n_h,T,d_k) -> (T,d) | Concatenate head outputs |
| Output projection | Matrix multiply | (T,d) x (d,d) = (T,d) | Mixes information across heads |

---

## 16. Key Identities and Properties

> **[FORMAL]** Properties Used in Attention

1. **Associativity**: (AB)C = A(BC) -- we can choose the most efficient multiplication order.

2. **Transpose of product**: (AB)^T = B^T A^T -- order reverses under transpose.

3. **Distributivity**: A(B + C) = AB + AC -- used implicitly when the residual connection adds back the input.

4. **Dot product as matrix multiply**: <x, y> = x^T * y -- a special case of matrix multiplication with (1, n) x (n, 1) = (1, 1).

5. **Batch broadcasting**: a (d, d) weight matrix applied to (B, T, d) acts independently on each of the B x T vectors. The batch and sequence dimensions are "spectators" that pass through unchanged.

6. **Softmax shift-invariance**: softmax(z + c) = softmax(z) -- enables the numerical stability trick.

7. **Convex combination stays in convex hull**: the output of A*V (with A being row-stochastic) lies within the convex hull of V's rows. Attention cannot create new information, only *blend* existing information.

---

## 17. What This Document Does Not Cover (Yet)

The following components also use linear algebra and will be documented as we implement them:

- **Layer Normalization**: normalization across the feature dimension, using mean and variance statistics. Involves element-wise operations and learned affine parameters.
- **Feedforward Network**: two linear transformations with a nonlinearity: `FFN(x) = W_2 * GELU(W_1 * x + b_1) + b_2`. Changes dimension d -> 4d -> d.
- **Residual Connection**: x + f(x), simple vector addition that enables gradient flow through deep networks.
- **Final Linear Projection**: (T, d) x (d, V) = (T, V), mapping from model space to vocabulary logits.
- **Backpropagation**: how gradients of the loss with respect to each weight matrix are computed using the chain rule and transposed Jacobians.

---

> *"The only real voyage of discovery consists not in seeking new landscapes, but in having new eyes."* -- Marcel Proust
