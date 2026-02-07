"""
Complete End-to-End Example: "Mañana será otro día"

We'll trace through every step of the transformer with real numbers.
"""

import numpy as np
np.random.seed(42)  # Reproducibility
np.set_printoptions(precision=2, suppress=True)

# =============================================================================
# THE PHRASE
# =============================================================================
phrase = "Mañana será otro día"
print("=" * 70)
print("COMPLETE TRANSFORMER EXAMPLE")
print("=" * 70)
print(f"\nPhrase: '{phrase}'")
print(f"Length: {len(phrase)} characters")

# =============================================================================
# STEP 1: TOKENIZER
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: TOKENIZER (Characters → Integer IDs)")
print("=" * 70)

# Build vocabulary from our phrase
unique_chars = sorted(list(set(phrase)))
char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}
vocab_size = len(unique_chars)

print(f"\nVocabulary ({vocab_size} unique characters):")
for char, idx in char_to_idx.items():
    display_char = repr(char) if char == ' ' else char
    print(f"  {display_char:4} → {idx}")

# Encode the phrase
token_ids = np.array([char_to_idx[c] for c in phrase])
print(f"\nEncoded phrase:")
print(f"  Characters: {list(phrase)}")
print(f"  Token IDs:  {token_ids}")

# Add batch dimension (batch_size = 1)
token_ids_batched = token_ids.reshape(1, -1)  # Shape: (1, 20)
print(f"\nWith batch dimension: shape = {token_ids_batched.shape}")

# =============================================================================
# STEP 2: EMBEDDING + POSITIONAL ENCODING
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: EMBEDDING (IDs → Vectors) + POSITIONAL ENCODING")
print("=" * 70)

# Small dimensions for readability
d_model = 8  # Normally 64, using 8 so we can see the numbers
seq_len = len(phrase)
batch_size = 1

# Token embedding matrix: each character gets a vector
# Shape: (vocab_size, d_model) = (15, 8)
token_embedding = np.random.randn(vocab_size, d_model) * 0.5

print(f"\nToken embedding matrix shape: ({vocab_size}, {d_model})")
print(f"  - {vocab_size} rows (one per unique character)")
print(f"  - {d_model} columns (embedding dimension)")

# Look up embeddings for our tokens
# token_ids_batched: (1, 20)
# After lookup: (1, 20, 8)
token_emb = token_embedding[token_ids_batched]

print(f"\nAfter embedding lookup: shape = {token_emb.shape}")
print(f"\nFirst 3 character embeddings (before positional encoding):")
for i in range(3):
    char = phrase[i]
    vec = token_emb[0, i, :]
    print(f"  '{char}' (id={token_ids[i]}): {vec}")

# Positional encoding (simplified: using small values for demo)
# Shape: (seq_len, d_model) = (20, 8)
positions = np.arange(seq_len)[:, np.newaxis]
dims = np.arange(d_model)[np.newaxis, :]
pos_encoding = np.sin(positions / (10 ** (dims / d_model))) * 0.1

print(f"\nPositional encoding shape: ({seq_len}, {d_model})")
print(f"Position 0 encoding: {pos_encoding[0]}")
print(f"Position 1 encoding: {pos_encoding[1]}")
print(f"Position 2 encoding: {pos_encoding[2]}")

# Add positional encoding to token embeddings
x = token_emb + pos_encoding  # Broadcasting: (1, 20, 8) + (20, 8)

print(f"\nAfter adding positional encoding: shape = {x.shape}")
print(f"\nFirst 3 characters with position info:")
for i in range(3):
    char = phrase[i]
    vec = x[0, i, :]
    print(f"  '{char}' at position {i}: {vec}")

print("\nNote: Same character at different positions would have different vectors!")

# =============================================================================
# STEP 3: MULTI-HEAD ATTENTION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: MULTI-HEAD ATTENTION")
print("=" * 70)

n_heads = 2
d_k = d_model // n_heads  # 8 // 2 = 4 dimensions per head

print(f"\nConfig:")
print(f"  d_model = {d_model}")
print(f"  n_heads = {n_heads}")
print(f"  d_k = {d_k} (dimension per head)")

# Weight matrices
W_Q = np.random.randn(d_model, d_model) * 0.3
W_K = np.random.randn(d_model, d_model) * 0.3
W_V = np.random.randn(d_model, d_model) * 0.3
W_O = np.random.randn(d_model, d_model) * 0.3

print(f"\nWeight matrices (all learned during training):")
print(f"  W_Q: {W_Q.shape} - creates Queries")
print(f"  W_K: {W_K.shape} - creates Keys")
print(f"  W_V: {W_V.shape} - creates Values")
print(f"  W_O: {W_O.shape} - combines heads")

# --- STEP 3a: Project to Q, K, V ---
print("\n--- Step 3a: Project to Q, K, V ---")

Q = x @ W_Q  # (1, 20, 8) @ (8, 8) = (1, 20, 8)
K = x @ W_K
V = x @ W_V

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")

print(f"\nQuery for 'M' (position 0): {Q[0, 0, :]}")
print(f"Key for 'M' (position 0):   {K[0, 0, :]}")
print(f"Value for 'M' (position 0): {V[0, 0, :]}")

# --- STEP 3b: Split into heads ---
print("\n--- Step 3b: Split into heads ---")

def split_heads(x, n_heads):
    batch, seq, d_model = x.shape
    d_k = d_model // n_heads
    x = x.reshape(batch, seq, n_heads, d_k)
    x = x.transpose(0, 2, 1, 3)
    return x

Q_heads = split_heads(Q, n_heads)  # (1, 20, 8) → (1, 2, 20, 4)
K_heads = split_heads(K, n_heads)
V_heads = split_heads(V, n_heads)

print(f"Before split: Q shape = {Q.shape}")
print(f"After split:  Q shape = {Q_heads.shape}")
print(f"  - Batch: {Q_heads.shape[0]}")
print(f"  - Heads: {Q_heads.shape[1]}")
print(f"  - Sequence: {Q_heads.shape[2]}")
print(f"  - Dim per head: {Q_heads.shape[3]}")

# --- STEP 3c: Compute attention scores ---
print("\n--- Step 3c: Compute attention scores (Q @ K^T) ---")

# (1, 2, 20, 4) @ (1, 2, 4, 20) = (1, 2, 20, 20)
scores = Q_heads @ K_heads.transpose(0, 1, 3, 2)
scores = scores / np.sqrt(d_k)

print(f"Scores shape: {scores.shape}")
print(f"  - This is a {scores.shape[2]}x{scores.shape[3]} matrix for each head")
print(f"  - Entry [i,j] = how much position i attends to position j")

# Show a small portion of scores for Head 1
print(f"\nScores for Head 1, first 5x5 positions (before mask):")
print(scores[0, 0, :5, :5])

# --- STEP 3d: Apply causal mask ---
print("\n--- Step 3d: Apply causal mask ---")

# Create mask: upper triangle = -infinity
mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)

print(f"Mask shape: {mask.shape}")
print(f"Mask (first 5x5):")
print(f"  0 = can attend, -inf = blocked")
mask_display = np.where(mask[:5, :5] < -1e8, "-inf", "0")
for row in mask_display:
    print(f"  {row}")

# Apply mask
scores_masked = scores + mask

print(f"\nScores after mask (Head 1, first 5x5):")
scores_display = scores_masked[0, 0, :5, :5].copy()
scores_display = np.where(scores_display < -1e8, float('nan'), scores_display)
print(np.round(scores_display, 2))

# --- STEP 3e: Softmax ---
print("\n--- Step 3e: Softmax (convert to probabilities) ---")

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

attention_weights = softmax(scores_masked, axis=-1)

print(f"Attention weights shape: {attention_weights.shape}")
print(f"\nHead 1 attention weights (first 5 positions):")
print(f"         Attending to position:")
print(f"         0     1     2     3     4")
for i in range(5):
    char = phrase[i]
    weights = attention_weights[0, 0, i, :5]
    print(f"Pos {i} '{char}': {weights}")

print(f"\nNotice:")
print(f"  - Row 0 ('M'): attends 100% to position 0 (only option)")
print(f"  - Row 1 ('a'): splits between positions 0, 1")
print(f"  - Row 4 ('n'): distributes across positions 0-4")
print(f"  - Upper triangle is 0 (future is masked)")

# --- STEP 3f: Weighted sum of values ---
print("\n--- Step 3f: Weighted sum of values ---")

# (1, 2, 20, 20) @ (1, 2, 20, 4) = (1, 2, 20, 4)
head_outputs = attention_weights @ V_heads

print(f"Head outputs shape: {head_outputs.shape}")
print(f"\nHead 1 output for position 0 ('M'):")
print(f"  {head_outputs[0, 0, 0, :]}")
print(f"\nHead 2 output for position 0 ('M'):")
print(f"  {head_outputs[0, 1, 0, :]}")
print(f"  (Different heads produce different outputs)")

# --- STEP 3g: Merge heads ---
print("\n--- Step 3g: Merge heads back together ---")

def merge_heads(x):
    batch, n_heads, seq, d_k = x.shape
    x = x.transpose(0, 2, 1, 3)
    x = x.reshape(batch, seq, n_heads * d_k)
    return x

concat = merge_heads(head_outputs)  # (1, 2, 20, 4) → (1, 20, 8)

print(f"Before merge: {head_outputs.shape}")
print(f"After merge:  {concat.shape}")
print(f"\nPosition 0 ('M') after merging heads:")
print(f"  {concat[0, 0, :]}")
print(f"  (First 4 dims from Head 1, last 4 dims from Head 2)")

# --- STEP 3h: Output projection ---
print("\n--- Step 3h: Output projection (W_O) ---")

output = concat @ W_O  # (1, 20, 8) @ (8, 8) = (1, 20, 8)

print(f"Final output shape: {output.shape}")
print(f"\nOutput for first 3 positions:")
for i in range(3):
    char = phrase[i]
    print(f"  Position {i} ('{char}'): {output[0, i, :]}")

# =============================================================================
# SUMMARY: THE COMPLETE FLOW
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: COMPLETE FLOW FOR 'Mañana será otro día'")
print("=" * 70)

print("""
Step 1: TOKENIZER
  'M' 'a' 'ñ' 'a' 'n' 'a' ' ' 's' 'e' 'r' 'á' ' ' 'o' 't' 'r' 'o' ' ' 'd' 'í' 'a'
   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
  [0] [3] [14][3] [8] [3] [1] [11][6] [10][5] [1] [9] [12][10][9] [1] [4] [7] [3]

Step 2: EMBEDDING + POSITION
  Each ID → 8-dimensional vector
  Add positional encoding (so model knows order)
  Shape: (1, 20, 8)

Step 3: ATTENTION
  a) Project to Q, K, V using W_Q, W_K, W_V
  b) Split into 2 heads (each 4 dimensions)
  c) Compute scores: Q @ K^T (who attends to whom)
  d) Apply causal mask (hide future positions)
  e) Softmax (convert to probabilities)
  f) Weighted sum of V (blend relevant information)
  g) Merge heads back together
  h) Output projection with W_O

Result: Each position now contains a blend of information
        from all PREVIOUS positions it could attend to.
""")

# =============================================================================
# BATCH EXAMPLE
# =============================================================================
print("=" * 70)
print("BONUS: BATCH EXAMPLE (Multiple phrases at once)")
print("=" * 70)

# Two phrases of same length (padded in practice)
phrase1 = "Hola mundo"
phrase2 = "Adiós amig"  # Same length for simplicity

print(f"\nPhrase 1: '{phrase1}'")
print(f"Phrase 2: '{phrase2}'")

# Build combined vocab
all_chars = sorted(list(set(phrase1 + phrase2)))
char_to_idx_batch = {c: i for i, c in enumerate(all_chars)}

# Encode both
ids1 = [char_to_idx_batch[c] for c in phrase1]
ids2 = [char_to_idx_batch[c] for c in phrase2]

batch_tokens = np.array([ids1, ids2])  # Shape: (2, 10)

print(f"\nBatched token IDs shape: {batch_tokens.shape}")
print(f"  Phrase 1 IDs: {ids1}")
print(f"  Phrase 2 IDs: {ids2}")

print(f"""
With batch_size=2:
  - All operations process BOTH phrases simultaneously
  - Shapes become (2, 10, ...) instead of (1, 10, ...)
  - Same mask applies to both (both respect causal constraint)
  - Much faster than processing one at a time!
""")

# =============================================================================
# ANSWERS TO YOUR QUESTIONS
# =============================================================================
print("=" * 70)
print("ANSWERS TO YOUR EARLIER QUESTIONS")
print("=" * 70)

print("""
Q1: If batch_size=5, seq_len=10, d_model=64, n_heads=2,
    what is attention_weights shape?

A1: (5, 2, 10, 10)
    - 5 examples in batch
    - 2 attention heads
    - 10x10 attention matrix (each position to each position)

Q2: In causal mask for seq_len=5, can position 3 attend to position 4?

A2: NO. Position 3 can only attend to positions 0, 1, 2, 3.
    Position 4 is the FUTURE, so it's masked with -inf.

    Mask for seq_len=5:
    [[  0, -inf, -inf, -inf, -inf],
     [  0,    0, -inf, -inf, -inf],
     [  0,    0,    0, -inf, -inf],
     [  0,    0,    0,    0, -inf],  ← position 3: mask[3,4] = -inf
     [  0,    0,    0,    0,    0]]
""")
