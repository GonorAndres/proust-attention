"""
Full Transformer Model - NumPy Implementation

=============================================================================
CLASS NOTES: What does a transformer block actually DO?
=============================================================================

You already understand attention: it lets each position gather information
from other positions. But attention alone is NOT enough. Here's why:

    Attention = "WHAT should I look at?"
    FeedForward = "Now that I'm looking at it, WHAT do I think about it?"

Think of Proust writing a sentence. Attention is like his memory scanning
across all his experiences. FeedForward is the actual *thinking* — the
processing of those memories into something new.

The full transformer block is:

    1. Multi-Head Attention  (gather information)
    2. Add & Normalize       (stabilize)
    3. FeedForward           (process information)
    4. Add & Normalize       (stabilize again)

And we stack 2 of these blocks. Each block refines the representation.

=============================================================================
SHAPE FLOW (our config: batch=1, seq=20, d_model=64, vocab_size=~50)
=============================================================================

    Input token IDs:         (batch, seq)
    After embedding:         (batch, seq, 64)
    After transformer block 1: (batch, seq, 64)   <- same shape!
    After transformer block 2: (batch, seq, 64)   <- same shape!
    After final projection:  (batch, seq, vocab_size)
    After softmax:           (batch, seq, vocab_size)  <- probabilities

KEY INSIGHT: Every transformer block has the SAME input and output shape.
That's what makes them stackable. Like LEGO blocks — same connectors,
different internal processing.

=============================================================================
"""

import numpy as np
from src.attention import MultiHeadAttention, softmax


# =============================================================================
# CONFIGURATION
# =============================================================================
# All hyperparameters in one place. No magic numbers anywhere below.

CONFIG = {
    'd_model': 64,          # Embedding dimension
    'n_heads': 2,           # Number of attention heads
    'n_layers': 2,          # Number of transformer blocks
    'd_ff': 256,            # FeedForward hidden dimension (4 * d_model)
    'max_seq_len': 256,     # Maximum sequence length
    'init_scale': 0.02,     # Weight initialization scale
    'mask_value': -1e9,     # Value for masked positions
}


# =============================================================================
# LAYER NORMALIZATION
# =============================================================================
#
# CLASS NOTES: Why do we need normalization?
#
# Imagine you're adding numbers together across many layers. Each layer
# adds a little bit. After 10 layers, the numbers might be HUGE or tiny.
# This makes learning unstable — gradients explode or vanish.
#
# LayerNorm says: "After each sub-layer, force the numbers to have
# mean=0 and variance=1 across the d_model dimension. Then let the
# model learn a scale (gamma) and shift (beta) if it wants."
#
# FORMULA:
#     LayerNorm(x) = gamma * (x - mean) / sqrt(variance + epsilon) + beta
#
# WHERE:
#     - mean and variance are computed PER POSITION across d_model
#     - gamma (scale) and beta (shift) are LEARNED parameters
#     - epsilon prevents division by zero
#
# SHAPE:
#     Input:  (batch, seq, d_model)
#     Output: (batch, seq, d_model)   <- same shape, just normalized
#
# VISUAL EXAMPLE (one position, d_model=4):
#     Before: [3.2, -1.5, 0.8, 2.1]
#     mean = 1.15, std = 1.73
#     After:  [1.18, -1.53, -0.20, 0.55]  (roughly mean=0, std=1)
#     Then:   gamma * normalized + beta    (learned adjustment)
#
# WHY across d_model and not across batch?
#     Because each position's 64 numbers represent ONE "thought".
#     We normalize each thought independently. BatchNorm (used in CNNs)
#     normalizes across examples — that doesn't make sense for sequences
#     because sequence statistics vary wildly sentence to sentence.
#
# =============================================================================

class LayerNorm:
    """
    Layer Normalization.

    Normalizes across the last dimension (d_model).
    Each position is normalized independently.
    """

    def __init__(self, d_model: int, epsilon: float = 1e-5):
        """
        Args:
            d_model: Dimension to normalize across
            epsilon: Small constant for numerical stability
        """
        self.d_model = d_model
        self.epsilon = epsilon

        # Learnable parameters (initialized to neutral values)
        # gamma=1 means "don't scale" initially
        # beta=0 means "don't shift" initially
        # Shape: (d_model,) = (64,)
        self.gamma = np.ones(d_model)     # Scale parameter  (learned)
        self.beta = np.zeros(d_model)     # Shift parameter  (learned)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize the input.

        Args:
            x: Shape (batch, seq, d_model)

        Returns:
            Normalized tensor of shape (batch, seq, d_model)

        Step by step for one position vector [3.2, -1.5, 0.8, 2.1]:
            1. mean = (3.2 + -1.5 + 0.8 + 2.1) / 4 = 1.15
            2. var  = mean((x - mean)^2) = 2.99
            3. norm = (x - 1.15) / sqrt(2.99 + 1e-5) = [1.18, -1.53, -0.20, 0.55]
            4. output = gamma * norm + beta
        """
        # Compute mean across d_model dimension (last axis)
        # Shape: (batch, seq, 1) - keepdims for broadcasting
        mean = np.mean(x, axis=-1, keepdims=True)

        # Compute variance across d_model dimension
        # Shape: (batch, seq, 1)
        variance = np.var(x, axis=-1, keepdims=True)

        # Normalize: zero mean, unit variance
        # Shape: (batch, seq, d_model)
        x_normalized = (x - mean) / np.sqrt(variance + self.epsilon)

        # Apply learned scale and shift
        # gamma and beta broadcast: (d_model,) -> (batch, seq, d_model)
        # Shape: (batch, seq, d_model)
        output = self.gamma * x_normalized + self.beta

        return output


# =============================================================================
# FEEDFORWARD NETWORK
# =============================================================================
#
# CLASS NOTES: What is the FeedForward layer doing?
#
# After attention gathers information from other positions, each position
# needs to PROCESS that information. The FeedForward network is a simple
# 2-layer MLP (multi-layer perceptron) that runs on each position
# INDEPENDENTLY.
#
# Think of it like this:
#     - Attention is a MEETING: everyone shares information
#     - FeedForward is THINKING ALONE: each position processes what it heard
#
# ARCHITECTURE:
#     Input (d_model=64)
#         |
#     Linear_1: (64 -> 256)   <- expand to higher dimension
#         |
#     ReLU activation          <- introduce non-linearity
#         |
#     Linear_2: (256 -> 64)   <- compress back
#         |
#     Output (d_model=64)
#
# WHY expand then compress?
#     The expansion to 256 dimensions (4x) creates a "wider" space where
#     the model can represent more complex features. Then it compresses
#     back to 64 to keep the shape compatible with the next layer.
#
#     It's like Proust expanding a fleeting sensation into a 3-page
#     meditation, then distilling it into a single perfect phrase.
#
# WHY ReLU?
#     Without a non-linearity, stacking two linear layers is equivalent to
#     one linear layer (linear algebra: A @ B = C, which is still linear).
#     ReLU(x) = max(0, x) is the simplest way to break linearity.
#     It says: "keep positive signals, silence negative ones."
#
# SHAPE FLOW:
#     Input:          (batch, seq, d_model)    = (1, 20, 64)
#     After Linear_1: (batch, seq, d_ff)       = (1, 20, 256)
#     After ReLU:     (batch, seq, d_ff)       = (1, 20, 256)
#     After Linear_2: (batch, seq, d_model)    = (1, 20, 64)
#
# =============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation: max(0, x)

    Keeps positive values, sets negative to zero.

    Why this works:
        - Creates sparsity (many zeros = efficient representation)
        - Gradient is 1 for positive, 0 for negative (simple backprop)
        - Breaks linearity so the network can learn non-linear patterns
    """
    return np.maximum(0, x)


class FeedForward:
    """
    Position-wise FeedForward Network.

    Each position is processed independently through the same 2-layer MLP.
    "Position-wise" means: position 0 goes through the network,
    position 1 goes through the SAME network (same weights), etc.
    """

    def __init__(self, d_model: int, d_ff: int):
        """
        Args:
            d_model: Input and output dimension (64)
            d_ff: Hidden dimension (256 = 4 * d_model)

        The 4x expansion is a convention from the original paper.
        It gives the network enough capacity to learn complex features
        without being too expensive.
        """
        self.d_model = d_model
        self.d_ff = d_ff

        # First linear layer: expand
        # Shape: (d_model, d_ff) = (64, 256)
        self.W_1 = np.random.randn(d_model, d_ff) * CONFIG['init_scale']
        self.b_1 = np.zeros(d_ff)  # Bias: (256,)

        # Second linear layer: compress
        # Shape: (d_ff, d_model) = (256, 64)
        self.W_2 = np.random.randn(d_ff, d_model) * CONFIG['init_scale']
        self.b_2 = np.zeros(d_model)  # Bias: (64,)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Two-layer MLP with ReLU activation.

        Args:
            x: Shape (batch, seq, d_model)

        Returns:
            Shape (batch, seq, d_model)

        Math:
            output = ReLU(x @ W_1 + b_1) @ W_2 + b_2
        """
        # Layer 1: expand to d_ff dimensions
        # (batch, seq, d_model) @ (d_model, d_ff) = (batch, seq, d_ff)
        hidden = x @ self.W_1 + self.b_1  # (1, 20, 64) @ (64, 256) = (1, 20, 256)

        # Activation: introduce non-linearity
        # Shape unchanged: (batch, seq, d_ff)
        hidden = relu(hidden)  # (1, 20, 256) — negative values become 0

        # Layer 2: compress back to d_model
        # (batch, seq, d_ff) @ (d_ff, d_model) = (batch, seq, d_model)
        output = hidden @ self.W_2 + self.b_2  # (1, 20, 256) @ (256, 64) = (1, 20, 64)

        return output


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================
#
# CLASS NOTES: The Residual Connection (the "Add" in "Add & Norm")
#
# This is one of the most important tricks in deep learning, and it's
# beautifully simple:
#
#     output = x + sublayer(x)
#
# Instead of replacing x with the sublayer output, we ADD it.
# The sublayer only needs to learn the DIFFERENCE (the "residual").
#
# Why this matters:
#     Without residuals, a 12-layer network must transform the input
#     completely through 12 transformations. If any layer messes up,
#     everything is lost.
#
#     With residuals, the original information ALWAYS flows through.
#     Each layer says "here's a small refinement to add on top."
#     Even if a layer learns nothing useful, the signal passes through.
#
# VISUAL:
#     x ─────────────────── + ──── output
#         |                 ↑
#         └── sublayer(x) ──┘
#
#     The straight line on top is the "skip connection" or "residual path".
#     Information can skip the sublayer entirely if needed.
#
# In our transformer block:
#
#     x ────────────── + ── LayerNorm ── y
#         |            ↑
#         └─ Attention ┘
#
#     y ────────────── + ── LayerNorm ── output
#         |            ↑
#         └─ FeedFwd ──┘
#
# NOTE: We use "Post-Norm" style (normalize AFTER the residual add).
# The original paper uses this. Some newer models use "Pre-Norm"
# (normalize BEFORE the sublayer), which trains more stably but
# we follow the original for educational clarity.
#
# =============================================================================

class TransformerBlock:
    """
    One transformer block = Attention + FeedForward, each with
    residual connections and layer normalization.

    This is the fundamental repeating unit. Our model stacks 2 of these.

    SHAPE INVARIANT:
        Input:  (batch, seq, d_model)
        Output: (batch, seq, d_model)
        The shape NEVER changes through a block. That's what makes
        blocks stackable.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        """
        Args:
            d_model: Model dimension (64)
            n_heads: Number of attention heads (2)
            d_ff: FeedForward hidden dimension (256)
        """
        # Sub-layer 1: Multi-Head Attention
        self.attention = MultiHeadAttention(d_model, n_heads)

        # Sub-layer 2: FeedForward
        self.feed_forward = FeedForward(d_model, d_ff)

        # Layer norms (one after each sub-layer)
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)

    def forward(self, x: np.ndarray, use_mask: bool = True):
        """
        Forward pass through one transformer block.

        Args:
            x: Shape (batch, seq, d_model)
            use_mask: Whether to apply causal mask in attention

        Returns:
            output: Shape (batch, seq, d_model)
            attention_weights: Shape (batch, n_heads, seq, seq)
        """
        # =================================================================
        # SUB-LAYER 1: Attention + Residual + Norm
        # =================================================================
        #
        # x shape: (batch, seq, d_model)
        #
        # Step A: Attention
        attention_output, attention_weights = self.attention.forward(x, use_mask)
        # attention_output shape: (batch, seq, d_model)

        # Step B: Residual connection — ADD the input back
        # This is the "skip connection": original + transformation
        # Shape: (batch, seq, d_model)
        x = x + attention_output

        # Step C: Normalize
        # Shape: (batch, seq, d_model)
        x = self.norm_1.forward(x)

        # =================================================================
        # SUB-LAYER 2: FeedForward + Residual + Norm
        # =================================================================
        #
        # Step A: FeedForward
        ff_output = self.feed_forward.forward(x)
        # ff_output shape: (batch, seq, d_model)

        # Step B: Residual connection
        # Shape: (batch, seq, d_model)
        x = x + ff_output

        # Step C: Normalize
        # Shape: (batch, seq, d_model)
        x = self.norm_2.forward(x)

        return x, attention_weights


# =============================================================================
# FULL TRANSFORMER MODEL
# =============================================================================
#
# CLASS NOTES: Putting it all together
#
# The full model is surprisingly simple once you have the pieces:
#
#     1. Embedding layer: tokens -> vectors
#     2. N transformer blocks: refine the vectors
#     3. Linear projection: vectors -> vocabulary probabilities
#
# The final linear layer is the "unembedding" — it maps from the
# internal d_model space back to vocab_size, producing a score for
# each possible next character.
#
# For Proust text generation:
#     Input:  "Mucho tiempo he estado acostándome tempra"
#     Model predicts for EACH position what comes next.
#     At the LAST position, it outputs probabilities:
#         'n' -> 0.65, 'o' -> 0.12, 'a' -> 0.08, ...
#     We sample from these probabilities to get the next character.
#
# =============================================================================

class Transformer:
    """
    Full transformer model for character-level text generation.

    Architecture:
        Embedding -> [TransformerBlock x n_layers] -> Linear -> Softmax

    Config (from CLAUDE.md):
        ~300k parameters, trainable on Colab free tier
        d_model=64, n_heads=2, n_layers=2, context=256 chars
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_seq_len: int):
        """
        Args:
            vocab_size: Number of unique characters (~50 for Spanish Proust)
            d_model: Embedding/model dimension (64)
            n_heads: Attention heads per block (2)
            n_layers: Number of transformer blocks (2)
            d_ff: FeedForward hidden dimension (256)
            max_seq_len: Maximum context window (256)
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        # --- Embedding layer ---
        # Converts token IDs to dense vectors with position info
        # We import it inline to avoid circular imports
        from src.embedding import Embedding
        self.embedding = Embedding(vocab_size, d_model, max_seq_len)

        # --- Transformer blocks ---
        # Each block: attention -> add&norm -> feedforward -> add&norm
        # We stack n_layers of these (2 in our case)
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]

        # --- Final layer norm ---
        # One more normalization before the output projection
        self.final_norm = LayerNorm(d_model)

        # --- Output projection (unembedding) ---
        # Maps from d_model back to vocab_size
        # Shape: (d_model, vocab_size) = (64, ~50)
        self.output_projection = (
            np.random.randn(d_model, vocab_size) * CONFIG['init_scale']
        )

    def forward(self, token_ids: np.ndarray) -> dict:
        """
        Full forward pass: tokens in, probabilities out.

        Args:
            token_ids: Shape (batch, seq) - integer token IDs

        Returns:
            Dictionary with:
                'logits': Raw scores, shape (batch, seq, vocab_size)
                'probabilities': After softmax, shape (batch, seq, vocab_size)
                'attention_maps': List of attention weights per layer
                    Each: (batch, n_heads, seq, seq)

        Shape flow:
            (batch, seq)
                |  embedding
            (batch, seq, 64)
                |  transformer block 1
            (batch, seq, 64)
                |  transformer block 2
            (batch, seq, 64)
                |  final norm
            (batch, seq, 64)
                |  output projection
            (batch, seq, vocab_size)
                |  softmax
            (batch, seq, vocab_size)  <- probability for each next character
        """
        # =================================================================
        # STEP 1: Embedding
        # =================================================================
        # token_ids: (batch, seq) -> x: (batch, seq, d_model)
        x = self.embedding.forward(token_ids)

        # =================================================================
        # STEP 2: Pass through all transformer blocks
        # =================================================================
        # Each block: (batch, seq, d_model) -> (batch, seq, d_model)
        # The shape stays the same — each block REFINES the representation
        attention_maps = []

        for layer_idx, block in enumerate(self.blocks):
            x, attention_weights = block.forward(x, use_mask=True)
            attention_maps.append(attention_weights)
            # x shape is still (batch, seq, d_model) after every block

        # =================================================================
        # STEP 3: Final normalization
        # =================================================================
        # Shape: (batch, seq, d_model)
        x = self.final_norm.forward(x)

        # =================================================================
        # STEP 4: Project to vocabulary size
        # =================================================================
        # This is the "unembedding" — from internal space to character scores
        # (batch, seq, d_model) @ (d_model, vocab_size) = (batch, seq, vocab_size)
        logits = x @ self.output_projection

        # =================================================================
        # STEP 5: Convert to probabilities
        # =================================================================
        # Softmax over the last dimension (vocab_size)
        # Each position now has a probability distribution over all characters
        probabilities = softmax(logits, axis=-1)

        return {
            'logits': logits,
            'probabilities': probabilities,
            'attention_maps': attention_maps,
        }

    def count_parameters(self) -> dict:
        """
        Count all parameters in the model.
        Educational: shows where the ~300k parameters live.
        """
        param_counts = {}

        # Embedding
        param_counts['token_embedding'] = self.embedding.token_embedding.size
        # Note: positional encoding is NOT learned, so not counted

        # Transformer blocks
        for i, block in enumerate(self.blocks):
            prefix = f'block_{i}'

            # Attention weights: W_Q, W_K, W_V, W_O each (d_model, d_model)
            param_counts[f'{prefix}_attention_W_Q'] = block.attention.W_Q.size
            param_counts[f'{prefix}_attention_W_K'] = block.attention.W_K.size
            param_counts[f'{prefix}_attention_W_V'] = block.attention.W_V.size
            param_counts[f'{prefix}_attention_W_O'] = block.attention.W_O.size

            # FeedForward: W_1, b_1, W_2, b_2
            param_counts[f'{prefix}_ff_W_1'] = block.feed_forward.W_1.size
            param_counts[f'{prefix}_ff_b_1'] = block.feed_forward.b_1.size
            param_counts[f'{prefix}_ff_W_2'] = block.feed_forward.W_2.size
            param_counts[f'{prefix}_ff_b_2'] = block.feed_forward.b_2.size

            # LayerNorms: gamma and beta for each
            param_counts[f'{prefix}_norm1_gamma'] = block.norm_1.gamma.size
            param_counts[f'{prefix}_norm1_beta'] = block.norm_1.beta.size
            param_counts[f'{prefix}_norm2_gamma'] = block.norm_2.gamma.size
            param_counts[f'{prefix}_norm2_beta'] = block.norm_2.beta.size

        # Final norm
        param_counts['final_norm_gamma'] = self.final_norm.gamma.size
        param_counts['final_norm_beta'] = self.final_norm.beta.size

        # Output projection
        param_counts['output_projection'] = self.output_projection.size

        return param_counts


# =============================================================================
# DEMONSTRATION: Full Forward Pass
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 70)
    print("FULL TRANSFORMER MODEL - FORWARD PASS DEMO")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Setup: Build vocabulary from a Proust sample
    # -------------------------------------------------------------------------
    sample_text = "Mucho tiempo he estado acostándome temprano."

    from src.tokenizer import CharTokenizer

    tokenizer = CharTokenizer()
    tokenizer.build_vocab(sample_text)

    print(f"\nVocabulary: {tokenizer.vocab_size} unique characters")
    print(f"Characters: {sorted(tokenizer.char_to_idx.keys())}")

    # -------------------------------------------------------------------------
    # Create the model
    # -------------------------------------------------------------------------
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        d_ff=CONFIG['d_ff'],
        max_seq_len=CONFIG['max_seq_len'],
    )

    print(f"\nModel created with config:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    # -------------------------------------------------------------------------
    # Count parameters
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PARAMETER COUNT — Where do the ~300k parameters live?")
    print(f"{'='*70}")

    param_counts = model.count_parameters()
    total = 0
    for name, count in param_counts.items():
        print(f"  {name:40s} {count:>8,}")
        total += count
    print(f"  {'─'*48}")
    print(f"  {'TOTAL':40s} {total:>8,}")

    # -------------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("FORWARD PASS — Tracing the shapes")
    print(f"{'='*70}")

    # Encode input text
    input_text = "Mucho tiempo"
    token_ids = tokenizer.encode(input_text)
    token_ids_batched = token_ids.reshape(1, -1)  # (1, 12)

    print(f"\nInput: '{input_text}'")
    print(f"Token IDs: {token_ids}")
    print(f"Batched shape: {token_ids_batched.shape}")

    # Run forward pass
    results = model.forward(token_ids_batched)

    print(f"\n--- Shape at each stage ---")
    print(f"Input token IDs:      {token_ids_batched.shape}")
    print(f"After full model:")
    print(f"  Logits:             {results['logits'].shape}")
    print(f"  Probabilities:      {results['probabilities'].shape}")
    print(f"  Attention maps:     {len(results['attention_maps'])} layers")
    for i, attn in enumerate(results['attention_maps']):
        print(f"    Layer {i}: {attn.shape}")

    # -------------------------------------------------------------------------
    # Show predictions at each position
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PREDICTIONS — What does the model predict at each position?")
    print(f"{'='*70}")
    print("(Random weights = random predictions. Training would fix this.)\n")

    probabilities = results['probabilities'][0]  # Remove batch dim: (seq, vocab)

    for pos in range(len(input_text)):
        current_char = input_text[pos]
        actual_next = input_text[pos + 1] if pos + 1 < len(input_text) else '?'

        # Get top 3 predictions for next character
        probs_at_pos = probabilities[pos]
        top_3_indices = np.argsort(probs_at_pos)[-3:][::-1]

        print(f"Position {pos:2d} '{current_char}' -> next should be '{actual_next}'")
        for idx in top_3_indices:
            char = tokenizer.idx_to_char[idx]
            prob = probs_at_pos[idx]
            display_char = repr(char) if char == ' ' else char
            print(f"    predicts '{display_char}': {prob:.2%}")

    # -------------------------------------------------------------------------
    # Show attention pattern
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("ATTENTION PATTERNS — Who attends to whom?")
    print(f"{'='*70}")
    print("(Layer 0, Head 0 — first 8 positions)\n")

    attn = results['attention_maps'][0][0, 0, :8, :8]  # (8, 8)
    chars = list(input_text[:8])

    # Header
    print(f"{'':>6}", end="")
    for c in chars:
        print(f" {c:>5}", end="")
    print()

    # Rows
    for i, row in enumerate(attn):
        print(f"  {chars[i]:>3} ", end="")
        for val in row:
            if val < 0.01:
                print(f"    .", end="")
            else:
                print(f" {val:4.2f}", end="")
        print()

    print(f"\nNotice the triangular pattern: each position can only")
    print(f"attend to itself and previous positions (causal mask).")

    print(f"\n{'='*70}")
    print("COMPLETE. The forward pass works end-to-end.")
    print(f"{'='*70}")
    print(f"""
Next steps to make this a Proust machine:
  1. Get the full corpus (all 7 volumes)
  2. Port this to PyTorch (same structure, but with autograd)
  3. Train with AdamW optimizer
  4. Generate text by sampling from the output probabilities
""")
