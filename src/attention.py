"""
Attention Mechanism - NumPy Implementation

This is the CORE of the transformer. Everything we discussed lives here.

FORMULA:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V

SHAPES (single-head):
    Input:   (batch, seq_len, d_model)
    Q, K, V: (batch, seq_len, d_k)
    Scores:  (batch, seq_len, seq_len)  <- who attends to whom
    Output:  (batch, seq_len, d_k)

SHAPES (multi-head):
    Input:   (batch, seq_len, d_model)
    Q, K, V: (batch, n_heads, seq_len, d_k)  <- extra head dimension
    Scores:  (batch, n_heads, seq_len, seq_len)
    Output:  (batch, seq_len, d_model)

REMEMBER: This is all just numbers. The "meaning" is our interpretation.
"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Stable softmax implementation.

    Subtracting max prevents numerical overflow.
    exp(x - max) / sum(exp(x - max)) = exp(x) / sum(exp(x))
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal (autoregressive) mask.

    Position i can only attend to positions 0, 1, ..., i
    NOT to positions i+1, i+2, ... (future)

    Returns:
        Mask of shape (seq_len, seq_len)
        Upper triangle is -inf, lower triangle + diagonal is 0
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
    return mask


class SingleHeadAttention:
    """
    Single-head self-attention.
    (Kept for educational comparison with multi-head)
    """

    def __init__(self, d_model: int, d_k: int):
        self.d_model = d_model
        self.d_k = d_k

        # Learnable projection matrices
        self.W_Q = np.random.randn(d_model, d_k) * 0.02
        self.W_K = np.random.randn(d_model, d_k) * 0.02
        self.W_V = np.random.randn(d_model, d_k) * 0.02

    def forward(self, x: np.ndarray, use_mask: bool = True):
        batch_size, seq_len, _ = x.shape

        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        scores = Q @ K.transpose(0, 2, 1)
        scores = scores / np.sqrt(self.d_k)

        if use_mask:
            mask = create_causal_mask(seq_len)
            scores = scores + mask

        attention_weights = softmax(scores, axis=-1)
        output = attention_weights @ V

        return output, attention_weights


class MultiHeadAttention:
    """
    Multi-head self-attention.

    Multiple attention "heads" run in parallel, each learning different
    patterns. Their outputs are concatenated and projected.

    Remember: the model doesn't "know" what patterns it's learning.
    It just finds numeric correlations that reduce prediction error.
    We call them "syntax heads" or "semantic heads" - that's our interpretation.
    """

    def __init__(self, d_model: int, n_heads: int):
        """
        Args:
            d_model: Model dimension (e.g., 64)
            n_heads: Number of attention heads (e.g., 2)

        Constraint: d_model must be divisible by n_heads
        """
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head

        # =====================================================================
        # WEIGHT MATRICES
        # =====================================================================
        # All project from d_model to d_model
        # The "splitting into heads" happens via reshape, not separate matrices

        # Shape: (d_model, d_model) = (64, 64)
        self.W_Q = np.random.randn(d_model, d_model) * 0.02
        self.W_K = np.random.randn(d_model, d_model) * 0.02
        self.W_V = np.random.randn(d_model, d_model) * 0.02

        # Output projection: combines information across heads
        # Shape: (d_model, d_model) = (64, 64)
        self.W_O = np.random.randn(d_model, d_model) * 0.02

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split the last dimension into (n_heads, d_k) and transpose.

        Args:
            x: Shape (batch, seq_len, d_model)

        Returns:
            Shape (batch, n_heads, seq_len, d_k)

        Example with batch=1, seq=4, d_model=64, n_heads=2, d_k=32:
            (1, 4, 64) → (1, 4, 2, 32) → (1, 2, 4, 32)
                              ↑              ↑
                          reshape      transpose
        """
        batch_size, seq_len, _ = x.shape

        # Reshape: (batch, seq, d_model) → (batch, seq, n_heads, d_k)
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)

        # Transpose: (batch, seq, n_heads, d_k) → (batch, n_heads, seq, d_k)
        # This puts heads before sequence, enabling parallel attention
        x = x.transpose(0, 2, 1, 3)

        return x

    def merge_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Reverse of split_heads: merge heads back into d_model dimension.

        Args:
            x: Shape (batch, n_heads, seq_len, d_k)

        Returns:
            Shape (batch, seq_len, d_model)

        Example:
            (1, 2, 4, 32) → (1, 4, 2, 32) → (1, 4, 64)
                                ↑              ↑
                          transpose      reshape
        """
        batch_size, _, seq_len, _ = x.shape

        # Transpose: (batch, n_heads, seq, d_k) → (batch, seq, n_heads, d_k)
        x = x.transpose(0, 2, 1, 3)

        # Reshape: (batch, seq, n_heads, d_k) → (batch, seq, d_model)
        x = x.reshape(batch_size, seq_len, self.d_model)

        return x

    def forward(self, x: np.ndarray, use_mask: bool = True):
        """
        Forward pass of multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            use_mask: Whether to apply causal mask

        Returns:
            output: Shape (batch, seq_len, d_model)
            attention_weights: Shape (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # =====================================================================
        # STEP 1: Project to Q, K, V (full d_model dimension)
        # =====================================================================
        # Shape: (batch, seq, d_model) @ (d_model, d_model) = (batch, seq, d_model)

        Q = x @ self.W_Q  # (1, 4, 64) @ (64, 64) = (1, 4, 64)
        K = x @ self.W_K
        V = x @ self.W_V

        # =====================================================================
        # STEP 2: Split into multiple heads
        # =====================================================================
        # Each head gets a slice of the dimension
        # Shape: (batch, seq, d_model) → (batch, n_heads, seq, d_k)

        Q = self.split_heads(Q)  # (1, 4, 64) → (1, 2, 4, 32)
        K = self.split_heads(K)  # (1, 4, 64) → (1, 2, 4, 32)
        V = self.split_heads(V)  # (1, 4, 64) → (1, 2, 4, 32)

        # =====================================================================
        # STEP 3: Compute attention scores (all heads in parallel)
        # =====================================================================
        # Q @ K^T for each head simultaneously
        # Shape: (batch, n_heads, seq, d_k) @ (batch, n_heads, d_k, seq)
        #      = (batch, n_heads, seq, seq)

        # K.transpose: swap last two dimensions (d_k and seq)
        scores = Q @ K.transpose(0, 1, 3, 2)  # (1, 2, 4, 32) @ (1, 2, 32, 4) = (1, 2, 4, 4)

        # Scale by sqrt(d_k) - note we use d_k (32), not d_model (64)
        scores = scores / np.sqrt(self.d_k)

        # =====================================================================
        # STEP 4: Apply causal mask
        # =====================================================================
        if use_mask:
            # Mask shape: (seq, seq) - broadcasts to (batch, n_heads, seq, seq)
            mask = create_causal_mask(seq_len)
            scores = scores + mask

        # =====================================================================
        # STEP 5: Softmax to get attention weights
        # =====================================================================
        # Each head has its own attention pattern
        # Shape: (batch, n_heads, seq, seq)

        attention_weights = softmax(scores, axis=-1)

        # =====================================================================
        # STEP 6: Weighted sum of values (per head)
        # =====================================================================
        # Shape: (batch, n_heads, seq, seq) @ (batch, n_heads, seq, d_k)
        #      = (batch, n_heads, seq, d_k)

        head_outputs = attention_weights @ V  # (1, 2, 4, 4) @ (1, 2, 4, 32) = (1, 2, 4, 32)

        # =====================================================================
        # STEP 7: Merge heads back together
        # =====================================================================
        # Shape: (batch, n_heads, seq, d_k) → (batch, seq, d_model)

        concat = self.merge_heads(head_outputs)  # (1, 2, 4, 32) → (1, 4, 64)

        # =====================================================================
        # STEP 8: Final output projection
        # =====================================================================
        # This lets the model mix information across heads
        # Shape: (batch, seq, d_model) @ (d_model, d_model) = (batch, seq, d_model)

        output = concat @ self.W_O  # (1, 4, 64) @ (64, 64) = (1, 4, 64)

        return output, attention_weights


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("MULTI-HEAD ATTENTION DEMO")
    print("=" * 60)

    # Config matching CLAUDE.md
    batch_size = 1
    seq_len = 4
    d_model = 64
    n_heads = 2

    print(f"\nConfig:")
    print(f"  d_model = {d_model}")
    print(f"  n_heads = {n_heads}")
    print(f"  d_k = {d_model // n_heads} (per head)")

    # Create multi-head attention layer
    mha = MultiHeadAttention(d_model, n_heads)

    # Random input (simulating embeddings)
    x = np.random.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output, weights = mha.forward(x, use_mask=True)

    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

    # Show attention patterns for each head
    print("\n--- Attention Patterns Per Head ---")
    print("(Each head learns different patterns during training)")
    print("(Right now they're similar because weights are random)")

    for head_idx in range(n_heads):
        print(f"\nHead {head_idx + 1} attention weights:")
        print("  Rows = query position, Cols = key position")
        print(np.round(weights[0, head_idx], 2))

    # Compare single-head vs multi-head
    print("\n--- Single-Head vs Multi-Head ---")
    sha = SingleHeadAttention(d_model, d_model)
    sha_output, sha_weights = sha.forward(x, use_mask=True)

    print(f"Single-head output shape: {sha_output.shape}")
    print(f"Multi-head output shape:  {output.shape}")
    print("\nBoth produce same output shape, but multi-head has 2 parallel attention patterns.")
