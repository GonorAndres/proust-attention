"""
Embedding Layer with Positional Encoding

PURPOSE:
    1. Convert token IDs to dense vectors (learned)
    2. Add position information so model knows word order

SHAPES:
    Input:  (batch, seq_len) - integer token IDs
    Output: (batch, seq_len, d_model) - dense vectors with position info
"""

import numpy as np


class Embedding:
    """
    Token embedding + positional encoding.

    The embedding matrix is LEARNED during training.
    Positional encodings are FIXED (computed once using sin/cos).
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int):
        """
        Args:
            vocab_size: Number of unique tokens (characters)
            d_model: Dimension of embedding vectors
            max_seq_len: Maximum sequence length (for positional encoding)
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embeddings - LEARNED
        # Shape: (vocab_size, d_model)
        # Initialize with small random values
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02

        # Positional encoding - FIXED (not learned)
        # Shape: (max_seq_len, d_model)
        self.pos_encoding = self._create_positional_encoding()

    def _create_positional_encoding(self) -> np.ndarray:
        """
        Create sinusoidal positional encodings.

        Why sin/cos?
        - Each position gets a unique pattern
        - Patterns are smooth (nearby positions are similar)
        - Model can learn to attend to relative positions

        Formula from "Attention Is All You Need":
            PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        # Shape: (max_seq_len, d_model)
        pos_encoding = np.zeros((self.max_seq_len, self.d_model))

        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        # Shape: (max_seq_len, 1)
        position = np.arange(self.max_seq_len)[:, np.newaxis]

        # Dimension indices for the formula
        # Shape: (d_model/2,)
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pos_encoding[:, 0::2] = np.sin(position * div_term)  # Even dimensions
        pos_encoding[:, 1::2] = np.cos(position * div_term)  # Odd dimensions

        return pos_encoding

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Convert token IDs to embeddings with positional encoding.

        Args:
            token_ids: Integer array of shape (batch, seq_len)

        Returns:
            Embeddings of shape (batch, seq_len, d_model)

        Process:
            1. Look up token embeddings (learned)
            2. Add positional encodings (fixed)
        """
        batch_size, seq_len = token_ids.shape

        # Step 1: Token embedding lookup
        # For each token ID, grab that row from the embedding matrix
        # Shape: (batch, seq_len, d_model)
        token_emb = self.token_embedding[token_ids]

        # Scale embeddings by sqrt(d_model) as in original paper
        # This prevents positional encodings (magnitude ~1) from
        # overwhelming the learned embeddings (magnitude ~0.02)
        token_emb = token_emb * np.sqrt(self.d_model)

        # Step 2: Add positional encoding
        # Slice to match sequence length, broadcast across batch
        # Shape: (seq_len, d_model) -> broadcasts to (batch, seq_len, d_model)
        pos_emb = self.pos_encoding[:seq_len, :]

        # Combine: each token now knows its content AND position
        # Shape: (batch, seq_len, d_model)
        output = token_emb + pos_emb

        return output


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EMBEDDING LAYER DEMO")
    print("=" * 60)

    # Config matching CLAUDE.md
    vocab_size = 100
    d_model = 64
    max_seq_len = 256

    # Create embedding layer
    emb = Embedding(vocab_size, d_model, max_seq_len)

    print(f"\nEmbedding matrix shape: {emb.token_embedding.shape}")
    print(f"Positional encoding shape: {emb.pos_encoding.shape}")

    # Simulate input: batch of 2 sequences, each 10 tokens
    # (random token IDs for demo)
    batch_size, seq_len = 2, 10
    token_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))

    print(f"\nInput token_ids shape: {token_ids.shape}")
    print(f"Sample token IDs (first sequence): {token_ids[0]}")

    # Forward pass
    output = emb.forward(token_ids)

    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: (batch={batch_size}, seq={seq_len}, d_model={d_model})")

    # Show that same token at different positions gives different vectors
    print("\n--- Same Token, Different Positions ---")
    token_id = 5
    test_input = np.array([[token_id, token_id, token_id]])  # Same token 3 times
    test_output = emb.forward(test_input)

    print(f"Token {token_id} at position 0: {test_output[0, 0, :4]}...")
    print(f"Token {token_id} at position 1: {test_output[0, 1, :4]}...")
    print(f"Token {token_id} at position 2: {test_output[0, 2, :4]}...")
    print("(Different because positional encoding is added!)")
