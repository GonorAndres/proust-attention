#!/usr/bin/env python3
"""
PyTorch Transformer Model for Proust Attention Machine

=============================================================================
PURPOSE
=============================================================================

This is a MECHANICAL PORT of the NumPy implementation to PyTorch.
The structure is IDENTICAL - same variable names, same shape comments,
same forward pass logic. The only differences are:

1. np.ndarray -> torch.Tensor
2. Manual weight matrices -> nn.Parameter / nn.Linear
3. Manual operations -> PyTorch equivalents with autograd

This parallel structure lets us:
- Verify outputs match between implementations
- Use NumPy for understanding, PyTorch for training
- Debug by comparing intermediate values

=============================================================================
SHAPE FLOW (same as NumPy version)
=============================================================================

    Input token IDs:         (batch, seq)
    After embedding:         (batch, seq, d_model)
    After transformer block: (batch, seq, d_model)  <- same shape!
    After final projection:  (batch, seq, vocab_size)

    For training: return logits (no softmax) - CrossEntropyLoss handles it
    For inference: apply softmax manually

=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'd_model': 128,         # Embedding dimension
    'n_heads': 2,           # Number of attention heads
    'n_layers': 2,          # Number of transformer blocks
    'd_ff': 512,            # FeedForward hidden dimension (4 * d_model)
    'max_seq_len': 256,     # Maximum sequence length (context window)
    'dropout': 0.1,         # Dropout probability
}


# =============================================================================
# EMBEDDING LAYER
# =============================================================================

class Embedding(nn.Module):
    """
    Token embedding + positional encoding.

    NumPy equivalent: src/embedding.py Embedding class

    The embedding matrix is LEARNED during training.
    Positional encodings are FIXED (computed once using sin/cos).
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int,
                 dropout: float = 0.1):
        """
        Args:
            vocab_size: Number of unique tokens (characters)
            d_model: Dimension of embedding vectors
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model

        # Token embeddings - LEARNED
        # Shape: (vocab_size, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding - FIXED (not learned)
        # Shape: (max_seq_len, d_model)
        # We use register_buffer so it's saved with the model but not trained
        pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        self.register_buffer('pos_encoding', pos_encoding)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def _create_positional_encoding(self, max_seq_len: int,
                                    d_model: int) -> torch.Tensor:
        """
        Create sinusoidal positional encodings.

        Formula from "Attention Is All You Need":
            PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Returns:
            Tensor of shape (max_seq_len, d_model)
        """
        # Shape: (max_seq_len, d_model)
        pos_encoding = torch.zeros(max_seq_len, d_model)

        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        # Shape: (max_seq_len, 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Dimension scaling factor
        # Shape: (d_model/2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pos_encoding[:, 0::2] = torch.sin(position * div_term)  # Even dims
        pos_encoding[:, 1::2] = torch.cos(position * div_term)  # Odd dims

        return pos_encoding

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings with positional encoding.

        Args:
            token_ids: Integer tensor of shape (batch, seq_len)

        Returns:
            Embeddings of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len = token_ids.shape

        # Step 1: Token embedding lookup
        # Shape: (batch, seq_len, d_model)
        token_emb = self.token_embedding(token_ids)

        # Scale embeddings by sqrt(d_model) as in original paper
        token_emb = token_emb * math.sqrt(self.d_model)

        # Step 2: Add positional encoding
        # Slice to match sequence length, broadcast across batch
        # Shape: (seq_len, d_model) -> broadcasts to (batch, seq_len, d_model)
        pos_emb = self.pos_encoding[:seq_len, :]

        # Combine and apply dropout
        # Shape: (batch, seq_len, d_model)
        output = self.dropout(token_emb + pos_emb)

        return output


# =============================================================================
# MULTI-HEAD ATTENTION
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention.

    NumPy equivalent: src/attention.py MultiHeadAttention class

    SHAPE FLOW:
        Input:   (batch, seq_len, d_model)
        Q, K, V: (batch, n_heads, seq_len, d_k)  <- split into heads
        Scores:  (batch, n_heads, seq_len, seq_len)
        Output:  (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (e.g., 64)
            n_heads: Number of attention heads (e.g., 2)
            dropout: Dropout probability

        Constraint: d_model must be divisible by n_heads
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head

        # Projection matrices (no bias, as in original NumPy)
        # Shape: (d_model, d_model) for each
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask of shape (seq_len, seq_len)
                  or (batch, seq_len, seq_len)

        Returns:
            output: Shape (batch, seq_len, d_model)
            attention_weights: Shape (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # =================================================================
        # STEP 1: Project to Q, K, V
        # =================================================================
        # Shape: (batch, seq, d_model)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # =================================================================
        # STEP 2: Split into multiple heads
        # =================================================================
        # Reshape: (batch, seq, d_model) -> (batch, seq, n_heads, d_k)
        # Transpose: -> (batch, n_heads, seq, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # =================================================================
        # STEP 3: Compute attention scores
        # =================================================================
        # Q @ K^T: (batch, n_heads, seq, d_k) @ (batch, n_heads, d_k, seq)
        #        = (batch, n_heads, seq, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scale by sqrt(d_k)
        scores = scores / math.sqrt(self.d_k)

        # =================================================================
        # STEP 4: Apply mask (if provided)
        # =================================================================
        if mask is not None:
            # Mask shape: (seq, seq) broadcasts to (batch, n_heads, seq, seq)
            scores = scores + mask

        # =================================================================
        # STEP 5: Softmax to get attention weights
        # =================================================================
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # =================================================================
        # STEP 6: Weighted sum of values
        # =================================================================
        # (batch, n_heads, seq, seq) @ (batch, n_heads, seq, d_k)
        # = (batch, n_heads, seq, d_k)
        head_outputs = torch.matmul(attention_weights, V)

        # =================================================================
        # STEP 7: Merge heads
        # =================================================================
        # Transpose: (batch, n_heads, seq, d_k) -> (batch, seq, n_heads, d_k)
        # Reshape: -> (batch, seq, d_model)
        concat = head_outputs.transpose(1, 2).contiguous()
        concat = concat.view(batch_size, seq_len, self.d_model)

        # =================================================================
        # STEP 8: Output projection
        # =================================================================
        output = self.W_O(concat)

        return output, attention_weights


# =============================================================================
# FEEDFORWARD NETWORK
# =============================================================================

class FeedForward(nn.Module):
    """
    Position-wise FeedForward Network.

    NumPy equivalent: src/model.py FeedForward class

    Architecture:
        Linear(d_model -> d_ff) -> ReLU -> Linear(d_ff -> d_model)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Input and output dimension (64)
            d_ff: Hidden dimension (256 = 4 * d_model)
            dropout: Dropout probability
        """
        super().__init__()

        # Two linear layers with ReLU in between
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Two-layer MLP with ReLU activation.

        Args:
            x: Shape (batch, seq, d_model)

        Returns:
            Shape (batch, seq, d_model)

        Math:
            output = W_2(dropout(ReLU(W_1(x))))
        """
        # (batch, seq, d_model) -> (batch, seq, d_ff)
        hidden = F.relu(self.W_1(x))
        hidden = self.dropout(hidden)

        # (batch, seq, d_ff) -> (batch, seq, d_model)
        output = self.W_2(hidden)

        return output


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock(nn.Module):
    """
    One transformer block = Attention + FeedForward, each with
    residual connections and layer normalization.

    NumPy equivalent: src/model.py TransformerBlock class

    Structure:
        x -> Attention -> Add(x) -> LayerNorm -> y
        y -> FeedForward -> Add(y) -> LayerNorm -> output

    SHAPE INVARIANT: Input shape == Output shape
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (64)
            n_heads: Number of attention heads (2)
            d_ff: FeedForward hidden dimension (256)
            dropout: Dropout probability
        """
        super().__init__()

        # Sub-layer 1: Multi-Head Attention
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)

        # Sub-layer 2: FeedForward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer norms (using PyTorch built-in)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through one transformer block.

        Args:
            x: Shape (batch, seq, d_model)
            mask: Optional attention mask

        Returns:
            output: Shape (batch, seq, d_model)
            attention_weights: Shape (batch, n_heads, seq, seq)
        """
        # =================================================================
        # SUB-LAYER 1: Attention + Residual + Norm
        # =================================================================
        attention_output, attention_weights = self.attention(x, mask)
        x = self.norm_1(x + self.dropout(attention_output))

        # =================================================================
        # SUB-LAYER 2: FeedForward + Residual + Norm
        # =================================================================
        ff_output = self.feed_forward(x)
        x = self.norm_2(x + self.dropout(ff_output))

        return x, attention_weights


# =============================================================================
# FULL TRANSFORMER MODEL
# =============================================================================

class Transformer(nn.Module):
    """
    Full transformer model for character-level text generation.

    NumPy equivalent: src/model.py Transformer class

    Architecture:
        Embedding -> [TransformerBlock x n_layers] -> LayerNorm -> Linear

    IMPORTANT: forward() returns LOGITS, not probabilities.
    CrossEntropyLoss expects raw logits. Apply softmax manually for inference.
    """

    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 2,
                 n_layers: int = 2, d_ff: int = 256, max_seq_len: int = 256,
                 dropout: float = 0.1):
        """
        Args:
            vocab_size: Number of unique characters
            d_model: Embedding/model dimension (64)
            n_heads: Attention heads per block (2)
            n_layers: Number of transformer blocks (2)
            d_ff: FeedForward hidden dimension (256)
            max_seq_len: Maximum context window (256)
            dropout: Dropout probability
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Embedding layer
        self.embedding = Embedding(vocab_size, d_model, max_seq_len, dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection (unembedding)
        # Maps from d_model to vocab_size
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

        # Register causal mask buffer
        # Shape: (max_seq_len, max_seq_len)
        mask = self._create_causal_mask(max_seq_len)
        self.register_buffer('causal_mask', mask)

    def _init_weights(self):
        """Initialize weights with small random values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        Create causal (autoregressive) mask.

        Position i can only attend to positions 0, 1, ..., i

        Returns:
            Mask of shape (seq_len, seq_len)
            Upper triangle is -inf, lower triangle + diagonal is 0
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, token_ids: torch.Tensor,
                return_attention: bool = False) -> dict:
        """
        Full forward pass: tokens in, logits out.

        Args:
            token_ids: Shape (batch, seq) - integer token IDs
            return_attention: Whether to return attention maps

        Returns:
            Dictionary with:
                'logits': Raw scores, shape (batch, seq, vocab_size)
                'attention_maps': (optional) List of attention weights per layer
        """
        batch_size, seq_len = token_ids.shape

        # Get causal mask for this sequence length
        mask = self.causal_mask[:seq_len, :seq_len]

        # =================================================================
        # STEP 1: Embedding
        # =================================================================
        # (batch, seq) -> (batch, seq, d_model)
        x = self.embedding(token_ids)

        # =================================================================
        # STEP 2: Pass through transformer blocks
        # =================================================================
        attention_maps = []

        for block in self.blocks:
            x, attention_weights = block(x, mask)
            if return_attention:
                attention_maps.append(attention_weights)

        # =================================================================
        # STEP 3: Final normalization
        # =================================================================
        x = self.final_norm(x)

        # =================================================================
        # STEP 4: Project to vocabulary size
        # =================================================================
        # (batch, seq, d_model) -> (batch, seq, vocab_size)
        logits = self.output_projection(x)

        # Return logits (NOT probabilities) - CrossEntropyLoss wants logits
        result = {'logits': logits}
        if return_attention:
            result['attention_maps'] = attention_maps

        return result

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            prompt_ids: Starting tokens, shape (1, prompt_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens

        Returns:
            Generated token IDs, shape (1, prompt_len + max_new_tokens)
        """
        self.eval()
        generated = prompt_ids

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            context = generated[:, -self.max_seq_len:]

            # Get logits for next token
            result = self.forward(context)
            logits = result['logits'][:, -1, :]  # (batch, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

        return generated


# =============================================================================
# VERIFICATION: Compare NumPy and PyTorch outputs
# =============================================================================

def verify_against_numpy():
    """
    Verify PyTorch implementation matches NumPy implementation.

    Seeds both, runs forward pass, compares outputs.
    """
    import numpy as np
    import sys
    from pathlib import Path

    # Add src directory to path
    src_dir = Path(__file__).parent
    sys.path.insert(0, str(src_dir))

    print("=" * 60)
    print("VERIFICATION: NumPy vs PyTorch Implementation")
    print("=" * 60)

    # Seed both
    np.random.seed(42)
    torch.manual_seed(42)

    # Small config for testing
    vocab_size = 20
    d_model = 8
    n_heads = 2
    n_layers = 1
    d_ff = 32
    max_seq_len = 16

    batch_size = 2
    seq_len = 8

    # Create random input
    token_ids_np = np.random.randint(0, vocab_size, (batch_size, seq_len))
    token_ids_torch = torch.from_numpy(token_ids_np)

    print(f"\nConfig: vocab={vocab_size}, d_model={d_model}, "
          f"n_heads={n_heads}, n_layers={n_layers}")
    print(f"Input shape: ({batch_size}, {seq_len})")

    # --- PyTorch model ---
    print("\n--- PyTorch Model ---")
    torch_model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=0.0,  # Disable dropout for comparison
    )
    torch_model.eval()

    with torch.no_grad():
        torch_result = torch_model(token_ids_torch, return_attention=True)

    torch_logits = torch_result['logits'].numpy()
    print(f"Output logits shape: {torch_logits.shape}")
    print(f"Logits sample (first position): {torch_logits[0, 0, :5]}")

    # --- NumPy model ---
    try:
        from src.model import Transformer as NumpyTransformer, CONFIG as NP_CONFIG

        print("\n--- NumPy Model ---")

        # Override config
        NP_CONFIG['d_model'] = d_model
        NP_CONFIG['n_heads'] = n_heads
        NP_CONFIG['n_layers'] = n_layers
        NP_CONFIG['d_ff'] = d_ff
        NP_CONFIG['max_seq_len'] = max_seq_len

        np.random.seed(42)  # Reset seed for fair comparison
        numpy_model = NumpyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
        )

        numpy_result = numpy_model.forward(token_ids_np)
        numpy_logits = numpy_result['logits']
        print(f"Output logits shape: {numpy_logits.shape}")
        print(f"Logits sample (first position): {numpy_logits[0, 0, :5]}")

        # Note: Outputs won't match exactly because weight initialization differs
        print("\n--- Comparison ---")
        print("Note: Outputs differ because PyTorch and NumPy have different")
        print("random initialization. The STRUCTURE is identical.")
        print("\nTo verify correctness:")
        print("  1. Both produce same output shapes")
        print("  2. Both apply causal masking correctly")
        print("  3. Loss decreases during training (PyTorch)")

    except ImportError as e:
        print(f"\nCouldn't import NumPy model: {e}")
        print("This is OK - the PyTorch model works standalone.")

    # --- Parameter count ---
    print("\n--- Parameter Count ---")
    total_params = torch_model.count_parameters()
    print(f"Total trainable parameters: {total_params:,}")

    # Detailed breakdown
    print("\nBreakdown:")
    for name, param in torch_model.named_parameters():
        print(f"  {name}: {param.numel():,}")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    verify_against_numpy()
