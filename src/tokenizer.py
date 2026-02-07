"""
Character-Level Tokenizer for Proust Attention Machine

PURPOSE:
    Convert text (strings) to numbers (integers) and back.
    This is just a lookup table - no learning happens here.

ANALOGY:
    Like a phone book: each character gets a unique ID number.
    The number itself is meaningless - just an address.
"""

import numpy as np
from typing import Dict


class CharTokenizer:
    """
    Maps characters <-> integers.

    Attributes:
        char_to_idx: Dictionary mapping character -> integer
        idx_to_char: Dictionary mapping integer -> character
        vocab_size: Number of unique characters
    """

    def __init__(self):
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size: int = 0

    def build_vocab(self, text: str) -> None:
        """
        Build vocabulary from text. Call this ONCE with full corpus.

        After calling, indices are FIXED forever.
        """
        # Sorted for reproducibility (same text = same indices)
        unique_chars = sorted(list(set(text)))

        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)

    def encode(self, text: str) -> np.ndarray:
        """
        String -> array of integers.

        "gato" -> [15, 0, 28, 23]
        Shape: (len(text),)
        """
        return np.array([self.char_to_idx[c] for c in text], dtype=np.int64)

    def decode(self, tokens: np.ndarray) -> str:
        """
        Array of integers -> string.

        [15, 0, 28, 23] -> "gato"
        """
        return ''.join(self.idx_to_char[int(t)] for t in tokens)
