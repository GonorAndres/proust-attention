#!/usr/bin/env python3
"""
PyTorch Dataset for Proust Attention Machine

=============================================================================
PURPOSE
=============================================================================

Wraps the Proust corpus as a PyTorch Dataset for training. Each sample is:
- Input:  256 characters (context window)
- Target: 256 characters shifted by 1 (what comes next)

For character-level language modeling:
    Input:  "Mucho tiempo he estado acostándome "
    Target: "ucho tiempo he estado acostándome t"

The model learns to predict the next character at each position.

=============================================================================
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer import CharTokenizer


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONTEXT_LENGTH = 256  # Characters of context (from CLAUDE.md)


# =============================================================================
# DATASET
# =============================================================================

class ProustDataset(Dataset):
    """
    PyTorch Dataset wrapping the Proust corpus.

    Each sample returns:
        - input_ids:  (context_length,) tensor of token IDs
        - target_ids: (context_length,) tensor of token IDs, shifted by 1

    The dataset uses overlapping windows with stride=1, giving us
    approximately (corpus_length - context_length) samples.
    """

    def __init__(self, corpus_path: str, vocab_path: str = None,
                 context_length: int = DEFAULT_CONTEXT_LENGTH,
                 tokenizer: CharTokenizer = None):
        """
        Args:
            corpus_path: Path to processed corpus text file
            vocab_path: Path to vocabulary JSON file (optional if tokenizer provided)
            context_length: Number of characters per sample
            tokenizer: Optional pre-built tokenizer (if None, loads from vocab_path)
        """
        self.context_length = context_length

        # Load corpus
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus not found: {corpus_path}\n"
                f"Run: python data/download_corpus.py"
            )

        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.text = f.read()

        print(f"Loaded corpus: {len(self.text):,} characters")

        # Load or build tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = CharTokenizer()

            if vocab_path and Path(vocab_path).exists():
                # Load vocabulary from JSON
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab_data = json.load(f)

                self.tokenizer.char_to_idx = vocab_data['char_to_idx']
                self.tokenizer.idx_to_char = {
                    int(k): v for k, v in vocab_data['idx_to_char'].items()
                }
                self.tokenizer.vocab_size = vocab_data['vocab_size']
                print(f"Loaded vocabulary: {self.tokenizer.vocab_size} characters")
            else:
                # Build vocabulary from corpus
                print("Building vocabulary from corpus...")
                self.tokenizer.build_vocab(self.text)
                print(f"Built vocabulary: {self.tokenizer.vocab_size} characters")

        # Encode entire corpus once (memory efficient for char-level)
        print("Encoding corpus...")
        self.encoded = self.tokenizer.encode(self.text)
        self.encoded = torch.from_numpy(self.encoded).long()
        print(f"Encoded shape: {self.encoded.shape}")

        # Calculate number of samples
        # We need context_length + 1 characters for each sample
        # (context_length for input, context_length for target with 1 char shift)
        self.n_samples = len(self.encoded) - context_length

        if self.n_samples <= 0:
            raise ValueError(
                f"Corpus too short ({len(self.encoded)} tokens) "
                f"for context length {context_length}"
            )

        print(f"Dataset size: {self.n_samples:,} samples")

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input_ids, target_ids), each shape (context_length,)

        Example for context_length=5:
            If text is "ABCDEFGH" and idx=0:
                input_ids  = encode("ABCDE") = [0, 1, 2, 3, 4]
                target_ids = encode("BCDEF") = [1, 2, 3, 4, 5]

            The model learns: given A, predict B; given AB, predict C; etc.
        """
        # Get input sequence
        input_ids = self.encoded[idx : idx + self.context_length]

        # Get target sequence (shifted by 1)
        target_ids = self.encoded[idx + 1 : idx + 1 + self.context_length]

        return input_ids, target_ids

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.tokenizer.vocab_size

    def decode(self, token_ids: torch.Tensor) -> str:
        """Convert token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        return self.tokenizer.decode(token_ids)


# =============================================================================
# DATA LOADER FACTORY
# =============================================================================

def create_dataloader(
    corpus_path: str,
    vocab_path: str = None,
    batch_size: int = 32,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    shuffle: bool = True,
    num_workers: int = 0,
    tokenizer: CharTokenizer = None,
) -> Tuple[DataLoader, ProustDataset]:
    """
    Create a DataLoader for the Proust corpus.

    Args:
        corpus_path: Path to processed corpus
        vocab_path: Path to vocabulary JSON
        batch_size: Samples per batch
        context_length: Characters per sample
        shuffle: Whether to shuffle samples
        num_workers: DataLoader workers (0 = main process)
        tokenizer: Optional pre-built tokenizer

    Returns:
        Tuple of (DataLoader, ProustDataset)
    """
    dataset = ProustDataset(
        corpus_path=corpus_path,
        vocab_path=vocab_path,
        context_length=context_length,
        tokenizer=tokenizer,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader, dataset


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PROUST DATASET DEMONSTRATION")
    print("=" * 60)

    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    corpus_path = data_dir / "proust_corpus.txt"
    vocab_path = data_dir / "vocab.json"

    if not corpus_path.exists():
        print(f"\nCorpus not found at: {corpus_path}")
        print("Creating a sample corpus for demonstration...\n")

        # Create sample corpus
        sample_text = """
        Mucho tiempo he estado acostándome temprano. A veces, apenas había
        apagado la bujía, cerrábanse mis ojos tan presto, que ni tiempo tenía
        para decirme: «Ya me duermo.» Y media hora después despertábame la idea
        de que ya era hora de ir a buscar el sueño; quería dejar el libro, que
        se me figuraba tener aún entre las manos, y apagar de un soplo la luz;
        durante mi sueño no había cesado de reflexionar sobre lo recién leído,
        pero era muy particular el tono que tomaban estas reflexiones, porque
        me parecía que yo pasaba a convertirme en el tema de la obra.
        """ * 100  # Repeat to have enough data

        # Create directories and files
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(corpus_path, 'w', encoding='utf-8') as f:
            f.write(sample_text.strip())

        # Build vocab
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(sample_text)
        vocab_data = {
            "char_to_idx": tokenizer.char_to_idx,
            "idx_to_char": {str(k): v for k, v in tokenizer.idx_to_char.items()},
            "vocab_size": tokenizer.vocab_size,
        }
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

        print(f"Created sample corpus: {len(sample_text):,} characters")

    # Create dataset
    print("\n--- Creating Dataset ---")
    context_length = 64  # Smaller for demo
    dataset = ProustDataset(
        corpus_path=str(corpus_path),
        vocab_path=str(vocab_path),
        context_length=context_length,
    )

    print(f"\nDataset length: {len(dataset):,}")
    print(f"Vocabulary size: {dataset.vocab_size}")

    # Get a sample
    print("\n--- Sample Data ---")
    input_ids, target_ids = dataset[0]

    print(f"Input shape:  {input_ids.shape}")
    print(f"Target shape: {target_ids.shape}")

    # Decode to show text
    input_text = dataset.decode(input_ids)
    target_text = dataset.decode(target_ids)

    print(f"\nInput text (first {context_length} chars):")
    print(f"  '{input_text[:60]}...'")
    print(f"\nTarget text (shifted by 1):")
    print(f"  '{target_text[:60]}...'")

    # Show alignment
    print("\n--- Alignment Example (first 20 chars) ---")
    print("Position:  ", end="")
    for i in range(20):
        print(f"{i:3d}", end="")
    print()
    print("Input:     ", end="")
    for c in input_text[:20]:
        display = repr(c)[1:-1] if c in ' \n' else c
        print(f"{display:>3}", end="")
    print()
    print("Target:    ", end="")
    for c in target_text[:20]:
        display = repr(c)[1:-1] if c in ' \n' else c
        print(f"{display:>3}", end="")
    print()
    print("\nAt each position, the model predicts the target from the input.")

    # Create DataLoader
    print("\n--- DataLoader ---")
    dataloader, _ = create_dataloader(
        corpus_path=str(corpus_path),
        vocab_path=str(vocab_path),
        batch_size=4,
        context_length=context_length,
    )

    # Get a batch
    for batch_input, batch_target in dataloader:
        print(f"Batch input shape:  {batch_input.shape}")
        print(f"Batch target shape: {batch_target.shape}")
        print(f"  (batch_size, context_length)")
        break

    print("\n" + "=" * 60)
    print("DATASET READY FOR TRAINING")
    print("=" * 60)
