#!/usr/bin/env python3
"""
Corpus Preparation Utility

This script processes text files already present in data/raw/
without attempting any downloads. Use this when you've manually
obtained the Proust corpus.

Usage:
    python data/prepare_corpus.py

Features:
    - Processes all .txt files in data/raw/
    - Cleans headers, normalizes Unicode, collapses whitespace
    - Outputs to data/processed/proust_corpus.txt
    - Generates vocabulary file data/processed/vocab.json
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from download_corpus (shared cleaning functions)
from download_corpus import (
    RAW_DIR,
    PROCESSED_DIR,
    CORPUS_FILE,
    VOCAB_FILE,
    process_raw_files,
    build_vocab,
    print_vocab_stats,
)
import json


def main():
    """Process raw text files into corpus."""
    print("=" * 60)
    print("PROUST ATTENTION MACHINE - CORPUS PREPARATION")
    print("=" * 60)

    # Ensure directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Check for raw files
    txt_files = list(RAW_DIR.glob("*.txt"))

    if not txt_files:
        print(f"\nNo .txt files found in: {RAW_DIR}")
        print("\nPlease place your Proust text files in that directory.")
        print("Supported: Any .txt files with UTF-8 or Latin-1 encoding.")
        sys.exit(1)

    # Process raw files
    corpus_text = process_raw_files()

    if not corpus_text:
        print("\nError: Could not extract text from files.")
        sys.exit(1)

    # Save processed corpus
    print(f"\nSaving processed corpus to: {CORPUS_FILE}")
    with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
        f.write(corpus_text)

    # Build and save vocabulary
    vocab = build_vocab(corpus_text)

    print(f"Saving vocabulary to: {VOCAB_FILE}")
    vocab_json = {
        "char_to_idx": vocab["char_to_idx"],
        "idx_to_char": {str(k): v for k, v in vocab["idx_to_char"].items()},
        "vocab_size": vocab["vocab_size"],
    }
    with open(VOCAB_FILE, 'w', encoding='utf-8') as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    # Print statistics
    print_vocab_stats(vocab, corpus_text)

    print("\n" + "=" * 60)
    print("CORPUS READY")
    print("=" * 60)
    print(f"\nCorpus: {CORPUS_FILE}")
    print(f"Vocab: {VOCAB_FILE}")
    print(f"\nNext step: python src/train.py")


if __name__ == "__main__":
    main()
