#!/usr/bin/env python3
"""
Corpus Download Script for Proust Attention Machine

Attempts to download "En busca del tiempo perdido" (Spanish translation)
from multiple public domain sources. If automated download fails, prints
instructions for manual acquisition.

CORPUS POLICY: Proust only. No fallback to other authors.

Usage:
    python data/download_corpus.py
"""

import os
import sys
import re
import json
import unicodedata
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests/beautifulsoup4 not installed. Install with:")
    print("  pip install requests beautifulsoup4")
    print("Proceeding with manual file processing only.\n")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory structure
SCRIPT_DIR = Path(__file__).parent
RAW_DIR = SCRIPT_DIR / "raw"
PROCESSED_DIR = SCRIPT_DIR / "processed"

# Output files
CORPUS_FILE = PROCESSED_DIR / "proust_corpus.txt"
VOCAB_FILE = PROCESSED_DIR / "vocab.json"

# Proust volumes (all 7)
PROUST_VOLUMES = [
    "Por el camino de Swann",
    "A la sombra de las muchachas en flor",
    "El mundo de Guermantes",
    "Sodoma y Gomorra",
    "La prisionera",
    "La fugitiva",  # Also known as "Albertine desaparecida"
    "El tiempo recobrado",
]

# Sources to try (in order of preference)
SOURCES = [
    {
        "name": "Project Gutenberg",
        "search_url": "https://www.gutenberg.org/ebooks/search/?query=proust+spanish",
        "type": "gutenberg",
    },
    {
        "name": "Internet Archive",
        "search_url": "https://archive.org/search?query=proust%20busca%20tiempo%20perdido%20spanish",
        "type": "archive",
    },
    {
        "name": "Biblioteca Virtual Miguel de Cervantes",
        "search_url": "https://www.cervantesvirtual.com/buscador/?q=proust",
        "type": "cervantes",
    },
]


# =============================================================================
# TEXT CLEANING
# =============================================================================

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode to NFC form.

    NFC = Canonical Decomposition followed by Canonical Composition
    This ensures consistent representation of accented characters.

    Example: 'e' + combining acute = 'e' (single character)
    """
    return unicodedata.normalize('NFC', text)


def clean_gutenberg_text(text: str) -> str:
    """
    Remove Project Gutenberg headers and footers.

    Gutenberg texts have standard markers:
    - "*** START OF THE PROJECT GUTENBERG EBOOK ***"
    - "*** END OF THE PROJECT GUTENBERG EBOOK ***"
    """
    # Find start marker
    start_patterns = [
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .+? \*\*\*",
        r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .+? \*\*\*",
        r"START OF THE PROJECT GUTENBERG",
    ]

    for pattern in start_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[match.end():]
            break

    # Find end marker
    end_patterns = [
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK",
        r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK",
        r"END OF THE PROJECT GUTENBERG",
    ]

    for pattern in end_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
            break

    return text


def clean_generic_headers(text: str) -> str:
    """
    Remove common header/footer patterns from various sources.
    """
    # Remove lines that look like metadata (all caps, short)
    lines = text.split('\n')
    cleaned_lines = []

    skip_patterns = [
        r"^transcribed by",
        r"^scanned by",
        r"^digitized by",
        r"^produced by",
        r"^\s*\d+\s*$",  # Page numbers
        r"^_+$",  # Underline separators
        r"^-+$",  # Dash separators
    ]

    for line in lines:
        line_lower = line.lower().strip()

        # Skip if matches any pattern
        skip = False
        for pattern in skip_patterns:
            if re.match(pattern, line_lower, re.IGNORECASE):
                skip = True
                break

        if not skip:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def collapse_whitespace(text: str) -> str:
    """
    Collapse multiple whitespace characters into single spaces.
    Preserve paragraph breaks (double newlines).
    """
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Collapse multiple spaces to single space
    text = re.sub(r'[ \t]+', ' ', text)

    # Collapse multiple newlines to double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def filter_characters(text: str) -> str:
    """
    Filter text to only include characters valid for Spanish literary text.

    Whitelist approach: define what we WANT, remove everything else.
    This eliminates OCR artifacts, formatting symbols, and programming
    characters that pollute the vocabulary.

    Kept characters:
        - Spanish letters (a-z, A-Z, accented vowels, n-tilde, u-umlaut)
        - Digits (0-9)
        - Common punctuation: . , ; : ! ? - ' "
        - Spanish-specific: inverted ! and ?, em-dash, guillemets, curly quotes
        - Whitespace: space, newline
        - Parentheses: ( )
        - Slash: /
    """
    # Define the whitelist as a set for O(1) lookup
    allowed_chars = set(
        # Basic Latin letters
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # Digits
        '0123456789'
        # Common punctuation
        '.,;:!?-\'"'
        # Whitespace
        ' \n'
        # Parentheses and slash
        '()/'
        # Spanish-specific accented vowels
        '\u00e1\u00e9\u00ed\u00f3\u00fa'  # a e i o u with acute
        '\u00c1\u00c9\u00cd\u00d3\u00da'  # A E I O U with acute
        '\u00f1\u00d1'                      # n-tilde, N-tilde
        '\u00fc'                            # u-umlaut (guell, etc.)
        '\u00bf\u00a1'                      # inverted ? and !
        '\u2014'                            # em-dash
        '\u00ab\u00bb'                      # guillemets << >>
        '\u201c\u201d'                      # curly double quotes
    )

    return ''.join(char for char in text if char in allowed_chars)


def clean_text(text: str, source_type: str = "generic") -> str:
    """
    Full text cleaning pipeline.

    Args:
        text: Raw text content
        source_type: "gutenberg", "archive", "cervantes", or "generic"

    Returns:
        Cleaned text
    """
    # Step 1: Normalize Unicode
    text = normalize_unicode(text)

    # Step 2: Remove source-specific headers
    if source_type == "gutenberg":
        text = clean_gutenberg_text(text)

    # Step 3: Remove generic headers
    text = clean_generic_headers(text)

    # Step 4: Filter to allowed characters only
    text = filter_characters(text)

    # Step 5: Collapse whitespace
    text = collapse_whitespace(text)

    return text


# =============================================================================
# VOCABULARY BUILDING
# =============================================================================

def build_vocab(text: str) -> dict:
    """
    Build character vocabulary from text.

    Returns:
        Dictionary with:
        - char_to_idx: mapping from character to integer
        - idx_to_char: mapping from integer to character
        - vocab_size: number of unique characters
        - char_counts: frequency of each character
    """
    # Count character frequencies
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1

    # Sort characters for reproducibility
    unique_chars = sorted(char_counts.keys())

    # Create mappings
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}

    return {
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "vocab_size": len(unique_chars),
        "char_counts": char_counts,
    }


def print_vocab_stats(vocab: dict, text: str) -> None:
    """Print vocabulary statistics."""
    print("\n" + "=" * 60)
    print("CORPUS STATISTICS")
    print("=" * 60)

    print(f"\nTotal characters: {len(text):,}")
    print(f"Unique characters (vocab size): {vocab['vocab_size']}")

    # Show most common characters
    sorted_chars = sorted(
        vocab['char_counts'].items(),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\nMost common characters:")
    for char, count in sorted_chars[:15]:
        display = repr(char) if char in ' \n\t' else char
        pct = 100 * count / len(text)
        print(f"  {display:6s}: {count:>8,} ({pct:5.2f}%)")

    # Show Spanish-specific characters
    spanish_chars = ['a', 'e', 'i', 'o', 'u', 'n',
                     '\u00e1', '\u00e9', '\u00ed', '\u00f3', '\u00fa',  # aeiou with acute
                     '\u00f1', '\u00fc',  # n tilde, u umlaut
                     '\u00bf', '\u00a1']  # inverted ? and !

    print(f"\nSpanish-specific characters found:")
    for char in spanish_chars:
        if char in vocab['char_counts']:
            count = vocab['char_counts'][char]
            print(f"  '{char}': {count:,}")


# =============================================================================
# FILE PROCESSING
# =============================================================================

def process_raw_files() -> str:
    """
    Process any .txt files found in data/raw/ directory.

    This is the fallback when automated download fails.
    Users can manually download Proust texts and place them here.

    Returns:
        Concatenated and cleaned text, or empty string if no files found.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = list(RAW_DIR.glob("*.txt"))

    if not txt_files:
        return ""

    print(f"\nFound {len(txt_files)} text file(s) in {RAW_DIR}:")
    for f in txt_files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")

    # Read and concatenate all files
    all_text = []
    for txt_file in sorted(txt_files):
        print(f"\nProcessing: {txt_file.name}")

        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(txt_file, 'r', encoding=encoding) as f:
                    text = f.read()
                print(f"  Encoding: {encoding}")
                print(f"  Raw length: {len(text):,} characters")

                # Clean the text
                text = clean_text(text)
                print(f"  Cleaned length: {len(text):,} characters")

                all_text.append(text)
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"  Warning: Could not decode {txt_file.name}")

    return "\n\n".join(all_text)


# =============================================================================
# DOWNLOAD ATTEMPTS (Optional - requires requests)
# =============================================================================

def try_download_sources() -> str:
    """
    Attempt to download Proust corpus from public sources.

    Note: Most sources require manual navigation/authentication.
    This function primarily provides guidance rather than automated download.

    Returns:
        Downloaded text, or empty string if unsuccessful.
    """
    if not REQUESTS_AVAILABLE:
        return ""

    print("\n" + "=" * 60)
    print("ATTEMPTING AUTOMATED DOWNLOAD")
    print("=" * 60)
    print("\nNote: Most public domain sources require manual download.")
    print("This script will check availability and provide instructions.\n")

    # For now, we don't attempt actual downloads as most sources
    # require JavaScript or authentication. Instead, we check
    # availability and guide the user.

    for source in SOURCES:
        print(f"\nChecking: {source['name']}")
        print(f"  URL: {source['search_url']}")

        try:
            response = requests.get(
                source['search_url'],
                timeout=10,
                headers={'User-Agent': 'ProustAttentionMachine/1.0'}
            )

            if response.status_code == 200:
                print(f"  Status: Accessible")

                # Check if Proust is mentioned in results
                if 'proust' in response.text.lower():
                    print(f"  Proust content may be available!")
                else:
                    print(f"  No Proust content detected in search results")
            else:
                print(f"  Status: {response.status_code}")

        except requests.RequestException as e:
            print(f"  Error: {e}")

    return ""


def print_manual_instructions():
    """Print instructions for manual corpus acquisition."""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)

    print("""
To train the Proust Attention Machine, you need the Spanish text of
"En busca del tiempo perdido" (In Search of Lost Time).

The 7 volumes are:
""")

    for i, volume in enumerate(PROUST_VOLUMES, 1):
        print(f"  {i}. {volume}")

    print("""
WHERE TO FIND THE TEXT:

1. Project Gutenberg (gutenberg.org)
   - Search for "Proust Spanish" or "busca tiempo perdido"
   - Download as plain text (.txt)

2. Internet Archive (archive.org)
   - Search for "Proust En busca del tiempo perdido Spanish"
   - Look for text versions or OCR'd scans

3. Biblioteca Virtual Miguel de Cervantes (cervantesvirtual.com)
   - Spanish literature digital library
   - Search for "Marcel Proust"

4. Wikisource (es.wikisource.org)
   - Search for "En busca del tiempo perdido"
   - May have partial or complete texts

5. Epublibre / Anna's Archive / Library Genesis
   - Public domain aggregators
   - Search for Spanish Proust translations

AFTER DOWNLOADING:

1. Save the text file(s) to: data/raw/
   - Any filename ending in .txt will be processed
   - Multiple files will be concatenated

2. Re-run this script:
   python data/download_corpus.py

The script will clean and process the text automatically.

MINIMUM CORPUS SIZE:
- Ideal: All 7 volumes (~3-4 million characters)
- Minimum: 1 volume (~500k characters) for initial experiments
- Even a few chapters will work for testing the pipeline
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for corpus download/processing."""
    print("=" * 60)
    print("PROUST ATTENTION MACHINE - CORPUS ACQUISITION")
    print("=" * 60)

    # Ensure directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    corpus_text = ""

    # Step 1: Check for existing processed corpus
    if CORPUS_FILE.exists():
        print(f"\nExisting corpus found: {CORPUS_FILE}")
        size_mb = CORPUS_FILE.stat().st_size / (1024 * 1024)
        print(f"Size: {size_mb:.2f} MB")

        response = input("\nReprocess raw files? (y/N): ").strip().lower()
        if response != 'y':
            print("Using existing corpus.")
            with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
                corpus_text = f.read()
        else:
            corpus_text = ""

    # Step 2: Process raw files if needed
    if not corpus_text:
        corpus_text = process_raw_files()

    # Step 3: Try automated download if still no corpus
    if not corpus_text and REQUESTS_AVAILABLE:
        corpus_text = try_download_sources()

    # Step 4: If still no corpus, print instructions
    if not corpus_text:
        print_manual_instructions()
        print("\n" + "=" * 60)
        print("No corpus available yet.")
        print("Please download text files to data/raw/ and re-run.")
        print("=" * 60)
        sys.exit(1)

    # Step 5: Save processed corpus
    print(f"\nSaving processed corpus to: {CORPUS_FILE}")
    with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
        f.write(corpus_text)

    # Step 6: Build and save vocabulary
    vocab = build_vocab(corpus_text)

    print(f"Saving vocabulary to: {VOCAB_FILE}")
    # Convert idx_to_char keys to strings for JSON
    vocab_json = {
        "char_to_idx": vocab["char_to_idx"],
        "idx_to_char": {str(k): v for k, v in vocab["idx_to_char"].items()},
        "vocab_size": vocab["vocab_size"],
    }
    with open(VOCAB_FILE, 'w', encoding='utf-8') as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    # Step 7: Print statistics
    print_vocab_stats(vocab, corpus_text)

    print("\n" + "=" * 60)
    print("CORPUS READY")
    print("=" * 60)
    print(f"\nCorpus file: {CORPUS_FILE}")
    print(f"Vocabulary file: {VOCAB_FILE}")
    print(f"\nYou can now run: python src/train.py")


if __name__ == "__main__":
    main()
