#!/usr/bin/env python3
"""
Text Generation for Proust Attention Machine

=============================================================================
PURPOSE
=============================================================================

Autoregressive text generation from a trained model. Given a prompt,
the model generates text one character at a time by:

1. Encode prompt to token IDs
2. Forward pass through model -> get logits for next char
3. Apply temperature scaling and optional top-k filtering
4. Sample from probability distribution
5. Append sampled token, repeat

=============================================================================
SAMPLING STRATEGIES
=============================================================================

Temperature:
    - temperature = 1.0: Use raw probabilities (baseline)
    - temperature < 1.0: Sharper distribution (more deterministic)
    - temperature > 1.0: Flatter distribution (more random)

    Math: probs = softmax(logits / temperature)

Top-k:
    - Only sample from the k most likely tokens
    - Prevents sampling very unlikely tokens (gibberish)
    - k=40 is a common choice

=============================================================================
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_torch import Transformer
from src.tokenizer import CharTokenizer


# =============================================================================
# TEXT GENERATION
# =============================================================================

@torch.no_grad()
def generate(
    model: Transformer,
    tokenizer: CharTokenizer,
    prompt: str,
    max_length: int = 500,
    temperature: float = 1.0,
    top_k: int = None,
    device: torch.device = None,
    show_progress: bool = True,
) -> str:
    """
    Generate text from a trained model.

    Args:
        model: Trained Transformer model
        tokenizer: Character tokenizer
        prompt: Starting text
        max_length: Maximum characters to generate
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top k tokens
        device: Device to run on
        show_progress: Whether to print generation progress

    Returns:
        Generated text (including prompt)
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    # Encode prompt
    prompt_ids = tokenizer.encode(prompt)
    generated = torch.from_numpy(prompt_ids).long().unsqueeze(0).to(device)

    if show_progress:
        print(f"Prompt: '{prompt}'")
        print(f"Generating {max_length} characters...")
        print("-" * 50)
        print(prompt, end='', flush=True)

    # Generate one token at a time
    for i in range(max_length):
        # Truncate to max_seq_len if needed
        context = generated[:, -model.max_seq_len:]

        # Forward pass
        result = model(context)
        logits = result['logits'][:, -1, :]  # (1, vocab_size)

        # Apply temperature
        logits = logits / temperature

        # Optional top-k filtering
        if top_k is not None and top_k > 0:
            # Get top k values and indices
            top_k_val = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k_val)
            # Set everything below the k-th value to -inf
            threshold = values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < threshold,
                torch.full_like(logits, float('-inf')),
                logits
            )

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample from distribution
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        generated = torch.cat([generated, next_token], dim=1)

        # Print generated character
        if show_progress:
            char = tokenizer.idx_to_char[next_token.item()]
            print(char, end='', flush=True)

    if show_progress:
        print("\n" + "-" * 50)

    # Decode full sequence
    text = tokenizer.decode(generated[0].cpu().numpy())
    return text


def generate_with_attention(
    model: Transformer,
    tokenizer: CharTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    device: torch.device = None,
) -> tuple[str, list]:
    """
    Generate text and return attention maps.

    Useful for visualization of what the model attends to.

    Returns:
        Tuple of (generated_text, list_of_attention_maps)
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    # Encode prompt
    prompt_ids = tokenizer.encode(prompt)
    generated = torch.from_numpy(prompt_ids).long().unsqueeze(0).to(device)

    all_attention_maps = []

    for i in range(max_length):
        context = generated[:, -model.max_seq_len:]

        # Forward with attention
        result = model(context, return_attention=True)
        logits = result['logits'][:, -1, :]
        attention_maps = result['attention_maps']

        # Store attention for this step
        all_attention_maps.append([
            attn.cpu().numpy() for attn in attention_maps
        ])

        # Sample
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

    text = tokenizer.decode(generated[0].cpu().numpy())
    return text, all_attention_maps


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model(checkpoint_path: str, device: torch.device = None) -> tuple:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract configuration
    model_config = checkpoint['model_config']
    vocab_data = checkpoint['vocab']

    print(f"Model config: {model_config}")

    # Create model
    model = Transformer(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create tokenizer
    tokenizer = CharTokenizer()
    tokenizer.char_to_idx = vocab_data['char_to_idx']
    tokenizer.idx_to_char = {
        int(k) if isinstance(k, str) else k: v
        for k, v in vocab_data['idx_to_char'].items()
    }
    tokenizer.vocab_size = vocab_data['vocab_size']

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model parameters: {model.count_parameters():,}")

    return model, tokenizer


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_mode(model, tokenizer, device, temperature=0.8, top_k=40):
    """
    Interactive text generation loop.

    Type a prompt, get generated text. Type 'quit' to exit.
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE GENERATION MODE")
    print("=" * 60)
    print("Enter a prompt to generate text.")
    print("Commands:")
    print("  'quit' or 'exit' - Exit interactive mode")
    print("  't=X' - Set temperature to X (e.g., 't=0.5')")
    print("  'k=X' - Set top-k to X (e.g., 'k=20')")
    print("=" * 60)

    current_temp = temperature
    current_k = top_k

    while True:
        try:
            prompt = input(f"\nPrompt [temp={current_temp}, k={current_k}]: ").strip()

            if not prompt:
                continue

            if prompt.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break

            # Handle temperature setting
            if prompt.startswith('t='):
                try:
                    current_temp = float(prompt[2:])
                    print(f"Temperature set to {current_temp}")
                    continue
                except ValueError:
                    print("Invalid temperature. Use 't=0.8' format.")
                    continue

            # Handle top-k setting
            if prompt.startswith('k='):
                try:
                    current_k = int(prompt[2:])
                    print(f"Top-k set to {current_k}")
                    continue
                except ValueError:
                    print("Invalid top-k. Use 'k=40' format.")
                    continue

            # Generate text
            print()
            text = generate(
                model, tokenizer, prompt,
                max_length=300,
                temperature=current_temp,
                top_k=current_k,
                device=device,
                show_progress=True,
            )

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate text from trained Proust model"
    )

    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--prompt', type=str, default=None,
        help='Text prompt for generation'
    )
    parser.add_argument(
        '--length', type=int, default=500,
        help='Number of characters to generate'
    )
    parser.add_argument(
        '--temperature', type=float, default=0.8,
        help='Sampling temperature (higher = more random)'
    )
    parser.add_argument(
        '--top-k', type=int, default=40,
        help='Top-k sampling (0 = disabled)'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device (cuda/cpu)'
    )
    parser.add_argument(
        '--interactive', '-i', action='store_true',
        help='Interactive generation mode'
    )

    args = parser.parse_args()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, tokenizer = load_model(args.checkpoint, device)

    # Interactive mode
    if args.interactive:
        interactive_mode(
            model, tokenizer, device,
            temperature=args.temperature,
            top_k=args.top_k
        )
        return

    # Single generation
    if args.prompt is None:
        args.prompt = "Mucho tiempo"

    print("\n" + "=" * 60)
    print("TEXT GENERATION")
    print("=" * 60)
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Length: {args.length}")
    print()

    text = generate(
        model, tokenizer, args.prompt,
        max_length=args.length,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        device=device,
        show_progress=True,
    )

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
