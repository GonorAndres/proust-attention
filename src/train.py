#!/usr/bin/env python3
"""
Training Script for Proust Attention Machine

=============================================================================
PURPOSE
=============================================================================

Complete training loop for the character-level transformer. Trains the model
to predict the next character in Proust's "En busca del tiempo perdido".

Key features:
- AdamW optimizer with weight decay
- Cosine learning rate schedule with warmup
- Gradient clipping for stability
- Periodic sample generation to monitor progress
- Checkpoint saving

Usage:
    python src/train.py --epochs 50 --batch_size 32
    python src/train.py --epochs 2 --batch_size 4  # Quick test

=============================================================================
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_torch import Transformer, CONFIG
from src.dataset import create_dataloader, create_dataloaders, ProustDataset
from src.tokenizer import CharTokenizer


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TRAIN_CONFIG = {
    # Optimizer
    'lr': 3e-4,                 # Learning rate
    'weight_decay': 0.01,       # L2 regularization
    'betas': (0.9, 0.95),       # Adam betas

    # Learning rate schedule
    'warmup_steps': 500,        # Linear warmup steps

    # Gradient clipping
    'max_grad_norm': 1.0,       # Clip gradient norm

    # Logging
    'log_interval': 50,         # Print loss every N steps
    'sample_interval': 500,     # Generate sample every N steps
    'sample_length': 200,       # Characters to generate in samples
    'sample_prompt': "Mucho tiempo",  # Prompt for samples

    # Checkpoints
    'save_interval': 1,         # Save every N epochs
}


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """
    Create learning rate scheduler with linear warmup and cosine decay.

    Schedule:
        - Steps 0 to warmup_steps: linear increase from 0 to lr
        - Steps warmup_steps to total_steps: cosine decay to 0

    This helps training stability:
        - Warmup prevents large early gradients from destabilizing
        - Cosine decay gradually reduces learning rate for fine-tuning
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup: 0 -> 1
            return step / warmup_steps
        else:
            # Cosine decay: 1 -> 0
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# SAMPLE GENERATION
# =============================================================================

@torch.no_grad()
def generate_sample(model: Transformer, tokenizer: CharTokenizer,
                    prompt: str, max_length: int, device: torch.device,
                    temperature: float = 0.8) -> str:
    """
    Generate a text sample from the model.

    Used during training to monitor learning progress.
    """
    model.eval()

    # Encode prompt
    prompt_ids = tokenizer.encode(prompt)
    prompt_ids = torch.from_numpy(prompt_ids).long().unsqueeze(0).to(device)

    # Generate
    generated = model.generate(
        prompt_ids,
        max_new_tokens=max_length,
        temperature=temperature,
        top_k=40,
    )

    # Decode
    text = tokenizer.decode(generated[0].cpu().numpy())

    model.train()
    return text


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: Transformer,
    dataloader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    global_step: int,
    tokenizer: CharTokenizer,
    config: dict,
) -> tuple[float, int]:
    """
    Train for one epoch.

    Returns:
        Tuple of (average_loss, updated_global_step)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    loss_fn = nn.CrossEntropyLoss()

    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        # Move to device
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Forward pass
        result = model(input_ids)
        logits = result['logits']

        # Reshape for cross entropy: (batch * seq, vocab) vs (batch * seq,)
        batch_size, seq_len, vocab_size = logits.shape
        loss = loss_fn(
            logits.view(-1, vocab_size),
            target_ids.view(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['max_grad_norm']
        )

        # Update weights
        optimizer.step()
        scheduler.step()

        # Accumulate stats
        total_loss += loss.item()
        n_batches += 1
        global_step += 1

        # Logging
        if global_step % config['log_interval'] == 0:
            avg_loss = total_loss / n_batches
            lr = scheduler.get_last_lr()[0]
            print(f"  Step {global_step:5d} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Avg: {avg_loss:.4f} | "
                  f"LR: {lr:.2e}")

        # Sample generation
        if global_step % config['sample_interval'] == 0:
            print("\n  --- Sample Generation ---")
            sample = generate_sample(
                model, tokenizer,
                config['sample_prompt'],
                config['sample_length'],
                device
            )
            # Print with line wrapping
            print(f"  Prompt: '{config['sample_prompt']}'")
            print(f"  Generated:")
            for i in range(0, len(sample), 70):
                print(f"    {sample[i:i+70]}")
            print("  " + "-" * 40 + "\n")

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return avg_loss, global_step


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate(model: Transformer, dataloader, device: torch.device) -> float:
    """
    Evaluate model on a validation DataLoader.

    Returns:
        Average cross-entropy loss over all validation batches.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    loss_fn = nn.CrossEntropyLoss()

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        result = model(input_ids)
        logits = result['logits']

        batch_size, seq_len, vocab_size = logits.shape
        loss = loss_fn(logits.view(-1, vocab_size), target_ids.view(-1))

        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / n_batches if n_batches > 0 else 0.0


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def save_checkpoint(
    model: Transformer,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    loss: float,
    checkpoint_dir: Path,
    tokenizer: CharTokenizer,
    model_config: dict,
    val_loss: float = None,
):
    """Save model checkpoint. Only overwrites best.pt when val_loss improves."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'val_loss': val_loss,
        'model_config': model_config,
        'vocab': {
            'char_to_idx': tokenizer.char_to_idx,
            'idx_to_char': tokenizer.idx_to_char,
            'vocab_size': tokenizer.vocab_size,
        },
    }

    # Save epoch checkpoint
    path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")

    # Save as best.pt only if val_loss improved
    best_path = checkpoint_dir / "best.pt"
    save_best = True
    if val_loss is not None and best_path.exists():
        prev_best = torch.load(best_path, map_location='cpu', weights_only=False)
        prev_val_loss = prev_best.get('val_loss')
        if prev_val_loss is not None and val_loss >= prev_val_loss:
            save_best = False

    if save_best:
        torch.save(checkpoint, best_path)
        print(f"Saved new best model (val_loss={val_loss:.4f})"
              if val_loss is not None else "Saved best.pt")


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(
    corpus_path: str,
    vocab_path: str,
    checkpoint_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    context_length: int = 256,
    resume_from: str = None,
    device: str = None,
):
    """
    Main training function.

    Args:
        corpus_path: Path to processed corpus
        vocab_path: Path to vocabulary JSON
        checkpoint_dir: Directory for checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        context_length: Context window length
        resume_from: Optional checkpoint to resume from
        device: Device to train on ('cuda', 'cpu', or None for auto)
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")

    # Create dataloader
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        corpus_path=corpus_path,
        vocab_path=vocab_path,
        batch_size=batch_size,
        context_length=context_length,
    )

    tokenizer = train_dataset.tokenizer
    vocab_size = train_dataset.vocab_size

    print(f"Vocabulary size: {vocab_size}")
    print(f"Batch size: {batch_size}")
    print(f"Context length: {context_length}")
    print(f"Train steps per epoch: {len(train_loader)}")
    print(f"Val steps per epoch: {len(val_loader)}")

    # Model configuration
    model_config = {
        'vocab_size': vocab_size,
        'd_model': CONFIG['d_model'],
        'n_heads': CONFIG['n_heads'],
        'n_layers': CONFIG['n_layers'],
        'd_ff': CONFIG['d_ff'],
        'max_seq_len': context_length,
        'dropout': CONFIG['dropout'],
    }

    # Create model
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)

    model = Transformer(**model_config)
    model = model.to(device)

    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")
    print(f"Model config: {model_config}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG['lr'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
        betas=TRAIN_CONFIG['betas'],
    )

    total_steps = epochs * len(train_loader)
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=TRAIN_CONFIG['warmup_steps'],
        total_steps=total_steps,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0

    if resume_from:
        checkpoint = load_checkpoint(Path(resume_from), device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"Epochs: {epochs}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {TRAIN_CONFIG['warmup_steps']}")
    print()

    checkpoint_dir = Path(checkpoint_dir)
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        avg_loss, global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            global_step=global_step,
            tokenizer=tokenizer,
            config=TRAIN_CONFIG,
        )

        # Validation
        val_loss = validate(model, val_loader, device)

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch + 1} complete | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # Save checkpoint
        if (epoch + 1) % TRAIN_CONFIG['save_interval'] == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                loss=avg_loss,
                checkpoint_dir=checkpoint_dir,
                tokenizer=tokenizer,
                model_config=model_config,
                val_loss=val_loss,
            )

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Final checkpoint: {checkpoint_dir / 'best.pt'}")
    print(f"\nTo generate text:")
    print(f"  python src/generate.py --checkpoint {checkpoint_dir / 'best.pt'}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train the Proust Attention Machine"
    )

    # Data paths
    parser.add_argument(
        '--corpus', type=str,
        default='data/processed/proust_corpus.txt',
        help='Path to processed corpus'
    )
    parser.add_argument(
        '--vocab', type=str,
        default='data/processed/vocab.json',
        help='Path to vocabulary JSON'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str,
        default='checkpoints',
        help='Directory for saving checkpoints'
    )

    # Training parameters
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--context-length', type=int, default=256,
        help='Context window length'
    )

    # Resume training
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Checkpoint to resume from'
    )

    # Device
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device (cuda/cpu)'
    )

    args = parser.parse_args()

    # Check corpus exists
    if not Path(args.corpus).exists():
        print(f"Error: Corpus not found at {args.corpus}")
        print("Please run: python data/download_corpus.py")
        sys.exit(1)

    # Run training
    train(
        corpus_path=args.corpus,
        vocab_path=args.vocab,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        context_length=args.context_length,
        resume_from=args.resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
