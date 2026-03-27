---
language: es
license: mit
tags:
  - text-generation
  - character-level
  - transformer
  - pytorch
  - from-scratch
  - proust
datasets:
  - custom
pipeline_tag: text-generation
---

# Proust Attention Machine

A character-level transformer trained from scratch on Marcel Proust's *En busca del tiempo perdido* (Spanish translation, all 7 volumes).

Built as an educational project to deeply understand attention mechanisms -- every component was implemented manually, first in NumPy, then ported to PyTorch for training.

## Model Details

| Parameter | Value |
|-----------|-------|
| Architecture | Decoder-only Transformer |
| Parameters | ~420,000 |
| Layers | 2 |
| Attention heads | 2 |
| Embedding dim | 128 |
| Feedforward dim | 512 |
| Context window | 256 characters |
| Vocabulary | 94 characters (Spanish + punctuation) |
| Tokenization | Character-level |

## Training

- **Corpus**: 7.15M characters from all 7 volumes of Proust's masterwork
- **Best val loss**: 1.1739 (at epoch 20)
- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.01)
- **Schedule**: Linear warmup (500 steps) + cosine decay
- **Hardware**: NVIDIA T4 GPU (Google Colab)

## Usage

```python
import torch
from model_torch import Transformer
from tokenizer import CharTokenizer

# Load checkpoint
ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)

# Rebuild tokenizer
tokenizer = CharTokenizer()
tokenizer.char_to_idx = ckpt["vocab"]["char_to_idx"]
tokenizer.idx_to_char = {int(k): v for k, v in ckpt["vocab"]["idx_to_char"].items()}
tokenizer.vocab_size = ckpt["vocab"]["vocab_size"]

# Rebuild model
model = Transformer(**ckpt["model_config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Generate
prompt = "Mucho tiempo he estado"
prompt_ids = torch.from_numpy(tokenizer.encode(prompt)).long().unsqueeze(0)
output = model.generate(prompt_ids, max_new_tokens=300, temperature=0.8, top_k=40)
print(tokenizer.decode(output[0].numpy()))
```

## What This Project Demonstrates

- Manual implementation of scaled dot-product attention
- Multi-head attention with proper head splitting/concatenation
- Causal masking for autoregressive generation
- Sinusoidal positional encoding
- Layer normalization and residual connections
- Temperature and top-k sampling strategies

## Links

- [GitHub Repository](https://github.com/GonorAndres/proust-attention)

## Author

Andres Gonzalez Ortega -- UNAM Actuarial Science graduate exploring ML through first-principles implementation.
