# CLAUDE.md — Proust Attention Machine

## Project Overview

Educational project building a transformer-based text generator from scratch, trained on Proust's "En busca del tiempo perdido" (Spanish translation). Goal: deep understanding of attention mechanisms through manual implementation, then scaling to full training.

**Philosophy**: Understanding over abstraction. NumPy first to see the math, PyTorch second for training power.

**Working mode**: Always conceptual clarity FIRST, code SECOND. Before implementing any component, recap the fundamental concepts behind it. Abstract thought and deep clarification are the priority -- never jump into code without understanding the "why." This is a learning project; the outputs are not the objective, the understanding is.

## Architecture Decisions

- **Model size**: ~420k parameters (trained on Colab free tier T4 GPU)
- **Blocks**: 2 transformer layers
- **Heads**: 2 attention heads
- **Embedding dim**: 128 (d_ff: 512)
- **Context window**: 256 characters (Proust's long sentences need room)
- **Tokenization**: Character-level (94-char vocab, Spanish + punctuation)
- **Training stride**: 128 (50% window overlap, 128x faster than stride=1)

## Project Structure

```
proust-attention/
├── CLAUDE.md
├── README.md
├── data/
│   ├── raw/                    # Original .mobi ebooks (7 volumes)
│   ├── processed/              # Cleaned corpus + vocab.json
│   └── download_corpus.py      # Mobi extraction + cleaning pipeline
├── notebooks/
│   ├── 01_numpy_attention.ipynb      # Phase 1: Understanding
│   ├── 02_pytorch_training.ipynb     # Phase 2: Training
│   ├── 03_visualization.ipynb        # Phase 3: Demo
│   └── colab_training.ipynb          # Colab GPU training notebook
├── src/
│   ├── __init__.py
│   ├── attention.py            # Core attention (NumPy + PyTorch versions)
│   ├── model.py                # Full transformer architecture (NumPy)
│   ├── model_torch.py          # Full transformer architecture (PyTorch)
│   ├── dataset.py              # PyTorch Dataset + DataLoader (with stride)
│   ├── tokenizer.py            # Character-level tokenizer
│   ├── train.py                # Training loop (AdamW, cosine LR, val tracking)
│   └── generate.py             # Text generation + sampling
├── space/
│   ├── app.py                  # Gradio demo for HF Spaces
│   ├── requirements.txt
│   └── README.md               # HF Space metadata
├── scripts/
│   ├── train_gcp.sh            # One-shot GCP GPU training
│   ├── push_to_hf.py           # Upload model to HF Hub
│   └── hf_model_card.md        # HF model card template
├── docs/
│   └── roadmap.md              # Detailed improvement roadmap
├── checkpoints/                # Saved model weights (best.pt)
└── requirements.txt
```

## Development Phases

### Phase 1: NumPy Implementation (Understanding)
- [x] Data download + cleaning
- [x] Character vocabulary builder
- [x] Embedding layer with positional encoding
- [x] Single-head attention forward pass
- [x] Multi-head attention
- [x] Feedforward sublayer
- [x] Layer normalization
- [x] Full transformer block forward pass
- [ ] Verify against PyTorch equivalents

### Phase 2: PyTorch Training (Power)
- [x] Port NumPy code to PyTorch (keep structure identical)
- [x] Implement training loop with AdamW
- [x] Add gradient clipping, learning rate scheduling
- [x] Train on full corpus (Colab T4 GPU) — best val_loss=1.1739 at epoch 20
- [x] Checkpoint saving/loading (best.pt by val_loss)
- [x] Configurable stride for training efficiency (128x speedup)

### Phase 3: Visualization + Demo
- [ ] Attention heatmaps per head
- [ ] Embedding space visualization (t-SNE/UMAP)
- [ ] Interactive generation with attention overlay
- [x] Hugging Face deployment (model repo + Gradio Space)

### Phase 4: Model Improvements (see docs/roadmap.md for details)
- [ ] Verify NumPy vs PyTorch equivalence
- [ ] BPE tokenization (character -> subword)
- [ ] Scaling experiments (depth vs width)
- [ ] Dropout/regularization tuning
- [ ] KV-cache for fast inference
- [ ] Rotary Positional Embeddings (RoPE)

## Code Conventions

- **Shape annotations everywhere**: `# (batch, seq, d_model)`
- **No magic numbers**: Constants in config dict at top of files
- **NumPy/PyTorch parity**: Same function names, same logic
- **Verbose variable names**: `query_projected` not `q`

## Data Source

**Spanish corpus**: "En busca del tiempo perdido"
- Check Gutenberg ES: https://www.gutenberg.org
- Internet Archive: search "Proust Spanish"
- Biblioteca Virtual Miguel de Cervantes
- Epublibre (public domain)

All 7 volumes:
1. Por el camino de Swann
2. A la sombra de las muchachas en flor
3. El mundo de Guermantes
4. Sodoma y Gomorra
5. La prisionera
6. La fugitiva (Albertine desaparecida)
7. El tiempo recobrado

## Key Implementation Notes

### Attention Formula
```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

### Causal Masking
For autoregressive generation, mask future tokens:
```python
mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
```

### Shape Flow Through Attention
```
Input:          (batch, seq, d_model)
Q, K, V proj:   (batch, seq, d_model) → (batch, seq, d_model)
Reshape heads:  (batch, seq, d_model) → (batch, heads, seq, d_k)
Attention:      (batch, heads, seq, d_k) @ (batch, heads, d_k, seq) → (batch, heads, seq, seq)
Output:         (batch, heads, seq, d_k) → (batch, seq, d_model)
```

## Testing Strategy

- Compare NumPy outputs with PyTorch on identical inputs
- Use seed for reproducibility: `np.random.seed(42)`
- Tiny tensors first: batch=1, seq=4, d_model=8

## Resources

- Karpathy "Let's build GPT": https://www.youtube.com/watch?v=kCc8FmEb1nY
- Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- Attention paper (sections 3.2, 3.3): https://arxiv.org/abs/1706.03762

## Commands

```bash
# Setup
pip install numpy torch matplotlib seaborn streamlit tqdm huggingface_hub

# Download/regenerate corpus
python data/download_corpus.py --force

# Training (stride=128 default, ~30 min on T4)
python src/train.py --epochs 50 --batch-size 64 --stride 128
python src/train.py --resume checkpoints/best.pt --epochs 20  # resume

# Generate text
python src/generate.py --checkpoint checkpoints/best.pt --prompt "Mucho tiempo"

# Push model to Hugging Face
python scripts/push_to_hf.py --repo-id GonorAndres/proust-attention

# GCP one-shot training
bash scripts/train_gcp.sh
```

## Session Log

### 2026-02-07
- Created bilingual blog post (ES + EN) for the portfolio site covering the Proust Attention Machine project
- Rewrote post in editorial/personal essay style with attention heatmap visualizations

### 2026-02-15
- Fixed blog language switcher so the Proust post works correctly when toggling ES/EN (slug prefix stripping in portfolio repo)

### 2026-03-27
- Added configurable stride to dataset (128x training speedup, stride=1 was the Colab bottleneck)
- Trained model on Colab T4: val_loss=1.1739 at epoch 20 (~420k params)
- Uploaded trained model to HF Hub: GonorAndres/proust-attention
- Built Gradio Space demo (Spanish UI): huggingface.co/spaces/GonorAndres/proust-attention
- Added GCP training script, HF push script, model card
- Created Phase 4 roadmap (docs/roadmap.md)

## Notes for Claude Code

When helping with this project:
1. Always include shape annotations in comments
2. Keep NumPy and PyTorch implementations parallel
3. Prefer explicit over clever—this is for learning
4. When stuck on math, break into smallest possible steps
5. Test each component in isolation before composing
