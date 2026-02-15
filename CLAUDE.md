# CLAUDE.md — Proust Attention Machine

## Project Overview

Educational project building a transformer-based text generator from scratch, trained on Proust's "En busca del tiempo perdido" (Spanish translation). Goal: deep understanding of attention mechanisms through manual implementation, then scaling to full training.

**Philosophy**: Understanding over abstraction. NumPy first to see the math, PyTorch second for training power.

**Working mode**: Always conceptual clarity FIRST, code SECOND. Before implementing any component, recap the fundamental concepts behind it. Abstract thought and deep clarification are the priority -- never jump into code without understanding the "why." This is a learning project; the outputs are not the objective, the understanding is.

## Architecture Decisions

- **Model size**: ~300k parameters (trainable on Colab free tier)
- **Blocks**: 2 transformer layers
- **Heads**: 2 attention heads
- **Embedding dim**: 64
- **Context window**: 256 characters (Proust's long sentences need room)
- **Tokenization**: Character-level (simple, no BPE complexity)

## Project Structure

```
proust-attention/
├── CLAUDE.md
├── README.md
├── data/
│   ├── raw/                    # Original downloaded texts
│   ├── processed/              # Cleaned, concatenated corpus
│   └── download_corpus.py      # Gutenberg/Archive scraper
├── notebooks/
│   ├── 01_numpy_attention.ipynb      # Phase 1: Understanding
│   ├── 02_pytorch_training.ipynb     # Phase 2: Training
│   └── 03_visualization.ipynb        # Phase 3: Demo
├── src/
│   ├── __init__.py
│   ├── attention.py            # Core attention (NumPy + PyTorch versions)
│   ├── model.py                # Full transformer architecture
│   ├── tokenizer.py            # Character-level tokenizer
│   ├── train.py                # Training loop
│   └── generate.py             # Text generation + sampling
├── viz/
│   ├── attention_heatmap.py
│   └── embedding_viz.py
├── checkpoints/                # Saved model weights
├── requirements.txt
└── demo/
    └── streamlit_app.py        # Interactive demo
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
- [ ] Train on full corpus (Colab GPU)
- [ ] Checkpoint saving/loading

### Phase 3: Visualization + Demo
- [ ] Attention heatmaps per head
- [ ] Embedding space visualization (t-SNE)
- [ ] Interactive generation with attention overlay
- [ ] Streamlit deployment

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
pip install numpy torch matplotlib seaborn streamlit

# Download data
python data/download_corpus.py

# Training (after Phase 2)
python src/train.py --epochs 50 --batch_size 32

# Demo
streamlit run demo/streamlit_app.py
```

## Session Log

### 2026-02-07
- Created bilingual blog post (ES + EN) for the portfolio site covering the Proust Attention Machine project
- Rewrote post in editorial/personal essay style with attention heatmap visualizations

### 2026-02-15
- Fixed blog language switcher so the Proust post works correctly when toggling ES/EN (slug prefix stripping in portfolio repo)

## Notes for Claude Code

When helping with this project:
1. Always include shape annotations in comments
2. Keep NumPy and PyTorch implementations parallel
3. Prefer explicit over clever—this is for learning
4. When stuck on math, break into smallest possible steps
5. Test each component in isolation before composing
