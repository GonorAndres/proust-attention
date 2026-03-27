#!/usr/bin/env python3
"""
Proust Attention Machine -- Gradio Space
Generador de texto al estilo de Marcel Proust, entrenado desde cero.
"""

import torch
import numpy as np
import gradio as gr
from huggingface_hub import hf_hub_download
from pathlib import Path
import sys
import time

# ---------------------------------------------------------------------------
# Load model from HF Hub
# ---------------------------------------------------------------------------

MODEL_REPO = "GonorAndres/proust-attention"

# Download files from the model repo
checkpoint_path = hf_hub_download(repo_id=MODEL_REPO, filename="best.pt")
model_torch_path = hf_hub_download(repo_id=MODEL_REPO, filename="model_torch.py")
tokenizer_path = hf_hub_download(repo_id=MODEL_REPO, filename="tokenizer.py")

# Add the download directory to path so we can import the model
sys.path.insert(0, str(Path(model_torch_path).parent))

from model_torch import Transformer
from tokenizer import CharTokenizer

# Load checkpoint
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# Rebuild tokenizer from checkpoint vocab
tokenizer = CharTokenizer()
tokenizer.char_to_idx = ckpt["vocab"]["char_to_idx"]
tokenizer.idx_to_char = {int(k): v for k, v in ckpt["vocab"]["idx_to_char"].items()}
tokenizer.vocab_size = ckpt["vocab"]["vocab_size"]

# Rebuild model
model = Transformer(**ckpt["model_config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

train_loss = ckpt.get("loss", 0)
val_loss = ckpt.get("val_loss", 0)
epoch = ckpt.get("epoch", 0)

print(f"Modelo cargado: epoch {epoch}, val_loss {val_loss:.4f}")
print(f"Parametros: {model.count_parameters():,}")


# ---------------------------------------------------------------------------
# Generation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_text(prompt, max_chars, temperature, top_k):
    """Generate text from a prompt."""
    if not prompt or not prompt.strip():
        return "Escribe algo para comenzar..."

    prompt = prompt.strip()

    # Encode prompt
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.from_numpy(prompt_ids).long().unsqueeze(0)

    # Generate token by token
    generated = model.generate(
        input_ids,
        max_new_tokens=int(max_chars),
        temperature=temperature,
        top_k=int(top_k),
    )

    # Decode full output
    full_text = tokenizer.decode(generated[0].numpy())
    return full_text


# ---------------------------------------------------------------------------
# Example prompts -- iconic Proust openings and phrases
# ---------------------------------------------------------------------------

EXAMPLES = [
    ["Mucho tiempo he estado acostándome temprano.", 500, 0.8, 40],
    ["La memoria involuntaria", 400, 0.7, 40],
    ["En aquel momento, la puerta se abrió", 500, 0.85, 40],
    ["El amor es un mal incurable", 300, 0.9, 30],
    ["Por el camino de Swann", 500, 0.8, 40],
    ["La marquesa salió a las cinco", 400, 0.75, 40],
]


# ---------------------------------------------------------------------------
# Custom CSS -- literary, elegant theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* Overall dark literary theme */
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
    font-family: 'Georgia', 'Times New Roman', serif !important;
}

/* Header styling */
.header-block {
    text-align: center;
    padding: 2rem 1rem 1rem 1rem;
    border-bottom: 1px solid #d4a574;
    margin-bottom: 1.5rem;
}

.header-block h1 {
    font-size: 2.2rem;
    font-weight: 400;
    letter-spacing: 0.05em;
    color: #2c1810;
    margin-bottom: 0.3rem;
}

.header-block p {
    font-style: italic;
    color: #8b6914;
    font-size: 1.1rem;
    margin: 0.3rem 0;
}

/* Output text area -- literary feel */
.output-text textarea {
    font-family: 'Georgia', 'Times New Roman', serif !important;
    font-size: 1.05rem !important;
    line-height: 1.8 !important;
    color: #2c1810 !important;
    padding: 1.5rem !important;
}

/* Input text area */
.input-text textarea {
    font-family: 'Georgia', 'Times New Roman', serif !important;
    font-size: 1.05rem !important;
    line-height: 1.6 !important;
}

/* Stats footer */
.stats-block {
    text-align: center;
    padding: 1rem;
    margin-top: 1rem;
    border-top: 1px solid #d4a574;
    font-size: 0.85rem;
    color: #8b7355;
}

/* Generate button */
.generate-btn {
    background: #8b6914 !important;
    border: none !important;
    font-family: 'Georgia', serif !important;
    letter-spacing: 0.05em !important;
}
"""


# ---------------------------------------------------------------------------
# Build the Gradio interface
# ---------------------------------------------------------------------------

with gr.Blocks(css=CUSTOM_CSS, title="Proust Attention Machine") as demo:

    # Header
    gr.HTML("""
    <div class="header-block">
        <h1>La Maquina de Atencion Proustiana</h1>
        <p>Un transformer construido desde cero, entrenado con las 7 novelas de
        <em>En busca del tiempo perdido</em></p>
        <p style="font-size: 0.9rem; font-style: normal; color: #8b7355; margin-top: 0.8rem;">
            ~420,000 parametros &middot; 7.15M caracteres &middot; Nivel de caracter &middot;
            Atencion causal
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Texto inicial",
                placeholder="Escribe el inicio de una frase...",
                lines=3,
                elem_classes=["input-text"],
            )

            with gr.Row():
                max_chars = gr.Slider(
                    minimum=50, maximum=1000, value=500, step=50,
                    label="Caracteres a generar",
                )
                temperature = gr.Slider(
                    minimum=0.3, maximum=1.5, value=0.8, step=0.05,
                    label="Temperatura",
                    info="Baja = predecible, Alta = creativo",
                )

            top_k = gr.Slider(
                minimum=5, maximum=94, value=40, step=5,
                label="Top-K",
                info="Cuantos caracteres considera en cada paso",
            )

            generate_btn = gr.Button(
                "Generar texto",
                variant="primary",
                elem_classes=["generate-btn"],
            )

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Texto generado",
                lines=16,
                show_copy_button=True,
                elem_classes=["output-text"],
            )

    # Examples
    gr.Examples(
        examples=EXAMPLES,
        inputs=[prompt_input, max_chars, temperature, top_k],
        outputs=output_text,
        fn=generate_text,
        cache_examples=False,
        label="Frases de ejemplo (haz clic para probar)",
    )

    # Footer with model info
    gr.HTML(f"""
    <div class="stats-block">
        Entrenado en Google Colab (T4 GPU) &middot;
        Epoca {epoch} &middot;
        Perdida de validacion: {val_loss:.4f} &middot;
        Vocabulario: {tokenizer.vocab_size} caracteres<br>
        <a href="https://github.com/GonorAndres/proust-attention"
           style="color: #8b6914; text-decoration: none;"
           target="_blank">
            Codigo fuente en GitHub
        </a>
        &middot;
        Construido por Andres Gonzalez Ortega
    </div>
    """)

    # Wire up the button
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, max_chars, temperature, top_k],
        outputs=output_text,
    )

    # Also generate on Enter
    prompt_input.submit(
        fn=generate_text,
        inputs=[prompt_input, max_chars, temperature, top_k],
        outputs=output_text,
    )


if __name__ == "__main__":
    demo.launch()
