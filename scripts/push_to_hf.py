#!/usr/bin/env python3
"""
Push the trained Proust Attention Machine model to Hugging Face Hub.

Usage:
    python scripts/push_to_hf.py --repo-id <your-username>/proust-attention

Requires: huggingface-cli login (run first to authenticate)
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, upload_folder


def main():
    parser = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="HF repo ID, e.g. 'GonorAndres/proust-attention'")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to checkpoint file")
    parser.add_argument("--private", action="store_true",
                        help="Create as private repo")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    checkpoint_path = project_root / args.checkpoint

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # Create a temporary directory with the files to upload
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy checkpoint
        shutil.copy2(checkpoint_path, tmpdir / "best.pt")
        print(f"Copied checkpoint: {checkpoint_path} ({checkpoint_path.stat().st_size / 1e6:.1f} MB)")

        # Copy model source files (needed for inference)
        for src_file in ["model_torch.py", "model.py", "tokenizer.py", "generate.py"]:
            src_path = project_root / "src" / src_file
            if src_path.exists():
                shutil.copy2(src_path, tmpdir / src_file)
                print(f"Copied source: {src_file}")

        # Copy model card as README.md
        model_card = project_root / "scripts" / "hf_model_card.md"
        if model_card.exists():
            shutil.copy2(model_card, tmpdir / "README.md")
            print("Copied model card as README.md")

        # Create or get repo
        api = HfApi()
        api.create_repo(repo_id=args.repo_id, exist_ok=True,
                        private=args.private, repo_type="model")
        print(f"\nRepo ready: https://huggingface.co/{args.repo_id}")

        # Upload everything
        print("Uploading files...")
        upload_folder(
            folder_path=str(tmpdir),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Upload trained Proust Attention Machine (val_loss=1.1739)",
        )

        print(f"\nDone! Model is live at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
