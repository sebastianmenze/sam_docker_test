"""
Download SAM-Audio model checkpoints from Hugging Face.

You must first:
  1. Request access at https://huggingface.co/facebook/sam-audio-large (and the other variants)
  2. Set your token: export HF_TOKEN=hf_...
     OR pass --token hf_... on the command line.

Usage:
    python download_models.py --model large
    python download_models.py --model base --token hf_yourtoken
    python download_models.py --model all
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download, login

MODELS = {
    "small":    "facebook/sam-audio-small",
    "base":     "facebook/sam-audio-base",
    "large":    "facebook/sam-audio-large",
    "small-tv": "facebook/sam-audio-small-tv",
    "base-tv":  "facebook/sam-audio-base-tv",
    "large-tv": "facebook/sam-audio-large-tv",
}

CACHE_DIR = Path("/workspace/models")


def download(model_key: str, token: str | None) -> None:
    repo_id = MODELS[model_key]
    dest = CACHE_DIR / model_key
    print(f"Downloading {repo_id} → {dest} ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        token=token,
    )
    print(f"  Done: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=[*MODELS.keys(), "all"],
        default="large",
        help="Which model variant to download (default: large)",
    )
    parser.add_argument("--token", default=None, help="Hugging Face token (overrides HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
    else:
        print("Warning: no HF_TOKEN set. Download will fail if the model is gated.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]
    for key in targets:
        download(key, token)

    print("\nAll requested models downloaded.")


if __name__ == "__main__":
    main()
