"""
Download a SAM-Audio model checkpoint from HuggingFace into /workspace/models/.

Usage:
    python download_model.py --model small
    python download_model.py --model base
    python download_model.py --model large
"""

import argparse
from huggingface_hub import snapshot_download

MODEL_REPOS = {
    "small":    "facebook/sam-audio-small",
    "base":     "facebook/sam-audio-base",
    "large":    "facebook/sam-audio-large",
    "small-tv": "facebook/sam-audio-small-tv",
    "base-tv":  "facebook/sam-audio-base-tv",
    "large-tv": "facebook/sam-audio-large-tv",
}

MODEL_DIRS = {
    "small":    "sam-audio-small",
    "base":     "sam-audio-base",
    "large":    "sam-audio-large",
    "small-tv": "sam-audio-small-tv",
    "base-tv":  "sam-audio-base-tv",
    "large-tv": "sam-audio-large-tv",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", choices=list(MODEL_REPOS))
    parser.add_argument("--models-dir", default="/workspace/models")
    args = parser.parse_args()

    repo_id   = MODEL_REPOS[args.model]
    local_dir = f"{args.models_dir}/{MODEL_DIRS[args.model]}"

    print(f"Downloading {repo_id} → {local_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Done. Model saved to {local_dir}")

if __name__ == "__main__":
    main()
