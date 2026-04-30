"""
Separate a target sound from a mixture using a text prompt.

Usage:
    python run_text_prompt.py \
        --audio  /workspace/audio_files/my_mix.wav \
        --prompt "whale call" \
        --model  large \
        --out    /workspace/output/separated.wav
"""

import argparse
from pathlib import Path

import torch
import torchaudio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",  required=True, help="Input audio file")
    parser.add_argument("--prompt", required=True, help="Text description of the target sound")
    parser.add_argument("--model",  default="large",
                        choices=["small", "base", "large", "small-tv", "base-tv", "large-tv"])
    parser.add_argument("--out",    default="/workspace/output/separated.wav")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MODEL_DIRS = {
        "small": "sam-audio-small", "base": "sam-audio-base", "large": "sam-audio-large",
        "small-tv": "sam-audio-small-tv", "base-tv": "sam-audio-base-tv", "large-tv": "sam-audio-large-tv",
    }
    model_dir = f"/workspace/models/{MODEL_DIRS[args.model]}"
    print(f"Loading model from {model_dir} ...")

    from sam_audio import SAMAudio, SAMAudioProcessor
    model = SAMAudio.from_pretrained(model_dir).to(device).eval()
    processor = SAMAudioProcessor.from_pretrained(model_dir)

    print(f"Audio:  {args.audio}")
    print(f"Prompt: {args.prompt}")
    inputs = processor(audios=[args.audio], descriptions=[args.prompt]).to(device)

    with torch.inference_mode():
        result = model.separate(inputs)

    separated = result.target[0].cpu()
    if separated.dim() == 1:
        separated = separated.unsqueeze(0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.out, separated, processor.audio_sampling_rate)
    print(f"Saved:  {args.out}")


if __name__ == "__main__":
    main()
