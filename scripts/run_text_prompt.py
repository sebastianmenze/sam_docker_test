"""
Separate a target sound from a mixture using a text prompt.

Usage:
    python run_text_prompt.py \
        --audio  /workspace/audio_files/my_mix.wav \
        --prompt "dog barking" \
        --model  large \
        --out    /workspace/output/separated.wav

The model directory is expected at /workspace/models/<model-size>/.
Run download_models.py first if you haven't already.
"""

import argparse
from pathlib import Path

import torch
import torchaudio


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(path)
    return waveform, sr


def save_audio(waveform: torch.Tensor, sr: int, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(path, waveform.cpu(), sr)
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",  required=True, help="Input audio file (.wav / .mp3 / etc.)")
    parser.add_argument("--prompt", required=True, help="Text description of the target sound")
    parser.add_argument("--model",  default="large", choices=["small", "base", "large",
                                                               "small-tv", "base-tv", "large-tv"])
    parser.add_argument("--out",    default="/workspace/output/separated.wav")
    parser.add_argument("--device", default=None, help="cuda / cpu (auto-detected if omitted)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_dir = f"/workspace/models/{args.model}"

    # --- Load SAM-Audio model ---
    try:
        from sam_audio import SAMAudio  # adjust to actual API if needed
        model = SAMAudio.from_pretrained(model_dir).to(device).eval()
    except ImportError as e:
        print(f"Could not import SAMAudio: {e}")
        print("Check the sam-audio package is installed and try test_import.py first.")
        return

    # --- Load audio ---
    print(f"Loading audio: {args.audio}")
    waveform, sr = load_audio(args.audio)
    waveform = waveform.to(device)

    # --- Run inference ---
    print(f"Separating sound with prompt: '{args.prompt}'")
    with torch.no_grad():
        separated = model.separate(waveform, prompt=args.prompt, sample_rate=sr)

    # separated may be a dict or tensor — handle both
    if isinstance(separated, dict):
        output_wav = separated.get("audio", separated.get("output", next(iter(separated.values()))))
    else:
        output_wav = separated

    save_audio(output_wav, sr, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
