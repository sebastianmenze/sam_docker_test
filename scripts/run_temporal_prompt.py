"""
Separate a target sound using a temporal prompt (time range where the target occurs).

Usage:
    python run_temporal_prompt.py \
        --audio /workspace/audio_files/my_mix.wav \
        --start 2.5 \
        --end   5.0 \
        --model large \
        --out   /workspace/output/separated_temporal.wav

The time range [start, end] (in seconds) tells the model where in the audio
the target sound occurs so it can generalise to the full clip.
"""

import argparse
from pathlib import Path

import torch
import torchaudio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",  required=True)
    parser.add_argument("--start",  type=float, required=True, help="Start time of target sound (seconds)")
    parser.add_argument("--end",    type=float, required=True, help="End time of target sound (seconds)")
    parser.add_argument("--model",  default="large")
    parser.add_argument("--out",    default="/workspace/output/separated_temporal.wav")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MODEL_DIRS = {
        "small": "sam-audio-small", "base": "sam-audio-base", "large": "sam-audio-large",
        "small-tv": "sam-audio-small-tv", "base-tv": "sam-audio-base-tv", "large-tv": "sam-audio-large-tv",
    }
    model_dir = f"/workspace/models/{MODEL_DIRS[args.model]}"

    try:
        from sam_audio import SAMAudio
        model = SAMAudio.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    except ImportError as e:
        print(f"Could not import SAMAudio: {e}")
        return

    waveform, sr = torchaudio.load(args.audio)
    waveform = waveform.to(device)

    print(f"Temporal prompt: {args.start}s – {args.end}s")
    with torch.no_grad():
        separated = model.separate(
            waveform,
            temporal_prompt=(args.start, args.end),
            sample_rate=sr,
        )

    if isinstance(separated, dict):
        output_wav = separated.get("audio", next(iter(separated.values())))
    else:
        output_wav = separated

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.out, output_wav.cpu(), sr)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
