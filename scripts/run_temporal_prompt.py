"""
Separate a target sound using a temporal prompt (time range where the target occurs).

Usage:
    python run_temporal_prompt.py \
        --audio /workspace/audio_files/my_mix.wav \
        --start 2.5 \
        --end   5.0 \
        --model large \
        --out   /workspace/output/separated_temporal.wav

The time range [start, end] (in seconds) marks where the target sound occurs.
That segment is extracted and used as an audio-query prompt so the model can
separate the same sound from the full clip.
"""

import argparse
import tempfile
from pathlib import Path

import torch
import torchaudio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",  required=True)
    parser.add_argument("--start",  type=float, required=True, help="Start time of target sound (seconds)")
    parser.add_argument("--end",    type=float, required=True, help="End time of target sound (seconds)")
    parser.add_argument("--model",  default="small",
                        choices=["small", "base", "large", "small-tv", "base-tv", "large-tv"])
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
    print(f"Loading model from {model_dir} ...")

    from sam_audio import SAMAudio, SAMAudioProcessor
    model = SAMAudio.from_pretrained(model_dir).to(device).eval()
    processor = SAMAudioProcessor.from_pretrained(model_dir)

    # Extract the target segment and save to a temp file to use as audio query
    waveform, sr = torchaudio.load(args.audio)
    start_frame = int(args.start * sr)
    end_frame   = int(args.end   * sr)
    segment = waveform[:, start_frame:end_frame]

    print(f"Audio:   {args.audio}")
    print(f"Segment: {args.start}s – {args.end}s  ({segment.shape[-1]} samples @ {sr} Hz)")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    torchaudio.save(tmp_path, segment, sr)

    # Build batch: full audio as the mixture, extracted segment as the audio query
    inputs = processor(audios=[args.audio], audio_queries=[tmp_path]).to(device)

    with torch.inference_mode():
        result = model.separate(inputs)

    separated = result.target[0].cpu()
    if separated.dim() == 1:
        separated = separated.unsqueeze(0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.out, separated, processor.audio_sampling_rate)
    print(f"Saved:   {args.out}")

    Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
