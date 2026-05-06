"""
Separate a target sound using a temporal (span) prompt.

The anchor ["+", start, end] tells the model where the target sound occurs
so it can separate that sound from the full mixture.

Usage:
    python run_temporal_prompt.py \
        --audio  /workspace/audio_files/my_mix.wav \
        --start  5.0 \
        --end    7.0 \
        --prompt "whale call" \
        --model  small \
        --out    /workspace/output/separated_temporal.wav

    # Limit to first 30 s to reduce memory on constrained machines:
    python run_temporal_prompt.py ... --max-duration 30
"""

import argparse
import tempfile
from pathlib import Path

import torch
import torchaudio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",        required=True)
    parser.add_argument("--start",        type=float, required=True,
                        help="Start of target sound (seconds)")
    parser.add_argument("--end",          type=float, required=True,
                        help="End of target sound (seconds)")
    parser.add_argument("--prompt",       default="",
                        help="Optional text hint about the target sound")
    parser.add_argument("--model",        default="small",
                        choices=["small", "base", "large", "small-tv", "base-tv", "large-tv"])
    parser.add_argument("--out",          default="/workspace/output/separated_temporal.wav")
    parser.add_argument("--device",       default=None)
    parser.add_argument("--max-duration", type=float, default=None,
                        help="Clip audio to this many seconds before processing (reduces memory)")
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
    model = SAMAudio.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    processor = SAMAudioProcessor.from_pretrained(model_dir)

    audio_path = args.audio
    tmp_path = None

    if args.max_duration is not None:
        waveform, sr = torchaudio.load(args.audio)
        max_frames = int(args.max_duration * sr)
        if waveform.shape[-1] > max_frames:
            waveform = waveform[:, :max_frames]
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            torchaudio.save(tmp_path, waveform, sr)
            audio_path = tmp_path
            print(f"Clipped to {args.max_duration}s")

    print(f"Audio:   {audio_path}")
    print(f"Anchor:  +[{args.start}s – {args.end}s]  prompt='{args.prompt}'")

    inputs = processor(
        audios=[audio_path],
        descriptions=[args.prompt],
        anchors=[[["+", args.start, args.end]]],
    ).to(device)

    with torch.inference_mode():
        result = model.separate(inputs)

    separated = result.target[0].cpu()
    if separated.dim() == 1:
        separated = separated.unsqueeze(0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.out, separated, processor.audio_sampling_rate)
    print(f"Saved:   {args.out}")

    if tmp_path:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
