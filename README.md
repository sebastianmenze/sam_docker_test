# SAM-Audio Docker Test Environment

Docker setup for testing [SAM-Audio](https://github.com/facebookresearch/sam-audio) with your own audio files on a CUDA-enabled server.

Tested against: NVIDIA driver 590.x / CUDA 13.1 host (container uses CUDA 12.4 runtime — fully compatible).

---

## Prerequisites

On the host server:
```bash
# nvidia-container-toolkit must be installed for GPU passthrough
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## Quick Start

### 1. Configure your environment
```bash
cp .env.example .env
# Edit .env: set HF_TOKEN and optionally NVIDIA_VISIBLE_DEVICES
```

### 2. Add your audio files
```bash
mkdir -p audio_files
cp /path/to/your/*.wav audio_files/
```

### 3. Build the image (takes ~10–15 min first time)
```bash
docker compose build
```

### 4. Start an interactive shell
```bash
docker compose run --rm sam-audio
```

---

## Inside the Container

### Verify everything works
```bash
python /workspace/scripts/test_import.py
```

### Download model checkpoints (requires HF access approval)
```bash
# Request access first: https://huggingface.co/facebook/sam-audio-large
python /workspace/scripts/download_models.py --model large
```

### Separate a sound with a text prompt
```bash
python /workspace/scripts/run_text_prompt.py \
    --audio  /workspace/audio_files/my_mix.wav \
    --prompt "boat engine" \
    --model  large \
    --out    /workspace/output/boat_engine.wav
```

### Separate using a time range (temporal prompt)
```bash
python /workspace/scripts/run_temporal_prompt.py \
    --audio  /workspace/audio_files/my_mix.wav \
    --start  3.0 \
    --end    7.5 \
    --model  large \
    --out    /workspace/output/temporal_result.wav
```

---

## GPU Selection

The server has three GPUs:
- GPU 0: RTX 2080 Ti (11 GB)
- GPU 1: RTX A6000 (49 GB) ← recommended for large model
- GPU 2: RTX 2080 Ti (11 GB)

To use only the A6000 (GPU 1):
```bash
NVIDIA_VISIBLE_DEVICES=1 docker compose run --rm sam-audio
```

Or set `NVIDIA_VISIBLE_DEVICES=1` in your `.env`.

Inside the container, pin the device explicitly:
```bash
python scripts/run_text_prompt.py --audio ... --prompt ... --device cuda:0
# (inside the container the single exposed GPU is always cuda:0)
```

---

## Directory Layout

```
.
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── audio_files/        # ← put your .wav / .mp3 files here (git-ignored)
├── output/             # ← separated audio written here (git-ignored)
├── models/             # ← HF checkpoints land here (git-ignored)
└── scripts/
    ├── test_import.py
    ├── download_models.py
    ├── run_text_prompt.py
    └── run_temporal_prompt.py
```

---

## Notes

- The `scripts/` directory is volume-mounted, so you can edit scripts on the host and run them immediately inside the container without rebuilding.
- SAM-Audio model checkpoints are gated on Hugging Face — you need to request access before downloading.
- The `laion-clap` and `ImageBind` re-rankers are optional but improve separation quality at the cost of extra latency.
