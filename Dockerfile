# nvidia/cuda base — CUDA 12.8 supports Blackwell (sm_120) on Windows and the Linux server
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GIT_TERMINAL_PROMPT=0

# Python 3.11 + minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    libsndfile1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --upgrade pip setuptools wheel

# PyTorch — no version pin, picks latest cu128 build (2.7+ for Blackwell sm_120)
RUN pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128

# Core deps
RUN pip install \
    einops \
    omegaconf \
    lightning \
    xarray \
    scikit-image \
    huggingface_hub

# Audio/video
RUN pip install \
    pydub \
    "laion-clap" \
    av \
    imageio \
    imageio-ffmpeg

# pytorchvideo + mmengine
RUN pip install pytorchvideo mmengine

# torchcodec: try cu128 first, fall back to cu124 if no cu128 wheel yet
RUN pip install torchcodec --index-url https://download.pytorch.org/whl/cu128 || \
    pip install torchcodec==0.2.1 --index-url https://download.pytorch.org/whl/cu124

# SAM-audio git dependencies
RUN pip install "git+https://github.com/facebookresearch/ImageBind.git"

# facebookresearch/perception is a private repo — needs a GitHub token.
# Token is injected via Docker build secret (never stored in image layers).
# Before building: set GITHUB_TOKEN in .env, then run docker compose build.
RUN --mount=type=secret,id=github_token \
    TOKEN=$(cat /run/secrets/github_token | tr -d '[:space:]') && \
    git clone "https://${TOKEN}@github.com/facebookresearch/perception.git" /tmp/perception && \
    pip install /tmp/perception && \
    rm -rf /tmp/perception

# Install xformers compatible with the installed torch version
RUN pip install xformers --index-url https://download.pytorch.org/whl/cu128

# Clone SAM-Audio and install
RUN git clone https://github.com/facebookresearch/sam-audio.git /workspace/sam-audio
RUN pip install -e /workspace/sam-audio --no-deps

# Pre-download imageio-ffmpeg static binary
RUN python -c "import imageio_ffmpeg; print('ffmpeg binary:', imageio_ffmpeg.get_ffmpeg_exe())"

RUN mkdir -p /workspace/audio_files /workspace/output /workspace/scripts

WORKDIR /workspace

CMD ["/bin/bash"]
