FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies — ffmpeg is required by pydub, torchcodec, av, and imageio-ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavfilter-dev \
    libavdevice-dev \
    libsndfile1-dev \
    libsox-dev \
    sox \
    git \
    wget \
    curl \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --upgrade pip setuptools wheel

# Match torchaudio / torchvision to the base PyTorch 2.6.0 + CUDA 12.4
RUN pip install \
    torchaudio==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Core scientific / utility packages
RUN pip install \
    einops \
    omegaconf \
    lightning \
    xarray \
    scikit-image \
    huggingface_hub

# Audio processing
RUN pip install \
    pydub \
    "laion-clap" \
    av \
    imageio \
    imageio-ffmpeg

# Video / codec processing
RUN pip install pytorchvideo
RUN pip install torchcodec==0.3.0 --index-url https://download.pytorch.org/whl/cu124

# OpenMMLab — provides mmengine used internally
RUN pip install mmengine

# SAM-audio git dependencies
RUN pip install "git+https://github.com/facebookresearch/ImageBind.git"
RUN pip install "git+https://github.com/facebookresearch/perception.git"

# Clone SAM-Audio and install (--no-deps since we already installed everything above)
RUN git clone https://github.com/facebookresearch/sam-audio.git /workspace/sam-audio
RUN pip install -e /workspace/sam-audio --no-deps

# Directories for user-supplied audio and outputs
RUN mkdir -p /workspace/audio_files /workspace/output /workspace/scripts

WORKDIR /workspace

CMD ["/bin/bash"]
