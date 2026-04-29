FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Prevent git from trying to open a TTY for credentials during docker build
ENV GIT_TERMINAL_PROMPT=0

# System dependencies — only runtime libs needed; -dev headers not required for pre-built wheels
# apt-get clean first to free any cached archives before downloading
RUN apt-get clean \
    && apt-get -o Acquire::Check-Valid-Until=false \
               -o Acquire::Check-Date=false \
               -o Acquire::AllowInsecureRepositories=true \
               update \
    && apt-get install -y --allow-unauthenticated --no-install-recommends \
       ffmpeg \
       libsndfile1 \
       git \
       build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

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
RUN pip install torchcodec==0.2.1 --index-url https://download.pytorch.org/whl/cu124

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
