FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Prevent git from trying to open a TTY for credentials during docker build
ENV GIT_TERMINAL_PROMPT=0

# Minimal system deps — ffmpeg comes from imageio-ffmpeg (bundled static binary),
# av/torchcodec ship their own. build-essential is already in the devel base image.
RUN apt-get clean \
    && apt-get -o Acquire::Check-Valid-Until=false \
               -o Acquire::Check-Date=false \
               -o Acquire::AllowInsecureRepositories=true \
               update \
    && apt-get install -y --allow-unauthenticated --no-install-recommends \
       git \
       libsndfile1 \
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

# Pre-download the imageio-ffmpeg static binary so it's available at runtime
RUN python -c "import imageio_ffmpeg; print('ffmpeg binary:', imageio_ffmpeg.get_ffmpeg_exe())"

# Directories for user-supplied audio and outputs
RUN mkdir -p /workspace/audio_files /workspace/output /workspace/scripts

WORKDIR /workspace

CMD ["/bin/bash"]
