"""Quick sanity-check: verify all key imports work and the GPU is visible."""

import sys

print(f"Python {sys.version}")

import torch
print(f"PyTorch  {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

import torchaudio
print(f"torchaudio {torchaudio.__version__}")

import torchvision
print(f"torchvision {torchvision.__version__}")

try:
    import sam_audio
    print(f"sam_audio loaded OK — location: {sam_audio.__file__}")
except Exception as e:
    print(f"sam_audio import failed: {e}")

try:
    from imagebind.models import imagebind_model
    print("ImageBind loaded OK")
except Exception as e:
    print(f"ImageBind import failed: {e}")

try:
    from core.audio_visual_encoder import PEAudioFrame
    print("perception_models (core) loaded OK")
except Exception as e:
    print(f"perception_models (core) import failed: {e}")

print("\nAll checks done.")
