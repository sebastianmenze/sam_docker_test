# Registers torchvision.transforms.functional_tensor (removed in 0.17+) as a
# compatibility shim so ImageBind and other old code can still import it.
try:
    import torchvision.transforms.functional as _ft
    import sys
    sys.modules.setdefault("torchvision.transforms.functional_tensor", _ft)
except ImportError:
    pass
