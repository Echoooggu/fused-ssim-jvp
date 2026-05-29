import torch
import torch.autograd.forward_ad as fwAD

def has_tangent(x):
    if isinstance(x, float):
        return False
    elif isinstance(x, list):
        return any(has_tangent(xi) for xi in x)
    return fwAD.unpack_dual(x).tangent is not None

def get_tangent(x):
    if isinstance(x, float):
        return 0.0
    elif isinstance(x, list):
        return [get_tangent(xi) for xi in x]
    elif has_tangent(x):
        return fwAD.unpack_dual(x).tangent
    elif isinstance(x, torch.Tensor):
        return torch.zeros_like(x)
    else:
        raise ValueError(f"Unsupported type for tangent extraction: {type(x)}")