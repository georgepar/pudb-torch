from typing import Any

try:
    import torch

    HAVE_TORCH = 1
except:
    HAVE_TORCH = 0


from pudb.var_view import default_stringifier, type_stringifier


def shortened_sfier(value: Any, max_chars: int = -1) -> str:
    out: str = default_stringifier(value)

    if max_chars > 0 and len(out) > max_chars:
        out = type_stringifier(value)

    return out


def pudb_stringifier(value: Any) -> str:
    if not HAVE_TORCH:
        return shortened_sfier(value)

    if isinstance(value, torch.nn.Module):
        device: str = str(next(value.parameters()).device)
        params: int = sum([p.numel() for p in value.parameters() if p.requires_grad])
        rep: str = value.__repr__() if len(value.__repr__()) < 55 else type(value).__name__

        return "{}[{}] Params: {}".format(rep, device, params)
    elif isinstance(value, torch.Tensor):
        return "{}[{}][{}] {}".format(
            type(value).__name__,
            str(value.dtype).replace("torch.", ""),
            str(value.device),
            str(list(value.shape)),
        )
    else:
        return shortened_sfier(value)
