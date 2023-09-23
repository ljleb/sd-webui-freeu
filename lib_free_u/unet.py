import functools
import math
import pathlib
import sys
from typing import Tuple
from lib_free_u import global_state
from modules import scripts
from modules.sd_hijack_unet import th
import torch


def free_u_cat_hijack(hs, *args, original_function, **kwargs):
    if not global_state.enabled:
        return original_function(hs, *args, **kwargs)

    try:
        h, h_skip = hs
        if list(kwargs.keys()) != ["dim"] or kwargs.get("dim", -1) != 1:
            return original_function(hs, *args, **kwargs)
    except ValueError:
        return original_function(hs, *args, **kwargs)

    dims = h.shape[1]
    try:
        index = [1280, 640].index(dims)
    except ValueError:
        index = None

    if index is not None:
        redion_begin, region_end, region_inverted = ratio_to_region(global_state.backbone_widths[index], global_state.backbone_offsets[index], dims)
        mask = torch.arange(dims)
        mask = (redion_begin <= mask) & (mask <= region_end)
        if region_inverted:
            mask = ~mask

        h[:, mask] *= global_state.backbone_factors[index]
        h_skip = filter_skip(h_skip, threshold=global_state.skip_thresholds[index], scale=global_state.skip_factors[index], scale_high=global_state.high_skip_factors[index])

    return original_function([h, h_skip], *args, **kwargs)


def filter_skip(x, threshold, scale, scale_high):
    # FFT
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.full((B, C, H, W), scale_high, device=x.device, dtype=x.dtype)

    crow, ccol = H // 2, W // 2
    threshold_row = max(1, math.floor(crow * threshold))
    threshold_col = max(1, math.floor(ccol * threshold))
    mask[..., crow - threshold_row:crow + threshold_row, ccol - threshold_col:ccol + threshold_col] = scale
    x_freq *= mask

    # IFFT
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real.to(dtype=x.dtype)

    return x_filtered


def ratio_to_region(width: float, offset: float, n: int) -> Tuple[int, int, bool]:
    if width < 0:
        offset += width
        width = -width
    width = min(width, 1)

    if offset < 0:
        offset = 1 + offset - int(offset)
    offset = math.fmod(offset, 1.0)

    if width + offset <= 1:
        inverted = False
        start = offset * n
        end = (width + offset) * n
    else:
        inverted = True
        start = (width + offset - 1) * n
        end = offset * n

    return round(start), round(end), inverted


def patch():
    cat_hijack = functools.partial(free_u_cat_hijack, original_function=th.cat)
    th.cat = cat_hijack

    cn_script_paths = [
        str(pathlib.Path(scripts.basedir()).parent / "sd-webui-controlnet"),
        str(pathlib.Path(scripts.basedir()).parent.parent / "extensions-builtin" / "sd-webui-controlnet"),
    ]
    sys.path[1:1] = cn_script_paths
    cn_status = "enabled"
    try:
        import scripts.hook as controlnet_hook
        controlnet_hook.th.cat = cat_hijack
    except ImportError:
        cn_status = "disabled"
    finally:
        for p in cn_script_paths:
            sys.path.remove(p)

        print("[sd-webui-freeu]", f"controlnet: *{cn_status}*")
