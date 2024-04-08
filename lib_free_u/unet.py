import functools
import math
import pathlib
import sys
from typing import Tuple, Union, Optional
from lib_free_u import global_state
from modules import scripts, shared
from modules.sd_hijack_unet import th
import torch


def patch():
    th.cat = functools.partial(free_u_cat_hijack, original_function=th.cat)

    cn_script_paths = [
        str(pathlib.Path(scripts.basedir()).parent.parent / "extensions-builtin" / "sd-webui-controlnet"),
        str(pathlib.Path(scripts.basedir()).parent / "sd-webui-controlnet"),
    ]
    sys.path[0:0] = cn_script_paths
    cn_status = "enabled"
    try:
        import scripts.hook as controlnet_hook
    except ImportError:
        cn_status = "disabled"
    else:
        controlnet_hook.th.cat = functools.partial(free_u_cat_hijack, original_function=controlnet_hook.th.cat)
    finally:
        for p in cn_script_paths:
            sys.path.remove(p)

        print("[sd-webui-freeu]", f"Controlnet support: *{cn_status}*")


def free_u_cat_hijack(hs, *args, original_function, **kwargs):
    if not global_state.instance.enable:
        return original_function(hs, *args, **kwargs)

    schedule_ratio = get_schedule_ratio()
    if schedule_ratio == 0:
        return original_function(hs, *args, **kwargs)

    try:
        h, h_skip = hs
        if list(kwargs.keys()) != ["dim"] or kwargs.get("dim", -1) != 1:
            return original_function(hs, *args, **kwargs)
    except ValueError:
        return original_function(hs, *args, **kwargs)

    dims = h.shape[1]
    try:
        index = [1280, 640, 320].index(dims)
        stage_info = global_state.instance.stage_infos[index]
    except ValueError:
        stage_info = None

    if stage_info is not None:
        redion_begin, region_end, region_inverted = ratio_to_region(stage_info.backbone_width, stage_info.backbone_offset, dims)
        mask = torch.arange(dims)
        mask = (redion_begin <= mask) & (mask <= region_end)
        if region_inverted:
            mask = ~mask

        h[:, mask] *= get_backbone_scale(
            h,
            backbone_factor=lerp(1, stage_info.backbone_factor, schedule_ratio),
        )
        h_skip = filter_skip(
            h_skip,
            threshold=stage_info.skip_cutoff,
            scale=lerp(1, stage_info.skip_factor, schedule_ratio),
            scale_high=lerp(1, stage_info.skip_high_end_factor, schedule_ratio),
        )

    return original_function([h, h_skip], *args, **kwargs)


def get_backbone_scale(h, backbone_factor):
    if global_state.instance.version == "1":
        return backbone_factor

    #if global_state.instance.version == "2":
    features_mean = h.mean(1, keepdim=True)
    batch_dims = h.shape[0]
    features_max, _ = torch.max(features_mean.view(batch_dims, -1), dim=-1, keepdim=True)
    features_min, _ = torch.min(features_mean.view(batch_dims, -1), dim=-1, keepdim=True)
    hidden_mean = (features_mean - features_min.unsqueeze(2).unsqueeze(3)) / (features_max - features_min).unsqueeze(2).unsqueeze(3)
    return 1 + (backbone_factor - 1) * hidden_mean


def filter_skip(x, threshold, scale, scale_high):
    if scale == 1 and scale_high == 1:
        return x

    fft_device = x.device
    if not is_gpu_complex_supported(x):
        fft_device = "cpu"

    # FFT
    x_freq = torch.fft.fftn(x.to(fft_device).float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.full((B, C, H, W), float(scale_high), device=fft_device)

    crow, ccol = H // 2, W // 2
    threshold_row = max(1, math.floor(crow * threshold))
    threshold_col = max(1, math.floor(ccol * threshold))
    mask[..., crow - threshold_row:crow + threshold_row, ccol - threshold_col:ccol + threshold_col] = scale
    x_freq *= mask

    # IFFT
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real.to(device=x.device, dtype=x.dtype)

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


def get_schedule_ratio():
    start_step = to_denoising_step(global_state.instance.start_ratio)
    stop_step = to_denoising_step(global_state.instance.stop_ratio)

    if start_step == stop_step:
        smooth_schedule_ratio = 0.0
    elif global_state.current_sampling_step < start_step:
        smooth_schedule_ratio = min(1.0, max(0.0, global_state.current_sampling_step / start_step))
    else:
        smooth_schedule_ratio = min(1.0, max(0.0, 1 + (global_state.current_sampling_step - start_step) / (start_step - stop_step)))

    flat_schedule_ratio = 1.0 if start_step <= global_state.current_sampling_step < stop_step else 0.0

    return lerp(flat_schedule_ratio, smooth_schedule_ratio, global_state.instance.transition_smoothness)


def to_denoising_step(number: Union[float, int], steps=None) -> int:
    if steps is None:
        steps = shared.state.sampling_steps

    if isinstance(number, float):
        return int(number * steps)

    return number


def lerp(a, b, r):
    return (1-r)*a + r*b


gpu_complex_support: Optional[bool] = None
def is_gpu_complex_supported(x):
    global gpu_complex_support

    if x.is_cpu:
        return True

    if gpu_complex_support is not None:
        return gpu_complex_support

    # catch known cases in advance
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    try:
        import torch_directml
    except ImportError:
        dml_available = False
    else:
        dml_available = torch_directml.is_available()

    gpu_complex_support = not (mps_available or dml_available)
    if gpu_complex_support:
        # try filter_skip fft to make sure it is viable on the gpu
        try:
            torch.fft.fftn(x.float(), dim=(-2, -1))
        except RuntimeError:
            gpu_complex_support = False

    return gpu_complex_support
