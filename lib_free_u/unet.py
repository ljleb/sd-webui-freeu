import math
from typing import Tuple
from lib_free_u import global_state
from ldm.modules.diffusionmodules import openaimodel
try:
    from sgm.modules.diffusionmodules import openaimodel as openaimodel_sdxl
except ImportError:
    openaimodel_sdxl = None


from modules.sd_hijack_unet import th as torch


class UNetModel(openaimodel.UNetModel):
    """
    copied from repositories.stable-diffusion-stability-ai.ldm.modules.diffusionmodules.openaimodel
    """

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if not global_state.enabled:
            return OriginalUNetModel.forward(self, x, timesteps, context, y, **kwargs)

        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = openaimodel.timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = free_u_cat(h, hs.pop())
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


if openaimodel_sdxl:
    class SdxlUNetModel(openaimodel_sdxl.UNetModel):
        """
        copied from repositories.generative-models.sgm.modules.diffusionmodules.openaimodel
        """

        def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
            """
            Apply the model to an input batch.
            :param x: an [N x C x ...] Tensor of inputs.
            :param timesteps: a 1-D batch of timesteps.
            :param context: conditioning plugged in via crossattn
            :param y: an [N] Tensor of labels, if class-conditional.
            :return: an [N x C x ...] Tensor of outputs.
            """
            if not global_state.enabled:
                return OriginalSdxlUNetModel.forward(self, x, timesteps, context, y, **kwargs)

            assert (y is not None) == (
                self.num_classes is not None
            ), "must specify y if and only if the model is class-conditional"
            hs = []
            t_emb = openaimodel_sdxl.timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)

            if self.num_classes is not None:
                assert y.shape[0] == x.shape[0]
                emb = emb + self.label_emb(y)

            # h = x.type(self.dtype)
            h = x
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)
            for module in self.output_blocks:
                h = free_u_cat(h, hs.pop())
                h = module(h, emb, context)
            h = h.type(x.dtype)
            if self.predict_codebook_ids:
                assert False, "not supported anymore. what the f*** are you doing?"
            else:
                return self.out(h)


def free_u_cat(h, h_skip):
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

    return torch.cat([h, h_skip], dim=1)


def filter_skip(x, threshold, scale, scale_high):
    # FFT
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.full((B, C, H, W), scale_high).to(x.device)

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


OriginalUNetModel = openaimodel.UNetModel
if openaimodel_sdxl:
    OriginalSdxlUNetModel = openaimodel_sdxl.UNetModel


def patch_model():
    openaimodel.UNetModel = UNetModel
    if openaimodel_sdxl:
        openaimodel_sdxl.UNetModel = SdxlUNetModel
