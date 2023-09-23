from lib_free_u import global_state
from ldm.modules.diffusionmodules import openaimodel
from sgm.modules.diffusionmodules import openaimodel as openaimodel_sdxl
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
    if h.shape[1] == 1280:
        h[:, :640] = h[:, :640] * global_state.backbone_factors[0]
        h_skip = filter_skip(h_skip, threshold=1, scale=global_state.skip_factors[0])
    if h.shape[1] == 640:
        h[:, :320] = h[:, :320] * global_state.backbone_factors[1]
        h_skip = filter_skip(h_skip, threshold=1, scale=global_state.skip_factors[1])

    return torch.cat([h, h_skip], dim=1)


def filter_skip(x, threshold, scale):
    # FFT
    x = x.to(dtype=torch.float32)
    x_freq = torch.fft.fftn(x, dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda()

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, s=(x.shape[-2], x.shape[-1])).real

    x_filtered = x_filtered.to(dtype=torch.float16)

    return x_filtered


OriginalUNetModel = openaimodel.UNetModel
OriginalSdxlUNetModel = openaimodel_sdxl.UNetModel


def patch_model():
    openaimodel.UNetModel = UNetModel
    openaimodel_sdxl.UNetModel = SdxlUNetModel
