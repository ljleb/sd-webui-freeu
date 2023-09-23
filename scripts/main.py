from modules import scripts, processing
import gradio as gr
from lib_free_u import global_state, unet, xyz_grid


class FreeUScript(scripts.Script):
    def title(self):
        return "FreeU"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(open=False, label=self.title()):
            with gr.Row():
                enabled = gr.Checkbox(
                    label="Enable",
                    value=False,
                )
                reset_to_defaults = gr.Button(
                        value="Reset to Defaults",
                        size="sm",
                    )
            with gr.Accordion(open=True, label="Block 1"):
                b0 = gr.Slider(
                    label="Backbone 1 Scale",
                    minimum=-1,
                    maximum=3,
                    value=1.2,
                )
                s0 = gr.Slider(
                    label="Skip 1 Scale",
                    minimum=-1,
                    maximum=3,
                    value=0.9,
                )
            with gr.Accordion(open=True, label="Block 2"):
                b1 = gr.Slider(
                    label="Backbone 2 Scale",
                    minimum=-1,
                    maximum=3,
                    value=1.4,
                )
                s1 = gr.Slider(
                    label="Skip 2 Scale",
                    minimum=-1,
                    maximum=3,
                    value=0.2,
                )

        reset_to_defaults.click(
            fn=lambda: (1.2, 0.9, 1.4, 0.2),
            outputs=[b0, s0, b1, s1],
        )

        backbone_infotext = gr.HTML(visible=False, interactive=False)
        skip_infotext = gr.HTML(visible=False, interactive=False)

        backbone_infotext.change(
            fn=self.on_update_scales,
            inputs=[backbone_infotext],
            outputs=[backbone_infotext, b0, b1],
        )
        skip_infotext.change(
            fn=self.on_update_scales,
            inputs=[skip_infotext],
            outputs=[skip_infotext, b0, b1],
        )

        self.infotext_fields = [
            (backbone_infotext, "FreeU Backbone Scales"),
            (skip_infotext, "FreeU Skip Scales"),
        ]
        self.paste_field_names = [p[1] for p in self.infotext_fields]

        return enabled, b0, s0, b1, s1

    def on_update_scales(self, infotext):
        if not infotext:
            return (gr.skip(),) * 3

        return (
            gr.update(value=""),
            *(gr.update(value=float(b)) for b in infotext.split(",")),
        )

    def process(self, p: processing.StableDiffusionProcessing, enabled: bool, b0: float, s0: float, b1: float, s1: float):
        global_state.update(
            enabled=enabled,
            backbone_factors=[b0, b1],
            skip_factors=[s0, s1],
        )
        global_state.xyz_locked_attrs.clear()

        if global_state.enabled:
            p.extra_generation_params["FreeU Backbone Scales"] = ",".join(str(b) for b in global_state.backbone_factors)
            p.extra_generation_params["FreeU Skip Scales"] = ",".join(str(b) for b in global_state.skip_factors)


unet.patch_model()
xyz_grid.patch()
