import json

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

                with gr.Row():
                    o0 = gr.Slider(
                        label="Backbone 1 Offset",
                        minimum=0,
                        maximum=1,
                        value=0,
                    )
                    w0 = gr.Slider(
                        label="Backbone 1 Width",
                        minimum=0,
                        maximum=1,
                        value=0.5,
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

                with gr.Row():
                    o1 = gr.Slider(
                        label="Backbone 2 Offset",
                        minimum=0,
                        maximum=1,
                        value=0,
                    )
                    w1 = gr.Slider(
                        label="Backbone 2 Width",
                        minimum=0,
                        maximum=1,
                        value=0.5,
                    )

        reset_to_defaults.click(
            fn=lambda: (1.2, 0.9, 0, 0.5, 1.4, 0.2, 0, 0.5),
            outputs=[b0, s0, o0, w0, b1, s1, o1, w1],
        )

        infotext_component = gr.HTML(visible=False, interactive=False)

        infotext_component.change(
            fn=self.on_update_scales,
            inputs=[infotext_component],
            outputs=[infotext_component, b0, s0, o0, w0, b1, s1, o1, w1],
        )

        self.infotext_fields = [
            (infotext_component, "FreeU"),
        ]
        self.paste_field_names = [p[1] for p in self.infotext_fields]

        return enabled, b0, s0, o0, w0, b1, s1, o1, w1

    def on_update_scales(self, infotext):
        if not infotext:
            return (gr.skip(),) * 9

        params = json.loads(infotext)

        return (
            gr.update(value=""),
            *(
                gr.update(value=v)
                for t in zip(params["backbone_factors"], params["skip_factors"], params["backbone_offsets"], params["backbone_widths"])
                for v in t)
        )

    def process(
        self,
        p: processing.StableDiffusionProcessing,
        enabled: bool,
        b0: float, s0: float, o0: float, w0: float,
        b1: float, s1: float, o1: float, w1: float
    ):
        global_state.update(
            enabled=enabled,
            backbone_factors=[b0, b1],
            skip_factors=[s0, s1],
            backbone_offsets=[o0, o1],
            backbone_widths=[w0, w1],
        )
        global_state.xyz_locked_attrs.clear()

        if global_state.enabled:
            p.extra_generation_params["FreeU"] = json.dumps({
                "backbone_factors": global_state.backbone_factors,
                "skip_factors": global_state.skip_factors,
                "backbone_offsets": global_state.backbone_offsets,
                "backbone_widths": global_state.backbone_widths,
            })


unet.patch_model()
xyz_grid.patch()
