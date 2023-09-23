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

            flat_components = []

            default_block_infos = [
                global_state.BlockInfo(
                    1.2, 0.9, 0, 0.5,
                ),
                global_state.BlockInfo(
                    1.4, 0.2, 0, 0.5,
                ),
            ]

            for index in range(2):
                block_n = index + 1
                default_block_info = default_block_infos[index]
                block_flat_components = []

                with gr.Accordion(open=True, label=f"Block {block_n}"):
                    block_flat_components.append(gr.Slider(
                        label=f"Backbone {block_n} Scale",
                        minimum=-1,
                        maximum=3,
                        value=default_block_info.backbone_factor,
                    ))
                    default_block_info.backbone_factor = block_flat_components[-1].value

                    block_flat_components.append(gr.Slider(
                        label=f"Skip {block_n} Scale",
                        minimum=-1,
                        maximum=3,
                        value=default_block_info.skip_factor,
                    ))
                    default_block_info.skip_factor = block_flat_components[-1].value

                    with gr.Row():
                        block_flat_components.append(gr.Slider(
                            label=f"Backbone {block_n} Offset",
                            minimum=0,
                            maximum=1,
                            value=default_block_info.backbone_offset,
                        ))
                        default_block_info.backbone_offset = block_flat_components[-1].value

                        block_flat_components.append(gr.Slider(
                            label=f"Backbone {block_n} Width",
                            minimum=0,
                            maximum=1,
                            value=default_block_info.backbone_width,
                        ))
                        default_block_info.backbone_width = block_flat_components[-1].value

                    flat_components.extend(block_flat_components)

        reset_to_defaults.click(
            fn=lambda: [
                v
                for block_info in default_block_infos
                for v in block_info.to_dict().values()
            ],
            outputs=flat_components,
        )

        infotext_component = gr.HTML(visible=False, interactive=False)

        infotext_component.change(
            fn=self.on_infotext_update,
            inputs=[infotext_component],
            outputs=[infotext_component, *flat_components],
        )

        self.infotext_fields = [
            (infotext_component, "FreeU"),
        ]
        self.paste_field_names = ["FreeU"]

        return enabled, *flat_components

    def on_infotext_update(self, infotext):
        if not infotext:
            return (gr.skip(),) * 9

        params = json.loads(infotext)
        block_infos = [
            global_state.BlockInfo(*block_info.values())
            for block_info in params["block_infos"]
        ]

        return (
            gr.update(value=""),
            *(
                gr.update(value=v)
                for block_info in block_infos
                for v in block_info.to_dict().values()
            )
        )

    def process(
        self,
        p: processing.StableDiffusionProcessing,
        enabled: bool,
        *args
    ):
        block_infos = parse_process_args(args)
        global_state.update(
            enabled=enabled,
            block_infos=block_infos,
        )
        block_infos = global_state.block_infos
        global_state.xyz_locked_attrs.clear()

        if global_state.enabled:
            p.extra_generation_params["FreeU"] = json.dumps({
                "block_infos": [
                    block_info.to_dict()
                    for block_info in block_infos
                ],
            })


def parse_process_args(flat_components):
    return [
        global_state.BlockInfo(*flat_components[i:i+4])
        for i in range(0, len(flat_components), 4)
    ]


unet.patch_model()
xyz_grid.patch()
