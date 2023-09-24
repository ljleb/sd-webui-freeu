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

            default_stage_infos = [
                global_state.StageInfo(1.2, 0.9),
                global_state.StageInfo(1.4, 0.2),
                global_state.StageInfo(1, 1),
            ]

            for index in range(len(global_state.stage_infos)):
                stage_n = index + 1
                default_stage_info = default_stage_infos[index]

                with gr.Accordion(open=index < 2, label=f"Stage {stage_n}"):
                    with gr.Row():
                        backbone_scale = gr.Slider(
                            label=f"Backbone {stage_n} Scale",
                            minimum=-1,
                            maximum=3,
                            value=default_stage_info.backbone_factor,
                        )
                        default_stage_info.backbone_factor = backbone_scale.value

                        backbone_offset = gr.Slider(
                            label=f"Backbone {stage_n} Offset",
                            minimum=0,
                            maximum=1,
                            value=default_stage_info.backbone_offset,
                        )
                        default_stage_info.backbone_offset = backbone_offset.value

                        backbone_width = gr.Slider(
                            label=f"Backbone {stage_n} Width",
                            minimum=0,
                            maximum=1,
                            value=default_stage_info.backbone_width,
                        )
                        default_stage_info.backbone_width = backbone_width.value

                    with gr.Row():
                        skip_scale = gr.Slider(
                            label=f"Skip {stage_n} Scale",
                            minimum=-1,
                            maximum=3,
                            value=default_stage_info.skip_factor,
                        )
                        default_stage_info.skip_factor = skip_scale.value

                        skip_high_end_scale = gr.Slider(
                            label=f"Skip {stage_n} High End Scale",
                            minimum=-1,
                            maximum=3,
                            value=default_stage_info.skip_high_end_factor,
                        )
                        default_stage_info.skip_high_end_factor = skip_high_end_scale.value

                        skip_cutoff = gr.Slider(
                            label=f"Skip {stage_n} Cutoff",
                            minimum=0.0,
                            maximum=1.0,
                            value=default_stage_info.skip_threshold,
                        )
                        default_stage_info.skip_threshold = skip_cutoff.value

                flat_components.extend([
                    backbone_scale,
                    skip_scale,
                    backbone_offset,
                    backbone_width,
                    skip_cutoff,
                    skip_high_end_scale,
                ])

        reset_to_defaults.click(
            fn=lambda: [
                v
                for stage_info in default_stage_infos
                for v in stage_info.to_dict(include_default=True).values()
            ],
            outputs=flat_components,
        )

        infotext_component = gr.HTML(visible=False, interactive=False)

        infotext_component.change(
            fn=self.on_infotext_update,
            inputs=[infotext_component],
            outputs=[infotext_component, enabled, *flat_components],
        )

        self.infotext_fields = [
            (infotext_component, "FreeU Stages"),
        ]
        self.paste_field_names = ["FreeU Stages"]

        return enabled, *flat_components

    def on_infotext_update(self, infotext):
        if not infotext:
            return (gr.skip(),) * (2 + len(global_state.stage_infos) * global_state.STAGE_INFO_ARGS_LEN)

        stage_infos = json.loads(infotext)
        stage_infos = [
            global_state.StageInfo(**stage_info)
            for stage_info in stage_infos
        ]
        stage_infos.extend([
            global_state.StageInfo()
            for _ in range(len(global_state.stage_infos) - len(stage_infos))
        ])

        return (
            gr.update(value=""),
            gr.update(value=True),
            *(
                gr.update(value=v)
                for stage_info in stage_infos
                for v in stage_info.to_dict(include_default=True).values()
            )
        )

    def process(
        self,
        p: processing.StableDiffusionProcessing,
        enabled: bool,
        *flat_stage_infos
    ):
        global_state.update(
            enabled=enabled,
            stage_infos=group_stage_infos(flat_stage_infos),
        )
        global_state.xyz_locked_attrs.clear()

        if global_state.enabled:
            last_d = False
            p.extra_generation_params["FreeU Stages"] = json.dumps(list(reversed([
                stage_info.to_dict()
                for stage_info in reversed(global_state.stage_infos)
                # strip all empty dicts
                if last_d or stage_info.to_dict() and (last_d := True)
            ])))


def group_stage_infos(flat_components):
    return [
        global_state.StageInfo(*flat_components[i:i + global_state.STAGE_INFO_ARGS_LEN])
        for i in range(0, len(flat_components), global_state.STAGE_INFO_ARGS_LEN)
    ]


unet.patch()
xyz_grid.patch()
