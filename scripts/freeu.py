import json
from modules import scripts, script_callbacks, processing
import gradio as gr
from lib_free_u import global_state, unet, xyz_grid


txt2img_steps_component = None
img2img_steps_component = None
txt2img_steps_callbacks = []
img2img_steps_callbacks = []


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
                    value="SD1.4 Recommendations",
                    size="sm",
                )
                reset_to_sd21 = gr.Button(
                    value="SD2.1 Recommendations",
                    size="sm",
                )
                reset_to_sdxl = gr.Button(
                    value="SDXL Recommendations",
                    size="sm",
                )

            with gr.Row():
                start_ratio = gr.Slider(
                    label="Start At Step",
                    minimum=0,
                    maximum=1,
                    value=0,
                )

                stop_ratio = gr.Slider(
                    label="Stop At Step",
                    minimum=0,
                    maximum=1,
                    value=1,
                )

                transition_smoothness = gr.Slider(
                    label="Transition Smoothness",
                    minimum=0,
                    maximum=1,
                    value=0,
                )

            flat_components = []

            default_stage_infos = [
                global_state.StageInfo(1.2, 0.9),
                global_state.StageInfo(1.4, 0.2),
                global_state.StageInfo(1, 1),
            ]

            for index in range(global_state.STAGES_COUNT):
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
                            value=default_stage_info.skip_cutoff,
                        )
                        default_stage_info.skip_cutoff = skip_cutoff.value

                flat_components.extend([
                    backbone_scale,
                    skip_scale,
                    backbone_offset,
                    backbone_width,
                    skip_cutoff,
                    skip_high_end_scale,
                ])


        sd21_default_stage_infos = [
            global_state.StageInfo(1.1, 0.9),
            global_state.StageInfo(1.2, 0.2),
            global_state.StageInfo(1, 1),
        ]
        sdxl_default_stage_infos = [
            global_state.StageInfo(1.1, 0.6),
            global_state.StageInfo(1.2, 0.4),
            global_state.StageInfo(1, 1),
        ]

        reset_to_defaults.click(
            fn=lambda: [
                v
                for stage_info in default_stage_infos
                for v in stage_info.to_dict(include_default=True).values()
            ],
            outputs=flat_components,
        )

        reset_to_sd21.click(
            fn=lambda: [
                v
                for stage_info in sd21_default_stage_infos
                for v in stage_info.to_dict(include_default=True).values()
            ],
            outputs=flat_components,
        )

        reset_to_sdxl.click(
            fn=lambda: [
                v
                for stage_info in sdxl_default_stage_infos
                for v in stage_info.to_dict(include_default=True).values()
            ],
            outputs=flat_components,
        )

        schedule_infotext = gr.HTML(visible=False, interactive=False)
        stages_infotext = gr.HTML(visible=False, interactive=False)

        def register_schedule_infotext_change(steps_component):
            schedule_infotext.change(
                fn=self.on_schedule_infotext_update,
                inputs=[schedule_infotext, steps_component],
                outputs=[schedule_infotext, start_ratio, stop_ratio, transition_smoothness],
            )

        steps_component, steps_callbacks = (
            (img2img_steps_component, img2img_steps_callbacks)
            if is_img2img else
            (txt2img_steps_component, txt2img_steps_callbacks)
        )

        if steps_component is None:
            steps_callbacks.append(register_schedule_infotext_change)
        else:
            register_schedule_infotext_change(steps_component)

        stages_infotext.change(
            fn=self.on_stages_infotext_update,
            inputs=[stages_infotext],
            outputs=[stages_infotext, enabled, *flat_components],
        )

        self.infotext_fields = [
            (schedule_infotext, "FreeU Schedule"),
            (stages_infotext, "FreeU Stages"),
        ]
        self.paste_field_names = [f for _, f in self.infotext_fields]

        return enabled, start_ratio, stop_ratio, transition_smoothness, *flat_components

    def on_schedule_infotext_update(self, infotext, steps):
        if not infotext:
            return (gr.skip(),) * 4

        start_ratio, stop_ratio, transition_smoothness, *_ = infotext.split(", ")

        return (
            gr.update(value=""),
            gr.update(value=unet.to_denoising_step(xyz_grid.int_or_float(start_ratio), steps) / steps),
            gr.update(value=unet.to_denoising_step(xyz_grid.int_or_float(stop_ratio), steps) / steps),
            gr.update(value=float(transition_smoothness)),
        )

    def on_stages_infotext_update(self, infotext):
        if not infotext:
            return (gr.skip(),) * (2 + global_state.STAGES_COUNT * global_state.STAGE_INFO_ARGS_LEN)

        stage_infos = json.loads(infotext)
        stage_infos = [
            global_state.StageInfo(**stage_info)
            for stage_info in stage_infos
        ]
        stage_infos.extend([
            global_state.StageInfo()
            for _ in range(global_state.STAGES_COUNT - len(stage_infos))
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
        *args
    ):
        global_state.current_sampling_step = 0
        if isinstance(args[0], dict):
            state_update = global_state.State(**args[0])
        elif isinstance(args[0], bool):
            state_update = global_state.State({
                "enable": args[0],
                "start_ratio": args[1],
                "stop_ratio": args[2],
                "transition_smoothness": args[3],
                "stage_infos": args[4:],
            })
        else:
            raise TypeError(f"Unrecognized args sequence starting with type {type(args[0])}")

        global_state.instance.update(state_update)
        global_state.xyz_locked_attrs.clear()

        if global_state.instance.enable:
            last_d = False
            p.extra_generation_params["FreeU Stages"] = json.dumps(list(reversed([
                stage_info.to_dict()
                for stage_info in reversed(global_state.instance.stage_infos)
                # strip all empty dicts
                if last_d or stage_info.to_dict() and (last_d := True)
            ])))
            p.extra_generation_params["FreeU Schedule"] = ", ".join([
                str(global_state.instance.start_ratio),
                str(global_state.instance.stop_ratio),
                str(global_state.instance.transition_smoothness),
            ])

    def postprocess_batch(self, p, *args, **kwargs):
        global_state.current_sampling_step = 0


def increment_sampling_step(*_args, **_kwargs):
    global_state.current_sampling_step += 1


try:
    script_callbacks.on_cfg_after_cfg(increment_sampling_step)
except AttributeError:
    # webui < 1.6.0
    # normally we should increment the current sampling step after cfg
    # but as long as we don't need to run code during cfg it should be fine to increment early
    script_callbacks.on_cfg_denoised(increment_sampling_step)


def on_after_component(component, **kwargs):
    global txt2img_steps_component, img2img_steps_component

    if kwargs.get("elem_id", None) == "img2img_steps":
        img2img_steps_component = component
        for callback in img2img_steps_callbacks:
            callback(component)

    if kwargs.get("elem_id", None) == "txt2img_steps":
        txt2img_steps_component = component
        for callback in txt2img_steps_callbacks:
            callback(component)


script_callbacks.on_after_component(on_after_component)


unet.patch()
xyz_grid.patch()
