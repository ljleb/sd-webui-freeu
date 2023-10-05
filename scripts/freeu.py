import json
import gradio as gr
from modules import scripts, script_callbacks, processing, shared
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
        global_state.reload_presets()
        default_stage_infos = next(iter(global_state.all_presets.values())).stage_infos

        with gr.Accordion(open=False, label=self.title()):
            with gr.Row():
                enabled = gr.Checkbox(
                    label="Enable",
                    value=False,
                )
                preset_name = gr.Dropdown(
                    show_label=False,
                    choices=list(global_state.all_presets.keys()),
                    value=next(iter(global_state.all_presets.keys())),
                    type="value",
                    elem_id=self.elem_id("preset_name"),
                    allow_custom_value=True,
                    tooltip="Apply button loads settings\nWrite custom name to enable save\nDelete automatically will save to file",
                    size="sm",
                )

                is_custom_preset = preset_name.value not in global_state.default_presets
                preset_exists = preset_name.value in global_state.all_presets

                apply_preset = gr.Button(
                    value="✅",
                    size="lg",
                    elem_classes="tool",
                    interactive=preset_exists,
                )
                save_preset = gr.Button(
                    value="💾",
                    size="lg",
                    elem_classes="tool",
                    interactive=is_custom_preset,
                )
                refresh_presets = gr.Button(
                    value="🔄",
                    size="lg",
                    elem_classes="tool"
                )
                delete_preset = gr.Button(
                    value="🗑️",
                    size="lg",
                    elem_classes="tool",
                    interactive=is_custom_preset and preset_exists,
                )

            with gr.Row():
                start_ratio = gr.Slider(
                    label="Start At Step",
                    elem_id=self.elem_id("start_at_step"),
                    minimum=0,
                    maximum=1,
                    value=0,
                )

                stop_ratio = gr.Slider(
                    label="Stop At Step",
                    elem_id=self.elem_id("stop_at_step"),
                    minimum=0,
                    maximum=1,
                    value=1,
                )

                transition_smoothness = gr.Slider(
                    label="Transition Smoothness",
                    elem_id=self.elem_id("transition_smoothness"),
                    minimum=0,
                    maximum=1,
                    value=0,
                )

            flat_stage_infos = []

            for index in range(global_state.STAGES_COUNT):
                stage_n = index + 1
                default_stage_info = default_stage_infos[index]

                with gr.Accordion(open=index < 2, label=f"Stage {stage_n}"):
                    with gr.Row():
                        backbone_scale = gr.Slider(
                            label=f"Backbone {stage_n} Scale",
                            elem_id=self.elem_id(f"backbone_scale_{stage_n}"),
                            minimum=-1,
                            maximum=3,
                            value=default_stage_info.backbone_factor,
                        )

                        backbone_offset = gr.Slider(
                            label=f"Backbone {stage_n} Offset",
                            elem_id=self.elem_id(f"backbone_offset_{stage_n}"),
                            minimum=0,
                            maximum=1,
                            value=default_stage_info.backbone_offset,
                        )

                        backbone_width = gr.Slider(
                            label=f"Backbone {stage_n} Width",
                            elem_id=self.elem_id(f"backbone_width_{stage_n}"),
                            minimum=0,
                            maximum=1,
                            value=default_stage_info.backbone_width,
                        )

                    with gr.Row():
                        skip_scale = gr.Slider(
                            label=f"Skip {stage_n} Scale",
                            elem_id=self.elem_id(f"skip_scale_{stage_n}"),
                            minimum=-1,
                            maximum=3,
                            value=default_stage_info.skip_factor,
                        )

                        skip_high_end_scale = gr.Slider(
                            label=f"Skip {stage_n} High End Scale",
                            elem_id=self.elem_id(f"skip_high_end_scale_{stage_n}"),
                            minimum=-1,
                            maximum=3,
                            value=default_stage_info.skip_high_end_factor,
                        )

                        skip_cutoff = gr.Slider(
                            label=f"Skip {stage_n} Cutoff",
                            elem_id=self.elem_id(f"skip_cutoff_{stage_n}"),
                            minimum=0.0,
                            maximum=1.0,
                            value=default_stage_info.skip_cutoff,
                        )

                flat_stage_infos.extend([
                    backbone_scale,
                    skip_scale,
                    backbone_offset,
                    backbone_width,
                    skip_cutoff,
                    skip_high_end_scale,
                ])

        def on_preset_name_change(preset_name):
            is_custom_preset = preset_name not in global_state.default_presets
            preset_exists = preset_name in global_state.all_presets
            return (
                gr.Button.update(interactive=preset_exists),
                gr.Button.update(interactive=is_custom_preset),
                gr.Button.update(interactive=is_custom_preset and preset_exists),
            )

        preset_name.change(
            fn=on_preset_name_change,
            inputs=[preset_name],
            outputs=[apply_preset, save_preset, delete_preset],
        )

        def on_apply_click(user_settings_name):
            preset = global_state.all_presets[user_settings_name]
            return (
                gr.Slider.update(value=preset.start_ratio),
                gr.Slider.update(value=preset.stop_ratio),
                gr.Slider.update(value=preset.transition_smoothness),
                *[
                    gr.update(value=v)
                    for stage_info in preset.stage_infos
                    for v in stage_info.to_dict(include_default=True).values()
                ],
            )

        apply_preset.click(
            fn=on_apply_click,
            inputs=[preset_name],
            outputs=[start_ratio, stop_ratio, transition_smoothness, *flat_stage_infos],
        )

        def on_save_click(preset_name, start_ratio, stop_ratio, transition_smoothness, *flat_stage_infos):
            global_state.all_presets[preset_name] = global_state.State(
                stage_infos=flat_stage_infos,
                start_ratio=start_ratio,
                stop_ratio=stop_ratio,
                transition_smoothness=transition_smoothness,
            )
            global_state.save_presets()

            return (
                gr.Dropdown.update(choices=list(global_state.all_presets.keys())),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
            )

        save_preset.click(
            fn=on_save_click,
            inputs=[preset_name, start_ratio, stop_ratio, transition_smoothness, *flat_stage_infos],
            outputs=[preset_name, apply_preset, delete_preset],
        )

        def on_refresh_click(preset_name):
            global_state.reload_presets()
            is_custom_preset = preset_name not in global_state.default_presets
            preset_exists = preset_name in global_state.all_presets

            return (
                gr.Dropdown.update(value=preset_name, choices=list(global_state.all_presets.keys())),
                gr.Button.update(interactive=preset_exists),
                gr.Button.update(interactive=is_custom_preset),
                gr.Button.update(interactive=is_custom_preset and preset_exists),
            )

        refresh_presets.click(
            fn=on_refresh_click,
            inputs=[preset_name],
            outputs=[preset_name, apply_preset, save_preset, delete_preset],
        )

        def on_delete_click(preset_name):
            preset_name_index = list(global_state.all_presets.keys()).index(preset_name)
            del global_state.all_presets[preset_name]
            global_state.save_presets()

            preset_name_index = min(len(global_state.all_presets) - 1, preset_name_index)
            preset_names = list(global_state.all_presets.keys())
            preset_name = preset_names[preset_name_index]

            is_custom_preset = preset_name not in global_state.default_presets
            preset_exists = preset_name in global_state.all_presets
            return (
                gr.Dropdown.update(value=preset_name, choices=preset_names),
                gr.Button.update(interactive=preset_exists),
                gr.Button.update(interactive=is_custom_preset),
                gr.Button.update(interactive=is_custom_preset and preset_exists),
            )

        delete_preset.click(
            fn=on_delete_click,
            inputs=[preset_name],
            outputs=[preset_name, apply_preset, save_preset, delete_preset],
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
            outputs=[stages_infotext, enabled, *flat_stage_infos],
        )

        self.infotext_fields = [
            (schedule_infotext, "FreeU Schedule"),
            (stages_infotext, "FreeU Stages"),
        ]
        self.paste_field_names = [f for _, f in self.infotext_fields]

        return enabled, start_ratio, stop_ratio, transition_smoothness, *flat_stage_infos

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
            gr.update(value=shared.opts.data.get("freeu_png_info_auto_enable", True)),
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
            i = global_state.STATE_ARGS_LEN - 1
            state_update = global_state.State(args[0], *[float(n) for n in args[1:i]], args[i:])
        else:
            raise TypeError(f"Unrecognized args sequence starting with type {type(args[0])}")

        global_state.instance.update(state_update)
        global_state.xyz_locked_attrs.clear()
        global_state.xyz_locked = False

        if not global_state.instance.enable:
            return

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


def on_ui_settings():
    section = ("freeu", "FreeU")
    shared.opts.add_option(
        "freeu_png_info_auto_enable",
        shared.OptionInfo(
            default=True,
            label="Auto enable when loading the PNG Info of a generation that used FreeU",
            section=section,
        )
    )


script_callbacks.on_ui_settings(on_ui_settings)


unet.patch()
xyz_grid.patch()
