import json
import os.path
from modules import scripts, script_callbacks, processing, shared
import gradio as gr
from lib_free_u import global_state, unet, xyz_grid


txt2img_steps_component = None
img2img_steps_component = None
txt2img_steps_callbacks = []
img2img_steps_callbacks = []

user_options = []
freeu_options = []
all_options = []

class FreeUScript(scripts.Script):
    def title(self):
        return "FreeU"

    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def load_function(self):
        path = os.path.join(os.path.dirname(__file__), "user-custom-settings.json")
        if os.path.exists(path): # check if file exists
            with open(path, "r") as f:
                all_options = json.load(f)
        else: # if file does not exist, return empty array
            all_options = []
        return all_options

    def save_function(self, options):
        path = os.path.join(os.path.dirname(__file__), "user-custom-settings.json")
        with open(path, "w") as f:
            json.dump(options, f)

    def get_index_by_name(self, name: str, array: list) -> int:
        for i in range(len(array)):
            if array[i]["name"] == name:
                return i
        return -1

    def ui(self, is_img2img):

        default_stage_infos = [
            global_state.StageInfo(1.2, 0.9),
            global_state.StageInfo(1.4, 0.2),
            global_state.StageInfo(1, 1),
        ]
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

        user_options.clear()
        freeu_options.clear()
        all_options.clear()

        user_options.extend(self.load_function())
        freeu_options.extend([
            {"name": "SD1.4 Recommendations", "freeu": [
                v
                for stage_info in default_stage_infos
                for v in stage_info.to_dict(include_default=True).values()
            ], "SystemDefault": True, "start": 0.2, "stop": 1, "smooth":0},
            {"name": "SD2.1 Recommendations", "freeu": [
                v
                for stage_info in sd21_default_stage_infos
                for v in stage_info.to_dict(include_default=True).values()
            ], "SystemDefault": True, "start": 0.2, "stop": 1, "smooth":0},
            {"name": "SDXL Recommendations", "freeu": [
                v
                for stage_info in sdxl_default_stage_infos
                for v in stage_info.to_dict(include_default=True).values()
            ], "SystemDefault": True, "start": 0.2, "stop": 1, "smooth":0},
            ])
        
        all_options.extend(freeu_options)
        all_options.extend(user_options)

        with gr.Accordion(open=False, label=self.title()):
            with gr.Row():
                enabled = gr.Checkbox(
                    label="Enable",
                    value=False,
                )
                user_settings_name = gr.Dropdown(
                    show_label=False,
                    choices=[x["name"] for x in all_options], value=all_options[0]["name"], 
                    type="value", 
                    elem_id=self.elem_id("user_settings"), 
                    allow_custom_value=True,
                    tooltip="Apply button loads settings\nWrite custom name to enable save\nDelete automatically will save to file",
                    size="sm")
                
                apply_config = gr.Button(
                    value="✅",
                    size="lg",
                    elem_classes="tool"
                )
                
                save_to_file = gr.Button(
                    value="💾",
                    size="lg",
                    elem_classes="tool",
                    interactive=False
                )
                refresh_settings = gr.Button(
                    value="🔄",
                    size="lg",
                    elem_classes="tool"
                )
                delete_setting = gr.Button(
                    value="🗑️",
                    size="lg",
                    elem_classes="tool",
                    interactive=False
                )
                
            with gr.Row():
                start_ratio = gr.Slider(
                    label="Start At Step",
                    elem_id=self.elem_id("StartAtStep"), 
                    minimum=0,
                    maximum=1,
                    value=0,
                )

                stop_ratio = gr.Slider(
                    label="Stop At Step",
                    elem_id=self.elem_id("StopAtStep"), 
                    minimum=0,
                    maximum=1,
                    value=1,
                )

                transition_smoothness = gr.Slider(
                    label="Transition Smoothness",
                    elem_id=self.elem_id("TransitionSmoothness"), 
                    minimum=0,
                    maximum=1,
                    value=0,
                )

            flat_components = []

            for index in range(global_state.STAGES_COUNT):
                stage_n = index + 1
                default_stage_info = default_stage_infos[index]

                with gr.Accordion(open=index < 2, label=f"Stage {stage_n}"):
                    with gr.Row():
                        backbone_scale = gr.Slider(
                            label=f"Backbone {stage_n} Scale",
                            elem_id=self.elem_id(f"Backbone{stage_n}Scale"), 
                            minimum=-1,
                            maximum=3,
                            value=default_stage_info.backbone_factor,
                        )
                        default_stage_info.backbone_factor = backbone_scale.value

                        backbone_offset = gr.Slider(
                            label=f"Backbone {stage_n} Offset",
                            elem_id=self.elem_id(f"Backbone{stage_n}Offset"), 
                            minimum=0,
                            maximum=1,
                            value=default_stage_info.backbone_offset,
                        )
                        default_stage_info.backbone_offset = backbone_offset.value

                        backbone_width = gr.Slider(
                            label=f"Backbone {stage_n} Width",
                            elem_id=self.elem_id(f"Backbone{stage_n}Width"),
                            minimum=0,
                            maximum=1,
                            value=default_stage_info.backbone_width,
                        )
                        default_stage_info.backbone_width = backbone_width.value

                    with gr.Row():
                        skip_scale = gr.Slider(
                            label=f"Skip {stage_n} Scale",
                            elem_id=self.elem_id(f"Skip{stage_n}Scale"),
                            minimum=-1,
                            maximum=3,
                            value=default_stage_info.skip_factor,
                        )
                        default_stage_info.skip_factor = skip_scale.value

                        skip_high_end_scale = gr.Slider(
                            label=f"Skip {stage_n} High End Scale",
                            elem_id=self.elem_id(f"Skip{stage_n}HighEndScale"),
                            minimum=-1,
                            maximum=3,
                            value=default_stage_info.skip_high_end_factor,
                        )
                        default_stage_info.skip_high_end_factor = skip_high_end_scale.value

                        skip_cutoff = gr.Slider(
                            label=f"Skip {stage_n} Cutoff",
                            elem_id=self.elem_id(f"Skip{stage_n}Cutoff"),
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
        
        def user_settings_change(user_settings_name):
            user_settings = self.get_index_by_name(user_settings_name, all_options)
            start = all_options[user_settings]["start"]
            stop = all_options[user_settings]["stop"]
            smooth = all_options[user_settings]["smooth"]
            flat = all_options[user_settings]["freeu"]
            return (gr.Slider.update(value=start), gr.Slider.update(value=stop), gr.Slider.update(value=smooth), *flat)
            
        apply_config.click(user_settings_change, inputs=[user_settings_name], outputs=[start_ratio, stop_ratio, transition_smoothness, *flat_components])
        
        def onSettingChange(user_settings_name):
            user_settings = self.get_index_by_name(user_settings_name, all_options)
            can_save = True
            if user_settings is not None and user_settings >= 0:
                can_save = all_options[user_settings]["SystemDefault"] != True
                return (gr.Button.update(interactive=True),gr.Button.update(interactive=can_save),gr.Button.update(interactive=can_save))
            else:
                return (gr.Button.update(interactive=False),gr.Button.update(interactive=False),gr.Button.update(interactive=can_save))

        user_settings_name.change(fn=onSettingChange, inputs=[user_settings_name], outputs=[apply_config,delete_setting, save_to_file])

        def onSaveClick(user_settings_name, start_ratio, stop_ratio, transition_smoothness,*flat_components):
            current_setting = {"name": user_settings_name, "freeu": flat_components,
                "SystemDefault": False, "start": start_ratio, "stop": stop_ratio, "smooth":transition_smoothness}

            user_settings = self.get_index_by_name(user_settings_name, all_options)
            if user_settings >= 0:
                all_options[user_settings] = current_setting
                user_options[user_settings - len(freeu_options)] = current_setting
            else:
                user_options.append(current_setting)
                all_options.append(current_setting)

            self.save_function(user_options)
            
            return (gr.Dropdown.update(value=user_settings_name, choices=[x["name"] for x in all_options]), gr.Button.update(interactive=True), gr.Button.update(interactive=True))

        save_to_file.click(fn=onSaveClick, 
            inputs=[user_settings_name, start_ratio, stop_ratio, transition_smoothness, *flat_components], outputs=[user_settings_name, apply_config, delete_setting]
        )

        def onDeleteClick(user_settings_name):            
            user_settings = self.get_index_by_name(user_settings_name, all_options)
            del all_options[user_settings]
            del user_options[user_settings - len(freeu_options)]
            self.save_function(user_options)
            return (gr.Dropdown.update(value=all_options[0]["name"], choices=[x["name"] for x in all_options]),
                    gr.Button.update(interactive=True),gr.Button.update(interactive=False),gr.Button.update(interactive=False))
        
        delete_setting.click(fn=onDeleteClick, inputs=[user_settings_name], outputs=[user_settings_name, apply_config, save_to_file, delete_setting])

        def onRefreshClick():
            user_options.clear()
            all_options.clear()
            user_options.extend(self.load_function())
            all_options.extend(freeu_options)
            all_options.extend(user_options)
            return (gr.Dropdown.update(value=all_options[0]["name"], choices=[x["name"] for x in all_options]),
                    gr.Button.update(interactive=True),gr.Button.update(interactive=False),gr.Button.update(interactive=False))

        refresh_settings.click(fn=onRefreshClick, outputs=[user_settings_name, apply_config, save_to_file, delete_setting])        

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
