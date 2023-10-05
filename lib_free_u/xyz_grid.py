import sys
from types import ModuleType
from typing import Optional
from modules import scripts
from lib_free_u import global_state


def patch():
    xyz_module = find_xyz_module()
    if xyz_module is None:
        print("[sd-webui-freeu]", "xyz_grid.py not found.", file=sys.stderr)
        return

    xyz_module.axis_options.extend([
        xyz_module.AxisOption("[FreeU] Enabled", str_to_bool, apply_global_state("enable"), choices=choices_bool),
        xyz_module.AxisOption("[FreeU] Start At Step", int_or_float, apply_global_state("start_ratio")),
        xyz_module.AxisOption("[FreeU] Stop At Step", int_or_float, apply_global_state("stop_ratio")),
        xyz_module.AxisOption("[FreeU] Transition Smoothness", int_or_float, apply_global_state("transition_smoothness")),
        xyz_module.AxisOption("[FreeU] Presets", str, apply_global_preset, choices=choices_preset),
        *[
            opt
            for index in range(global_state.STAGES_COUNT)
            for opt in [
                xyz_module.AxisOption(f"[FreeU] Stage {index+1} Backbone Scale", float, apply_global_state(f"b{index}")),
                xyz_module.AxisOption(f"[FreeU] Stage {index+1} Backbone Offset", float, apply_global_state(f"o{index}")),
                xyz_module.AxisOption(f"[FreeU] Stage {index+1} Backbone Width", float, apply_global_state(f"w{index}")),
                xyz_module.AxisOption(f"[FreeU] Stage {index+1} Skip Scale", float, apply_global_state(f"s{index}")),
                xyz_module.AxisOption(f"[FreeU] Stage {index+1} Skip Cutoff", float, apply_global_state(f"t{index}")),
                xyz_module.AxisOption(f"[FreeU] Stage {index+1} Skip High End Scale", float, apply_global_state(f"h{index}")),
            ]
        ]
    ])

def apply_global_preset(_p, v, _vs):
    global_state.instance.__dict__.update(global_state.all_presets[v].__dict__)
    global_state.xyz_locked = True

def apply_global_state(k):
    def callback(_p, v, _vs):
        global_state.instance.update_attr(k, v)
        global_state.xyz_locked_attrs.add(k)

    return callback


def str_to_bool(string):
    string = str(string)
    if string in ["None", ""]:
        return None
    elif string.lower() in ["true", "1"]:
        return True
    elif string.lower() in ["false", "0"]:
        return False
    else:
        raise ValueError(f"Could not convert string to boolean: {string}")


def int_or_float(string):
    try:
        return int(string)
    except ValueError:
        return float(string)


def choices_bool():
    return ["False", "True"]

def choices_preset():
    return list(global_state.all_presets.keys())

def find_xyz_module() -> Optional[ModuleType]:
    for data in scripts.scripts_data:
        if data.script_class.__module__ in {"xyz_grid.py", "xy_grid.py"} and hasattr(data, "module"):
            return data.module

    return None
