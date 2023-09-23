from types import ModuleType
from typing import Optional
from modules import scripts
from lib_free_u import global_state


def patch():
    xyz_module = find_xyz_module()
    xyz_module.axis_options.extend([
        xyz_module.AxisOption("[FreeU] Enabled", str_to_bool, apply_global_state("enabled"), choices=choices_bool),
        *[
            opt
            for index in range(2)
            for opt in [
                xyz_module.AxisOption(f"[FreeU] Block {index+1} Backbone Scale", float, apply_global_state(f"b{index}")),
                xyz_module.AxisOption(f"[FreeU] Block {index+1} Backbone Offset", float, apply_global_state(f"o{index}")),
                xyz_module.AxisOption(f"[FreeU] Block {index+1} Backbone Width", float, apply_global_state(f"w{index}")),
                xyz_module.AxisOption(f"[FreeU] Block {index+1} Skip Scale", float, apply_global_state(f"s{index}")),
            ]
        ]
    ])


def apply_global_state(attr):
    def callback(_p, v, _vs):
        global_state.update(**{attr: v})
        global_state.xyz_locked_attrs.add(attr)

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


def choices_bool():
    return ["False", "True"]


def find_xyz_module() -> Optional[ModuleType]:
    for data in scripts.scripts_data:
        if data.script_class.__module__ in {"xyz_grid.py", "xy_grid.py"} and hasattr(data, "module"):
            return data.module

    return None
