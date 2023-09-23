from types import ModuleType
from typing import Optional
from modules import scripts
from lib_free_u import global_state


def patch():
    xyz_module = find_xyz_module()
    xyz_module.axis_options.extend([
        xyz_module.AxisOption("[Free U] Enabled", str_to_bool, apply_global_state("enabled"), choices=choices_bool),
        xyz_module.AxisOption("[Free U] Block 1 Backbone Scale", float, apply_global_state("b1")),
        xyz_module.AxisOption("[Free U] Block 1 Skip Scale", float, apply_global_state("s1")),
        xyz_module.AxisOption("[Free U] Block 2 Backbone Scale", float, apply_global_state("b2")),
        xyz_module.AxisOption("[Free U] Block 2 Skip Scale", float, apply_global_state("s2")),
    ])


def apply_global_state(attr):
    def callback(_p, v, _vs):
        if hasattr(global_state, attr):
            setattr(global_state, attr, v)
            global_state.xyz_locked_attrs.add(attr)
            return

        index = int(attr[1:])
        if "b" in attr:
            factors = global_state.backbone_factors
            global_state.xyz_locked_attrs.add("backbone_factors")
        else:
            factors = global_state.skip_factors
            global_state.xyz_locked_attrs.add("skip_factors")

        factors[index] = v

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
