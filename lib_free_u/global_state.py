import dataclasses
import inspect
import re
from typing import Union


@dataclasses.dataclass
class StageInfo:
    backbone_factor: float = 1.0
    skip_factor: float = 1.0
    backbone_offset: float = 0.0
    backbone_width: float = 0.5
    skip_threshold: float = 0.0
    skip_high_end_factor: float = 1.0
    # <- add new fields at the end here for png info backwards compatibility

    def to_dict(self, include_default=False):
        default_stage_info = StageInfo()
        res = {}

        if self.backbone_factor != default_stage_info.backbone_factor or include_default:
            res["backbone_factor"] = self.backbone_factor

        if self.skip_factor != default_stage_info.skip_factor or include_default:
            res["skip_factor"] = self.skip_factor

        if self.backbone_offset != default_stage_info.backbone_offset or include_default:
            res["backbone_offset"] = self.backbone_offset

        if self.backbone_width != default_stage_info.backbone_width or include_default:
            res["backbone_width"] = self.backbone_width

        if self.skip_threshold != default_stage_info.skip_threshold or include_default:
            res["skip_threshold"] = self.skip_threshold

        if self.skip_high_end_factor != default_stage_info.skip_high_end_factor or include_default:
            res["skip_high_end_factor"] = self.skip_high_end_factor

        return res


STAGE_INFO_ARGS_LEN = len(inspect.getfullargspec(StageInfo.__init__)[0]) - 1  # off by one because of self


enabled: bool = False
start_ratio: Union[float, int] = 0.0
stop_ratio: Union[float, int] = 1.0
transition_smoothness: float = 0.0
stage_infos = [
    StageInfo(),
    StageInfo(),
    StageInfo(),
]
xyz_locked_attrs: set = set()
current_sampling_step: float = 0

shorthand_re = re.compile(r"^([a-z]{1,2})([0-9]+)$")


def update(**kwargs):
    for k, v in kwargs.items():
        update_attr(k, v)


def update_attr(key, value):
    if key in xyz_locked_attrs:
        return

    if match := shorthand_re.match(key):
        char, index = match.group(1, 2)
        stage_info = stage_infos[int(index)]
        if char == "b":
            stage_info.backbone_factor = value
            return
        elif char == "s":
            stage_info.skip_factor = value
            return
        elif char == "o":
            stage_info.backbone_offset = value
            return
        elif char == "w":
            stage_info.backbone_width = value
            return
        elif char == "t":
            stage_info.skip_threshold = value
            return
        elif char == "h":
            stage_info.skip_high_end_factor = value
            return

    if key == "stage_infos":
        for index, stage_info in enumerate(value):
            for key, value in stage_info.to_dict(include_default=True).items():
                if key == "backbone_factor":
                    if f"b{index}" not in xyz_locked_attrs:
                        stage_infos[index].backbone_factor = value
                elif key == "skip_factor":
                    if f"s{index}" not in xyz_locked_attrs:
                        stage_infos[index].skip_factor = value
                elif key == "backbone_offset":
                    if f"o{index}" not in xyz_locked_attrs:
                        stage_infos[index].backbone_offset = value
                elif key == "backbone_width":
                    if f"w{index}" not in xyz_locked_attrs:
                        stage_infos[index].backbone_width = value
                elif key == "skip_threshold":
                    if f"t{index}" not in xyz_locked_attrs:
                        stage_infos[index].skip_threshold = value
                elif key == "skip_high_end_factor":
                    if f"h{index}" not in xyz_locked_attrs:
                        stage_infos[index].skip_high_end_factor = value
    else:
        globals()[key] = value
