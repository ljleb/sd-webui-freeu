import dataclasses
import re


@dataclasses.dataclass
class BlockInfo:
    backbone_factor: float = 1.0
    skip_factor: float = 1.0
    backbone_width: float = 0.5
    backbone_offset: float = 0.0
    skip_threshold: float = 0.03
    skip_high_end_factor: float = 1.0

    def to_dict(self):
        return {
            "backbone_factor": self.backbone_factor,
            "skip_factor": self.skip_factor,
            "backbone_width": self.backbone_width,
            "backbone_offset": self.backbone_offset,
            "skip_threshold": self.skip_threshold,
            "skip_high_end_factor": self.skip_high_end_factor,
        }


enabled: bool = False
block_infos = [
    BlockInfo(),
    BlockInfo(),
]
xyz_locked_attrs: set = set()

shorthand_re = re.compile(r"^([a-z]{1,2})([0-9]+)$")


def update(**kwargs):
    for k, v in kwargs.items():
        update_attr(k, v)


def update_attr(key, value):
    if key in xyz_locked_attrs:
        return

    if match := shorthand_re.match(key):
        char, index = match.group(1, 2)
        block_info = block_infos[int(index)]
        if char == "b":
            block_info.backbone_factor = value
            return
        elif char == "s":
            block_info.skip_factor = value
            return
        elif char == "o":
            block_info.backbone_offset = value
            return
        elif char == "w":
            block_info.backbone_width = value
            return
        elif char == "t":
            block_info.skip_threshold = value
            return
        elif char == "h":
            block_info.high_skip_factor = value
            return

    if key == "block_infos":
        for index, block_info in enumerate(value):
            for key, value in block_info.to_dict().items():
                if key == "backbone_factor":
                    if f"b{index}" not in xyz_locked_attrs:
                        block_infos[index].backbone_factor = value
                elif key == "skip_factor":
                    if f"s{index}" not in xyz_locked_attrs:
                        block_infos[index].skip_factor = value
                elif key == "backbone_offset":
                    if f"o{index}" not in xyz_locked_attrs:
                        block_infos[index].backbone_offset = value
                elif key == "backbone_width":
                    if f"w{index}" not in xyz_locked_attrs:
                        block_infos[index].backbone_width = value
                elif key == "skip_threshold":
                    if f"t{index}" not in xyz_locked_attrs:
                        block_infos[index].skip_threshold = value
                elif key == "skip_high_end_factor":
                    if f"h{index}" not in xyz_locked_attrs:
                        block_infos[index].skip_high_end_factor = value
    else:
        globals()[key] = value
