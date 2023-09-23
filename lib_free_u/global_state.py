import re


enabled: bool = False
backbone_factors: list = [1.0, 1.0]
backbone_offsets: list = [1.0, 1.0]
backbone_widths: list = [1.0, 1.0]
skip_factors: list = [1.0, 1.0]
skip_thresholds: list = [0., 0.0]
high_skip_factors: list = [1.0, 1.0]
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
        index = int(index)
        if char == "b":
            backbone_factors[index] = value
            return
        elif char == "s":
            skip_factors[index] = value
            return
        elif char == "o":
            backbone_offsets[index] = value
            return
        elif char == "w":
            backbone_widths[index] = value
            return
        elif char == "t":
            skip_thresholds[index] = value
            return
        elif char == "h":
            high_skip_factors[index] = value
            return

    if key == "backbone_factors":
        for index, value in enumerate(value):
            if f"b{index}" in xyz_locked_attrs:
                continue

            backbone_factors[index] = value
    elif key == "skip_factors":
        for index, value in enumerate(value):
            if f"s{index}" in xyz_locked_attrs:
                continue

            skip_factors[index] = value
    elif key == "backbone_offsets":
        for index, value in enumerate(value):
            if f"o{index}" in xyz_locked_attrs:
                continue

            backbone_offsets[index] = value
    elif key == "backbone_widths":
        for index, value in enumerate(value):
            if f"w{index}" in xyz_locked_attrs:
                continue

            backbone_widths[index] = value
    elif key == "skip_thresholds":
        for index, value in enumerate(value):
            if f"t{index}" in xyz_locked_attrs:
                continue

            skip_thresholds[index] = value
    elif key == "high_skip_factors":
        for index, value in enumerate(value):
            if f"h{index}" in xyz_locked_attrs:
                continue

            high_skip_factors[index] = value
    else:
        globals()[key] = value
