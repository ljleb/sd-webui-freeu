import dataclasses
import inspect
import json
import pathlib
import re
from typing import Union, List, Any


@dataclasses.dataclass
class StageInfo:
    backbone_factor: float = 1.0
    skip_factor: float = 1.0
    backbone_offset: float = 0.0
    backbone_width: float = 0.5
    skip_cutoff: float = 0.0
    skip_high_end_factor: float = 1.0
    # <- add new fields at the end here for png info backwards compatibility

    def to_dict(self, include_default=False):
        default_stage_info = StageInfo()
        res = vars(self).copy()
        for k, v in res.copy().items():
            if not include_default and v == getattr(default_stage_info, k):
                del res[k]

        return res


STAGE_INFO_ARGS_LEN = len(inspect.getfullargspec(StageInfo.__init__)[0]) - 1  # off by one because of self
STAGES_COUNT = 3
shorthand_re = re.compile(r"^([a-z]{1,2})([0-9]+)$")

xyz_locked_attrs: set = set()
xyz_locked: bool = False
current_sampling_step: float = 0


@dataclasses.dataclass
class State:
    enable: bool = True
    start_ratio: Union[float, int] = 0.0
    stop_ratio: Union[float, int] = 1.0
    transition_smoothness: float = 0.0
    stage_infos: List[Union[StageInfo, dict, Any]] = dataclasses.field(default_factory=lambda: [StageInfo() for _ in range(STAGES_COUNT)])

    def __post_init__(self):
        self.stage_infos = self.group_stage_infos()

    def group_stage_infos(self):
        res = []
        i = 0
        while i < len(self.stage_infos) and len(res) < STAGES_COUNT:
            if isinstance(self.stage_infos[i], StageInfo):
                res.append(self.stage_infos[i])
                i += 1
            elif isinstance(self.stage_infos[i], dict):
                res.append(StageInfo(**self.stage_infos[i]))
                i += 1
            else:
                next_i = i + STAGE_INFO_ARGS_LEN
                res.append(StageInfo(*self.stage_infos[i:next_i]))
                i = next_i

        for _ in range(STAGES_COUNT - len(res)):
            res.append(StageInfo())

        return res

    def to_dict(self):
        result = vars(self).copy()
        result["stage_infos"] = [stage_info.to_dict() for stage_info in result["stage_infos"]]
        del result["enable"]
        return result

    def update(self, state):
        for k, v in vars(state).items():
            self.update_attr(k, v)

    def update_attr(self, key, value):
        if xyz_locked or key in xyz_locked_attrs:
            return

        if match := shorthand_re.match(key):
            char, index = match.group(1, 2)
            stage_info = self.stage_infos[int(index)]
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
                stage_info.skip_cutoff = value
                return
            elif char == "h":
                stage_info.skip_high_end_factor = value
                return

        if key == "stage_infos":
            for index, stage_info in enumerate(value):
                for key, value in stage_info.to_dict(include_default=True).items():
                    if key == "backbone_factor":
                        if f"b{index}" not in xyz_locked_attrs:
                            self.stage_infos[index].backbone_factor = value
                    elif key == "skip_factor":
                        if f"s{index}" not in xyz_locked_attrs:
                            self.stage_infos[index].skip_factor = value
                    elif key == "backbone_offset":
                        if f"o{index}" not in xyz_locked_attrs:
                            self.stage_infos[index].backbone_offset = value
                    elif key == "backbone_width":
                        if f"w{index}" not in xyz_locked_attrs:
                            self.stage_infos[index].backbone_width = value
                    elif key == "skip_cutoff":
                        if f"t{index}" not in xyz_locked_attrs:
                            self.stage_infos[index].skip_cutoff = value
                    elif key == "skip_high_end_factor":
                        if f"h{index}" not in xyz_locked_attrs:
                            self.stage_infos[index].skip_high_end_factor = value
        else:
            self.__dict__[key] = value


STATE_ARGS_LEN = len(inspect.getfullargspec(State.__init__)[0]) - 1  # off by one because of self
PRESETS_PATH = pathlib.Path(__file__).parent.parent / "presets.json"

instance = State()
default_presets = {
    "SD1.4 Recommendations": State(
        stage_infos=[
            StageInfo(1.2, 0.9),
            StageInfo(1.4, 0.2),
            StageInfo(1, 1),
        ],
    ),
    "SD2.1 Recommendations": State(
        stage_infos=[
            StageInfo(1.1, 0.9),
            StageInfo(1.2, 0.2),
            StageInfo(1, 1),
        ],
    ),
    "SDXL Recommendations": State(
        stage_infos=[
            StageInfo(1.1, 0.6),
            StageInfo(1.2, 0.4),
            StageInfo(1, 1),
        ],
    ),
}
all_presets = {}


def reload_presets():
    all_presets.clear()
    all_presets.update(default_presets)
    all_presets.update(load_presets())


def load_presets():
    if not PRESETS_PATH.exists():
        return []

    with open(PRESETS_PATH, "r") as f:
        return {
            k: State(**v)
            for k, v in json.load(f).items()
        }


def save_presets(presets=None):
    if presets is None:
        presets = get_user_presets()

    presets = {k: v.to_dict() for k, v in presets.items()}

    with open(PRESETS_PATH, "w") as f:
        json.dump(presets, f)


def get_user_presets():
    return {
        k: v
        for k, v in all_presets.items()
        if k not in default_presets
    }
