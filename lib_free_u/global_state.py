import re
import sys
from types import ModuleType


enabled: bool = False
backbone_factors: list = [1.0, 1.0]
skip_factors: list = [1.0, 1.0]
xyz_locked_attrs: set = set()


class GlobalState(ModuleType):
    shorthand_re = re.compile(r"^([a-z]{1,2})([0-9]+)$")

    def __init__(self, globs):
        super().__init__(__name__)
        for k, v in globs.items():
            self.__dict__[k] = v

    def __setattr__(self, key, value):
        if key in self.xyz_locked_attrs:
            return

        if match := self.shorthand_re.match(key):
            char, index = match.group(1, 2)
            index = int(index)
            if char == "b":
                self.backbone_factors[index] = value
                return
            elif char == "s":
                self.skip_factors[index] = value
                return

        if key == "backbone_factors":
            for index, value in enumerate(value):
                if f"b{index}" in self.xyz_locked_attrs:
                    continue

                self.backbone_factors[index] = value
        elif key == "skip_factors":
            for index, value in enumerate(value):
                if f"s{index}" in self.xyz_locked_attrs:
                    continue

                self.skip_factors[index] = value
        else:
            self.__dict__[key] = value


sys.modules[__name__] = GlobalState(globals())
