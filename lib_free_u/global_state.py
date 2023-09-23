import sys
from types import ModuleType


enabled: bool = False
backbone_factors: list = [1.0, 1.0]
skip_factors: list = [1.0, 1.0]
xyz_locked_attrs: set = set()


class GlobalState(ModuleType):
    def __init__(self, globs):
        super().__init__(__name__)
        for k, v in globs.items():
            setattr(self, k, v)

    def __setattr__(self, key, value):
        if key not in getattr(self, "xyz_locked_attrs", set()) or key == "xyz_locked_attrs":
            self.__dict__[key] = value


sys.modules[__name__] = GlobalState(globals())
