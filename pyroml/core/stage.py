from enum import Enum


class Stage(Enum):
    TRAIN = "train"
    VAL = "validation"
    PREDICT = "predict"

    def to_prefix(self):
        return {
            Stage.TRAIN: "tr",
            Stage.VAL: "val",
            Stage.PREDICT: "pr",
        }[self]
