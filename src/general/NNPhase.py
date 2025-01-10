from enum import Enum

# TODO REmove
class NNPhase(Enum):
    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"
    PREDICT = "predict"
    INFERENCE = "inference"
