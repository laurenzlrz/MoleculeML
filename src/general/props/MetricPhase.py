from enum import Enum

class MetricDataType(Enum):
    """
    Enum for different metric logging intervals. Used to create summaries of models in the end on different levels.
    """
    EPOCH = "epoch"
    STEP = "step"
    SUMMARY = "summary"
    CHECK = "check"
    COMPARE = "compare"

class MODUS(Enum):
    """
    Enum for different modes of the model. Used to create summaries of models in the end on different levels.
    """
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    SUMMARY = "summary"
    COMPARE = "compare"