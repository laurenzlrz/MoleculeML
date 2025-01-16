from enum import Enum

class NNProperty(Enum):
    NAME = "name"
    MAX_EPOCHS = "max_epochs"
    DEVICES = "devices"
    STRATEGY = "strategy"
    PRECISION = "precision"
    TOTAL_PARAMETERS = "total_parameters"
    TRAINABLE_PARAMETERS = "trainable_parameters"
    OPTIMIZER = "optimizer"
    LEARNING_RATE = "learning_rate"
    PREDICTIONS = "predictions"
    INPUTS = "inputs"
    OUTPUTS = "outputs"

