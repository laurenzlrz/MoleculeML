from src.general.props.MolProperty import MolProperty
from src.end2end.E2E import E2E
from src.general.utils.Utility import setup_gpu
from src.training.SchnetNN import SchnetNN, AdditionSchnetNN
from src.training.ModelConfig import SchnetNNBuilder, AdditionSchnetNNBuilder

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# TODO improve overlaying structure
# TODO check if addition models work correctly
# TODO also csv saving from logging results
# TODO In future change trainer to train multiple models at once
# TODO make a top-down approach, so that at first models and training programs are defined and
#  change program structure accordingly to this process

TRAIN_DB_NAME = "Aspirin1000Benzol1000"
TRAIN_DATASET_PATH = "data/databases/"

TEST_DB_NAME = "Aspirin1000Benzol1000"
TEST_DATASET_PATH = "data/databases/"

TRAINING_ROOT = "data/trainingAspirin2"
TRAIN_NAME = "TestRunAB"

TEST_MODEL1 = SchnetNNBuilder([], [MolProperty.TOTAL_ENERGY_TRUTH])

TEST_MODEL2 = AdditionSchnetNNBuilder([],
                                      [MolProperty.TOTAL_ENERGY_DIFFERENCE],
                                      [MolProperty.TOTAL_ENERGY_TRUTH],
                                      MolProperty.TOTAL_ENERGY_DIFFERENCE,
                                      MolProperty.TOTAL_ENERGY_CALCULATED,
                                      MolProperty.TOTAL_ENERGY_TRUTH)

TEST_MODELS = {"add_1700": TEST_MODEL1}

if __name__ == "__main__":
    setup_gpu()
    e2e = E2E(TEST_MODELS, TRAINING_ROOT, TRAIN_NAME, TRAIN_DATASET_PATH, TRAIN_DB_NAME, TEST_DATASET_PATH, TEST_DB_NAME,
              [MolProperty.TOTAL_ENERGY_TRUTH, MolProperty.TOTAL_ENERGY_CALCULATED,
               MolProperty.TOTAL_ENERGY_DIFFERENCE],
              20, 200, 1400)
    e2e.process()

"""
LEGACY
"""
MODELS = [
    {
        "name": "model1",
        "constructor": SchnetNN,
        "prediction_keys": [MolProperty.TOTAL_ENERGY],
        "additional_input_keys": [MolProperty.TOTAL_ENERGY_CALCULATED]
    },
    {
        "name": "model2",
        "constructor": AdditionSchnetNN,
        "prediction_keys": [MolProperty.TOTAL_ENERGY_DIFFERENCE],
        "additional_input_keys": [],
        "measure_keys": [MolProperty.TOTAL_ENERGY],
        "add1": MolProperty.TOTAL_ENERGY_DIFFERENCE,
        "add2": MolProperty.TOTAL_ENERGY_CALCULATED,
        "output": MolProperty.TOTAL_ENERGY
    }]
