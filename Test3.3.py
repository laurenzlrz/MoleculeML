from src.general.props.MolProperty import MolProperty
from src.pipeline.TrainOrganizer import E2ETraining, Comparator
from src.training.SchnetNN import SchnetNN, AdditionSchnetNN

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DB_NAME = "Geometry5"
DATASET_PATH = "data/schnet_data/"
TRAINING_ROOT = "data/training"

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

# TODO improve overlaying structure
# TODO check batches numbers (why so short)
# TODO clear path defining
# TODO also csv saving from logging results

if __name__ == "__main__":
    e2e = E2ETraining(DB_NAME, DATASET_PATH, TRAINING_ROOT, [MolProperty.TOTAL_ENERGY,
                                                             MolProperty.TOTAL_ENERGY_CALCULATED,
                                                             MolProperty.TOTAL_ENERGY_DIFFERENCE],
                      2, 4, 6)
    e2e.iterate_pipeline(MODELS)

"""
 
"""
