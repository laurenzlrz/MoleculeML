from src.general.props.MolProperty import MolProperty
from src.pipeline.TrainOrganizer import E2ETraining, Comparator
from src.training.SchnetNN import SchnetNN, AdditionSchnetNN

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


DB_NAME = "Geometry5"
DATASET_PATH = "data/schnet_data/"
TRAINING_ROOT = "data/training"

MODELS = [{
    "name": "model1",
    "constructor": SchnetNN,
    "prediction_keys": [MolProperty.TOTAL_ENERGY],
    "additional_input_keys": [MolProperty.TOTAL_ENERGY_CALCULATED]
}, {
    "name": "model2",
    "constructor": AdditionSchnetNN,
    "prediction_keys": [MolProperty.TOTAL_ENERGY_DIFFERENCE],
    "additional_input_keys": [],
    "measure_keys": [MolProperty.TOTAL_ENERGY],
    "add1": MolProperty.TOTAL_ENERGY_DIFFERENCE,
    "add2": MolProperty.TOTAL_ENERGY_CALCULATED,
    "output": MolProperty.TOTAL_ENERGY
}]



if __name__ == "__main__":
    e2e = E2ETraining(DB_NAME, DATASET_PATH, TRAINING_ROOT, [MolProperty.TOTAL_ENERGY,
                                                             MolProperty.TOTAL_ENERGY_CALCULATED,
                                                             MolProperty.TOTAL_ENERGY_DIFFERENCE],
                      2, 4, 6, 2)
    for model in MODELS:
        e2e.pipeline(model)
    comparator = Comparator("train")
    comparator.save_model_stats(e2e.multi_train_organizer.organizers)