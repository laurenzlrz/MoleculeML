import os
from typing import Dict, Any

import pandas as pd
import schnetpack as spk
from pytorch_lightning import loggers, Trainer

from src.general.props import NNDefaultValue
from src.general.props.NNProperty import NNProperty
from src.general.SchnetAdapterStrings import NN_PERFORMANCE_KPI

LOG_DIR_NAME = "schnet_logs"
BEST_MODEL_PREFIX = "best"
STAT_PATH_FORMAT = "{path}/trainer_properties.csv"
SEPARATOR = "_"

DEF_NUMBER_SAVES = NNDefaultValue.DEF_CHECKPOINT_NUMBER_SAVES
DEF_LOGGING_INTERVAL = NNDefaultValue.DEF_LOGGING_INTERVAL
DEF_NUMBER_EPOCHS = NNDefaultValue.DEF_NUMBER_EPOCHS

LOGGING_SUB_DIR = "logging"

# logg_dir=None,

class SchnetTrainer:

    def __init__(self, training_root, logging_root, callbacks=None):
        self.trainer: Trainer = None

        self.trainings_root = training_root
        self.logging_root = logging_root

        #if logg_dir is None:
        #    self.log_dir = LOGGING_SUB_DIR

        best_model_name = f"{self.trainings_root}/{BEST_MODEL_PREFIX}"

        if not os.path.exists(self.trainings_root):
            # Ordner erstellen
            os.makedirs(self.trainings_root)

        if not os.path.exists(self.logging_root):
            # Ordner erstellen
            os.makedirs(logging_root)

        if callbacks is None:
            self.callbacks = []
        else:
            self.callbacks = callbacks

        # TODO Use Tensorboard Logger and implement computational graph logging

        # Monitor is imported from schnet adapter strings because schnet names the output internally
        self.logger = loggers.TensorBoardLogger(self.logging_root, name=LOGGING_SUB_DIR)
        self.callbacks.append(
            spk.train.ModelCheckpoint(
                model_path=best_model_name,
                save_top_k=DEF_NUMBER_SAVES,
                monitor=NN_PERFORMANCE_KPI
            )
        )

    def train(self, task, data_module, epochs=None):
        if epochs is None:
            epochs = DEF_NUMBER_EPOCHS

        self.trainer = Trainer(
            log_every_n_steps=DEF_LOGGING_INTERVAL,
            callbacks=self.callbacks,
            logger=self.logger,
            default_root_dir=self.trainings_root,
            max_epochs=epochs,
        )

        self.trainer.fit(task, datamodule=data_module)

    def test(self, task, data_module):
        self.trainer = Trainer(
            log_every_n_steps=DEF_LOGGING_INTERVAL,
            callbacks=self.callbacks,
            logger=self.logger,
            default_root_dir=self.trainings_root,
            max_epochs=DEF_NUMBER_EPOCHS,
        )


        self.trainer.test(task, datamodule=data_module)

    def summarize(self) -> Dict[NNProperty, Any]:
        summary = {
            NNProperty.MAX_EPOCHS: self.trainer.max_epochs,
            NNProperty.DEVICES: self.trainer.device_ids,
            NNProperty.STRATEGY: self.trainer.strategy,
            NNProperty.PRECISION: self.trainer.precision,
        }

        return summary
