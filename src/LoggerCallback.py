import time
from typing import Any, List

import pandas as pd
import psutil
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.general.NNMetrics import NNMetrics
# TODO: Add custom Atomistic Task to enhance Logging (Inheritance facilitates this)

from src.general.Utility import split_string

EPOCH_LOGGING_KEY = "epoch_logging"
BATCH_LOGGING_KEY = "batch_logging"
SEPARATOR = "_"
TRAIN_PHASE = "train"
VALIDATION_PHASE = "val"

TRAIN_SAVE_PATH = "train_metrics.csv"
VALIDATION_SAVE_PATH = "validation_metrics.csv"


class LoggerSaverCallback(Callback):
    # Designed from Atomistic Schnet Task
    # Known from Atomistic Schnet Task: validation metrics are epoch-wise and train metrics are batch-wise
    def __init__(self, measure_metrics: List[NNMetrics]):
        super().__init__()
        self.measure_metrics = measure_metrics
        self.train_dicts = []
        self.validation_dicts = []

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        metrics = self.filter_metrics(TRAIN_PHASE, trainer.callback_metrics)
        metrics[BATCH_LOGGING_KEY] = batch_idx
        metrics[EPOCH_LOGGING_KEY] = trainer.current_epoch
        self.train_dicts.append(metrics)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        metrics = self.filter_metrics(VALIDATION_PHASE, trainer.callback_metrics)
        metrics[EPOCH_LOGGING_KEY] = trainer.current_epoch
        self.validation_dicts.append(metrics)

    def filter_metrics(self, phase, metrics):
        return_dict = {}
        for nnmetric in self.measure_metrics:
            for log_key, log_value in metrics.items():
                metric_key, output_key, phase_key = split_string(log_key, phase, nnmetric.value, SEPARATOR)
                if metric_key is None or output_key is None or phase_key is None:
                    continue
                return_dict[NNMetrics(metric_key)] = log_value
        return return_dict

    def get_train_dicts(self):
        return self.train_dicts

    def get_validation_dicts(self):
        return self.validation_dicts

    def save_to_csv(self):
        train_df = pd.DataFrame(self.get_train_dicts())
        validation_df = pd.DataFrame(self.get_validation_dicts())
        train_df.to_csv(TRAIN_SAVE_PATH)
        validation_df.to_csv(VALIDATION_SAVE_PATH)


GIGABYTE = 1024 ** 3
GPU_NOT_AVAILABLE_DEFAULT = 0.0


class EpochMetricsCallback(Callback):
    # Here individual Logging and adding to default logger

    def __init__(self):
        super().__init__()
        self.logging_dict = {}
        self.logging_dicts = []
        self.log = None
        self.epoch_start_time = None
        self.process = psutil.Process()

    def on_train_epoch_start(self, trainer, pl_module):
        self.log = pl_module.log
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        self.log = pl_module.log
        self.logging_dict = {EPOCH_LOGGING_KEY: trainer.current_epoch}
        self.measure_times()
        self.measure_memory()
        self.measure_gpu_memory()
        self.logging_dicts.append(self.logging_dict)

    def epoch_logging(self, key, value):
        self.log(key, value, on_epoch=True, on_step=False, prog_bar=False)
        self.logging_dict[key.value] = value

    def measure_times(self):
        epoch_times = time.time() - self.epoch_start_time
        self.logging_dict[NNMetrics.EPOCH_TIME] = epoch_times

    def measure_memory(self):
        memory_info = self.process.memory_info()
        self.logging_dict[NNMetrics.EPOCH_MEMORY] = memory_info.used / GIGABYTE

    def measure_gpu_memory(self):
        if torch.cuda.is_available():
            self.logging_dict[NNMetrics.EPOCH_GPU_MEMORY] = torch.cuda.memory_allocated() / GIGABYTE
        else:
            self.logging_dict[NNMetrics.EPOCH_GPU_MEMORY] = GPU_NOT_AVAILABLE_DEFAULT

    def save_to_csv(self):
        df = pd.DataFrame(self.logging_dicts)
        df.to_csv("epoch_metrics.csv")