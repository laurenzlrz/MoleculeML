import os
import time
from typing import Any, List, Optional
import psutil

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.general.props.NNMetric import NNMetrics
from src.general.SchnetAdapterStrings import METRIC_TRAIN_DESCR, METRIC_VALIDATION_DESCR, METRIC_NAME_SEPARATOR
from src.general.props.MetricPhase import MetricDataType
from src.general.utils.Utility import split_string

OUTPUT_GENERAL_KEY = "general"

GIGABYTE = 1024 ** 3
GPU_NOT_AVAILABLE_DEFAULT = 0.0

FILE_FORMAT = "{path}/{name}_{phase}_{logger}.csv"


class LoggingSaver:

    def __init__(self, name: str, save_path: str, logger_name: str, frequency: int = 1):
        self.name = name
        self.save_path = save_path
        self.frequency = frequency
        self.logger_name = logger_name
        self.batch_dicts = []
        self.epoch_dicts = []
        self.check_dicts = []

    def save_batch_dict(self, batch_dict):
        self.batch_dicts.append(batch_dict)

    def save_epoch_dict(self, epoch_dict):
        self.epoch_dicts.append(epoch_dict)

    def save_check_dict(self, check_dict):
        self.check_dicts.append(check_dict)

    def get_epoch_dicts(self):
        return self.epoch_dicts

    def get_batch_dicts(self):
        return self.batch_dicts

    def get_check_dicts(self):
        return self.check_dicts

    def save_to_csv(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        epoch_df = self.epoch_data_to_pd()
        batch_df = self.batch_data_to_pd()
        check_df = self.check_data_to_pd()

        if len(epoch_df) > 0:
            epoch_df.to_csv(FILE_FORMAT.format(path=self.save_path, name=self.name,
                                               phase=MetricDataType.EPOCH.value, logger=self.logger_name))
        if len(batch_df) > 0:
            batch_df.to_csv(FILE_FORMAT.format(path=self.save_path, name=self.name,
                                               phase=MetricDataType.STEP.value, logger=self.logger_name))
        if len(check_df) > 0:
            check_df.to_csv(FILE_FORMAT.format(path=self.save_path, name=self.name,
                                               phase=MetricDataType.CHECK.value, logger=self.logger_name))

    def epoch_data_to_pd(self) -> Optional[pd.DataFrame]:
        return pd.DataFrame(self.get_epoch_dicts()).drop_duplicates(subset=[NNMetrics.EPOCH])

    def batch_data_to_pd(self) -> Optional[pd.DataFrame]:
        return pd.DataFrame(self.get_batch_dicts()).drop_duplicates(subset=[NNMetrics.STEP])

    def check_data_to_pd(self) -> Optional[pd.DataFrame]:
        return pd.DataFrame(self.get_check_dicts())

    def check_and_increment_frequency(self, step):
        if step % self.frequency == 0:
            self.frequency += 1
            return True
        self.frequency += 1
        return False

class LoggingProxy:

    # Adapter to fit into schnet logging practice
    def log_epoch_dict(self, epoch_dict, module):
        self.logging(epoch_dict, module, METRIC_VALIDATION_DESCR)

    def log_batch_dict(self, batch_dict, module):
        self.logging(batch_dict, module, METRIC_TRAIN_DESCR)

    def log_check_dict(self, check_dict, module):
        self.logging(check_dict, module, METRIC_TRAIN_DESCR)

    def logging(self, metric_dict, module: pl.LightningModule, phase):
        for metric_key, metric_value in metric_dict.items():
            module.log(f"{phase}{METRIC_NAME_SEPARATOR}{OUTPUT_GENERAL_KEY}"
                       f"{METRIC_NAME_SEPARATOR}{metric_key.value}",
                       metric_value, on_epoch=True, on_step=False, prog_bar=False)


class LoggerSaverCallback(Callback, LoggingSaver):
    # Designed from Atomistic Schnet Task
    # Known from Atomistic Schnet Task: validation metrics are epoch-wise and train metrics are batch-wise
    def __init__(self, name: str, save_path: str, frequency=1, measure_metrics: List[NNMetrics] = None):
        LoggingSaver.__init__(self, name, save_path, "Saver", frequency)
        Callback.__init__(self)
        if measure_metrics is None:
            measure_metrics = [mm for mm in NNMetrics]
        self.measure_metrics = measure_metrics
        self.max_steps = 0

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if batch_idx > self.max_steps:
            self.max_steps = batch_idx
        if not self.check_and_increment_frequency(batch_idx):
            return
        metrics = self.filter_metrics(METRIC_TRAIN_DESCR, trainer.callback_metrics)
        metrics[NNMetrics.STEP] = batch_idx + self.max_steps * trainer.current_epoch
        metrics[NNMetrics.EPOCH] = trainer.current_epoch

        self.save_batch_dict(metrics)
        self.save_check_dict(trainer.callback_metrics.copy())

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        metrics = self.filter_metrics(METRIC_VALIDATION_DESCR, trainer.callback_metrics)
        metrics[NNMetrics.EPOCH] = trainer.current_epoch
        self.save_epoch_dict(metrics)

    # Converts metrics entered from schnet back to enum
    def filter_metrics(self, phase, metrics):
        return_dict = {}
        for nnmetric in self.measure_metrics:
            for log_key, log_value in metrics.items():
                phase_key, output_key, metric_key = split_string(log_key, phase, nnmetric.value, METRIC_NAME_SEPARATOR)
                if metric_key is None or output_key is None or phase_key is None:
                    continue
                return_dict[NNMetrics(metric_key)] = log_value
        return return_dict


class EpochMetricsCallback(Callback, LoggingSaver, LoggingProxy):
    # Here individual Logging and adding to default logger

    def __init__(self, name: str, save_path: str, frequency=1):
        LoggingSaver.__init__(self, name, save_path, "ressources", frequency)
        LoggingProxy.__init__(self)
        Callback.__init__(self)
        self.epoch_dict = None
        self.epoch_start_time = None
        self.process = psutil.Process()

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_dict = {}

        self.measure_times()
        self.measure_memory()
        self.measure_gpu_memory()
        self.log_epoch_dict(self.epoch_dict, pl_module)

        self.epoch_dict[NNMetrics.EPOCH] = trainer.current_epoch
        self.save_epoch_dict(self.epoch_dict)

    def measure_times(self):
        epoch_times = time.time() - self.epoch_start_time
        self.epoch_dict[NNMetrics.EPOCH_TIME] = epoch_times

    def measure_memory(self):
        memory_info = self.process.memory_info()
        self.epoch_dict[NNMetrics.EPOCH_MEMORY] = memory_info.rss / GIGABYTE

    def measure_gpu_memory(self):
        if torch.cuda.is_available():
            self.epoch_dict[NNMetrics.EPOCH_GPU_MEMORY] = torch.cuda.memory_allocated() / GIGABYTE
        else:
            self.epoch_dict[NNMetrics.EPOCH_GPU_MEMORY] = GPU_NOT_AVAILABLE_DEFAULT

    def get_epoch_dicts(self):
        return self.epoch_dicts

    def finalize(self):
        self.save_to_csv()
