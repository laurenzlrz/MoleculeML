import time
from typing import Any, List

import pandas as pd
import psutil
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.general.SchnetAdapterStrings import METRIC_TRAIN_DESCR, METRIC_VALIDATION_DESCR, METRIC_NAME_SEPARATOR
from src.general.props.NNDefaultValue import DEF_LOG_SAVE_INTERVAL
from src.general.props.NNMetric import NNMetrics
from src.general.utils.Utility import split_string

OUTPUT_GENERAL_KEY = "general"

GIGABYTE = 1024 ** 3
GPU_NOT_AVAILABLE_DEFAULT = 0.0

FILE_FORMAT = "{path}/{name}_{phase}_{logger}.csv"


class LoggingSaver:

    def __init__(self, logger_name: str, frequency: int = DEF_LOG_SAVE_INTERVAL):
        self.frequency = frequency
        self._logger_name = logger_name

        self._batch_dicts = []
        self._epoch_dicts = []
        self._check_dicts = []

        self._batch_df = None
        self._epoch_df = None
        self._check_df = None

        self.last_step = 0
        self.all_steps = 0
        self.counter = 0

    def save_batch_dict(self, batch_dict):
        self._batch_dicts.append(batch_dict.copy())

    def save_epoch_dict(self, epoch_dict):
        self._epoch_dicts.append(epoch_dict.copy())

    def save_check_dict(self, check_dict):
        self._check_dicts.append(check_dict.copy())

    def finalise(self):
        self._batch_df = pd.DataFrame(self._batch_dicts).drop_duplicates(subset=[NNMetrics.STEP])
        self._epoch_df = pd.DataFrame(self._epoch_dicts).drop_duplicates(subset=[NNMetrics.EPOCH])
        self._check_df = pd.DataFrame(self._check_dicts)

    def check_and_increment_frequency(self, step):
        self.counter += 1

        diff = max(1, step) if self.last_step > step else step - self.last_step

        self.last_step = step
        self.all_steps += diff

        return self.counter % self.frequency == 0

    @property
    def batch_dicts(self):
        return self.batch_dicts

    @property
    def epoch_dicts(self):
        return self.epoch_dicts

    @property
    def check_dicts(self):
        return self.check_dicts

    @property
    def batch_df(self):
        return self._batch_df

    @property
    def epoch_df(self):
        return self._epoch_df

    @property
    def check_df(self):
        return self._check_df

    @property
    def logger_name(self):
        return self._logger_name


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


SAVER_NAME = "saver"


class LoggerSaverCallback(Callback, LoggingSaver):
    # Designed from Atomistic Schnet Task
    # Known from Atomistic Schnet Task: validation metrics are epoch-wise and train metrics are batch-wise
    def __init__(self, frequency=DEF_LOG_SAVE_INTERVAL,
                 measure_metrics: List[NNMetrics] = None):
        LoggingSaver.__init__(self, SAVER_NAME, frequency)
        Callback.__init__(self)
        if measure_metrics is None:
            measure_metrics = [mm for mm in NNMetrics]
        self.measure_metrics = measure_metrics

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self.batch_save(trainer, batch_idx)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.epoch_save(trainer)

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.batch_save(trainer, batch_idx)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.epoch_save(trainer)

    def batch_save(self, trainer, batch_idx):
        if not self.check_and_increment_frequency(batch_idx):
            return
        metrics = self.filter_metrics(METRIC_TRAIN_DESCR, trainer.callback_metrics)
        metrics[NNMetrics.STEP] = self.all_steps
        metrics[NNMetrics.EPOCH] = trainer.current_epoch

        self.save_batch_dict(metrics)
        self.save_check_dict(trainer.callback_metrics)

    def epoch_save(self, trainer):
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


EPOCH_METRIC_NAME = "ressources"


class EpochMetricsCallback(Callback, LoggingSaver, LoggingProxy):
    # Here individual Logging and adding to default logger

    def __init__(self, frequency=DEF_LOG_SAVE_INTERVAL):
        LoggingSaver.__init__(self, EPOCH_METRIC_NAME, frequency=frequency)
        LoggingProxy.__init__(self)
        Callback.__init__(self)
        self.epoch_dict = None
        self.epoch_start_time = None
        self.process = psutil.Process()

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start(trainer, pl_module)

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.epoch_start(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.epoch_end(trainer, pl_module)

    def epoch_end(self, trainer, pl_module):
        self.epoch_dict = {}

        self.measure_times()
        self.measure_memory()
        self.measure_gpu_memory()
        self.log_epoch_dict(self.epoch_dict, pl_module)

        self.epoch_dict[NNMetrics.EPOCH] = trainer.current_epoch
        self.save_epoch_dict(self.epoch_dict)

    def epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

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
