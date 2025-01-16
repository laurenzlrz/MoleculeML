import time
from typing import Any, List

import pandas as pd
import psutil
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.general.props.MetricPhase import MetricDataType
from src.general.SchnetAdapterStrings import METRIC_TRAIN_DESCR, METRIC_VALIDATION_DESCR, METRIC_NAME_SEPARATOR
from src.general.props.NNDefaultValue import DEF_LOG_SAVE_INTERVAL
from src.general.props.NNMetric import NNMetrics
from src.general.utils.Utility import split_string

OUTPUT_GENERAL_KEY = "general"

GIGABYTE = 1024 ** 3
GPU_NOT_AVAILABLE_DEFAULT = 0.0

FILE_FORMAT = "{path}/{name}_{phase}_{logger}.csv"

EPOCH_PHASE = MetricDataType.EPOCH
BATCH_PHASE = MetricDataType.BATCH


class LoggingSaver:

    def __init__(self, logger_name: str, frequency: int = DEF_LOG_SAVE_INTERVAL):
        self.frequency = frequency
        self._logger_name = logger_name

        self._batch_dicts = {}
        self._epoch_dicts = {}
        self._check_dicts = {}

        self._batch_df = {}
        self._epoch_df = {}
        self._check_df = {}

        self.last_step = {}
        self.all_steps = {}
        self.counter = {}

    def save_batch_dict(self, batch_dict, pl, phase):
        key = (pl, phase)
        if key not in self._batch_dicts:
            self._batch_dicts[key] = []

        self._batch_dicts[key].append(batch_dict.copy())

    def save_epoch_dict(self, epoch_dict, pl, phase):
        key = (pl, phase)
        if key not in self._epoch_dicts:
            self._epoch_dicts[key] = []

        self._epoch_dicts[key].append(epoch_dict.copy())

    def save_check_dict(self, check_dict, pl, phase):
        key = (pl, phase)
        if key not in self._check_dicts:
            self._check_dicts[key] = []

        self._check_dicts[key].append(check_dict.copy())

    def finalise(self, pl, phase):
        key = (pl, phase)

        self._batch_df[key] = pd.DataFrame(self._batch_dicts[key]).drop_duplicates(subset=[NNMetrics.STEP])
        self._epoch_df[key] = pd.DataFrame(self._epoch_dicts[key]).drop_duplicates(subset=[NNMetrics.EPOCH])
        self._check_df[key] = pd.DataFrame(self._check_dicts[key])

    def check_and_increment_frequency(self, step, pl, phase):
        key = (pl, phase)
        if key not in self.counter:
            self.counter[key] = 0
            self.last_step[key] = 0
            self.all_steps[key] = 0

        self.counter[key] += 1

        diff = max(1, step) if self.last_step[key] > step else step - self.last_step[key]
        self.last_step[key] = step
        self.all_steps[key] += diff

        return self.counter[key] % self.frequency == 0

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
        if not self.check_and_increment_frequency(batch_idx, pl):
            return
        metrics = self.filter_metrics(METRIC_TRAIN_DESCR, trainer.callback_metrics)
        metrics[NNMetrics.STEP] = self.all_steps
        metrics[NNMetrics.EPOCH] = trainer.current_epoch

        self.save_batch_dict(metrics, pl, BATCH_PHASE)
        self.save_check_dict(trainer.callback_metrics)

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

"""
class LoggingSaver:

    def __init__(self, logger_name: str, frequency: int = DEF_LOG_SAVE_INTERVAL):
        self.frequency = frequency
        self._logger_name = logger_name

        self._batch_dicts = {}
        self._epoch_dicts = {}
        self._check_dicts = {}

        self._batch_df = {}
        self._epoch_df = {}
        self._check_df = {}

        self.last_step = {}
        self.all_steps = {}
        self.counter = {}

    def save_batch_dict(self, batch_dict, pl, phase):
        if pl not in self._batch_dicts:
            self._batch_dicts[pl] = {}

        if not phase in self._batch_dicts[pl]:
            self._batch_dicts[pl][phase] = []

        self._batch_dicts[pl][phase].append(batch_dict.copy())

    def save_epoch_dict(self, epoch_dict, pl, phase):
        if pl not in self._epoch_dicts:
            self._epoch_dicts[pl] = {}

        if not phase in self._epoch_dicts[pl]:
            self._epoch_dicts[pl][phase] = []

        self._epoch_dicts[pl][phase].append(epoch_dict.copy())

    def save_check_dict(self, check_dict, pl, phase):
        if pl not in self._check_dicts:
            self._check_dicts[pl] = {}

        if not phase in self._check_dicts[pl]:
            self._check_dicts[pl][phase] = []

        self._check_dicts[pl][phase].append(check_dict.copy())

    def finalise(self, pl, phase):
        if pl not in self._batch_dicts:
            self._batch_dicts[pl] = {}
        if pl not in self._epoch_dicts:
            self._epoch_dicts[pl] = {}
        if pl not in self._check_dicts:
            self._check_dicts[pl] = {}

        self._batch_df[pl][phase] = pd.DataFrame(self._batch_dicts[pl]).drop_duplicates(subset=[NNMetrics.STEP])
        self._epoch_df[pl][phase] = pd.DataFrame(self._epoch_dicts[pl]).drop_duplicates(subset=[NNMetrics.EPOCH])
        self._check_df[pl][phase] = pd.DataFrame(self._check_dicts[pl])

    def check_and_increment_frequency(self, step, pl, phase):
        if pl not in self.counter:
            self.counter[pl] = {}
            self.last_step[pl] = {}
            self.all_steps[pl] = {}
        
        if phase not in self.counter[pl]:
            self.counter[pl][phase] = 0
        
        if phase not in self.last_step[pl]:
            self.last_step[pl][phase] = 0
            
        if phase not in self.all_steps[pl]:
            self.all_steps[pl][phase] = 0
            
        self.counter[pl][phase] += 1
        

        diff = max(1, step) if self.last_step[pl][phase] > step else step - self.last_step[pl][phase]

        self.last_step[pl][phase] = step
        self.all_steps += diff

        return self.counter[pl][phase] % self.frequency == 0
        """