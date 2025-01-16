from pytorch_lightning import loggers, Trainer

from src.training.LoggerCallback import LoggerSaverCallback, EpochMetricsCallback
from src.training.SchnetTrainer import SchnetTrainer
from src.end2end.PathManager import PathManager, DEF_FREQUENCY

TB_LOGGER_NAME = 'tb_logger'
TRAINING_DIR_NAME = 'training'

class TrainerPaths:

    def __init__(self, root, train_run_name):
        self._root = root
        self.train_run_name = train_run_name

        self._path_manager = PathManager(root)
        self._logger_directory = self._path_manager.get_free_version_name(TB_LOGGER_NAME)
        self._logger_path = self.root
        self._training_directory = self._path_manager.get_free_version_path(TRAINING_DIR_NAME)


    @property
    def root(self):
        return self._root

    @property
    def logger_path(self):
        return self._logger_path

    @property
    def logger_directory(self):
        return self._logger_directory

    @property
    def training_directory(self):
        return self._training_directory

class TrainingBundle:

    def __init__(self):
        self._model = None
        self._task = None
        self._data_module = None

    @property
    def model(self):
        return self._model

    @property
    def task(self):
        return self._task

    @property
    def data_module(self):
        return self._data_module

    def get_tb_loader(self):

class DataProxy:

    def __init__(self, callbacks, tb_logger, module, phase):
        self.callbacks = callbacks
        self.tb_logger_path = tb_logger
        self.module = module
        self.phase = phase

    def load_callback_data(self):
        self._cb_epoch_data = {callback: callback.epoch_df[self.module][self.phase]
                               for callback in self.callbacks if len(callback.epoch_df) > 0}
        self._cb_batch_data = {callback: callback.batch_df[self.module][self.phase]
                               for callback in self.callbacks if len(callback.batch_df) > 0}
        self._cb_check_data = {callback: callback.check_df[self.module][self.phase]
                               for callback in self.callbacks if len(callback.check_df) > 0}
        self._

class TrainerManager:

    def __init__(self, trainer_paths: TrainerPaths):
        self.trainer_paths = trainer_paths

        self.trainer: SchnetTrainer = None
        self.logging_callbacks = None

    def load_trainer(self):
        self.load_logging_callbacks()
        logger_path = self.trainer_paths.logger_path
        logger_directory = self.trainer_paths.logger_directory
        training_directory = self.trainer_paths.training_directory

        self.trainer = SchnetTrainer(training_directory, logger_path, logger_directory, self.logging_callbacks)

    def load_logging_callbacks(self):
        logger_saver_callback = LoggerSaverCallback(DEF_FREQUENCY)
        epoch_metrics_callback = EpochMetricsCallback(DEF_FREQUENCY)
        self.logging_callbacks = [logger_saver_callback, epoch_metrics_callback]

    def train(self, training_bundle: TrainingBundle):
        self.trainer.train(training_bundle.task, training_bundle.data_module)



