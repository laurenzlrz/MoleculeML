import pandas as pd

from src.end2end.PathManager import DEF_FREQUENCY
from src.general.utils import VisualisationUtility
from src.training.LoggerCallback import LoggerSaverCallback, EpochMetricsCallback
from src.training.SchnetTrainPackage import SchnetTrainPackage
from src.training.SchnetTrainer import SchnetTrainer
from src.visualisation.TrainingVisualisation import TrainingVisualisation


class ModelRun:
    """
    Responsible for the holistic training of a model from data loading to saving the model and logging data and
    visualising.
    """

    # TODO New Creation Methods (Directly from model e.g.)
    def __init__(self):
        self.saver = None

        self.callbacks = None
        self.schnet_train_wrapper = None

        self._lg_batch_data = None
        self._lg_epoch_data = None
        self.model_data = None
        self.figures = None
        self._cb_check_data = None
        self._cb_batch_data = None
        self._cb_epoch_data = None

    # Can be used to update the saver and make a new run with the same configuration to another saver
    def reload_saver(self, saver):
        self.saver = saver

    def reload_package_and_fill_trainer(self, wrapper: SchnetTrainPackage):
        logger_saver_callback = LoggerSaverCallback(DEF_FREQUENCY)
        epoch_metrics_callback = EpochMetricsCallback(DEF_FREQUENCY)
        self.callbacks = [logger_saver_callback, epoch_metrics_callback]

        trainer = SchnetTrainer(self.saver.train_dir, self.saver.logg_dir, self.callbacks.copy())
        wrapper.set_trainer(trainer)
        self.schnet_train_wrapper = wrapper

    def train_process(self):
        self.train()
        self.finalize_train()
        self.load_stats()
        self.visualize()
        self.save_stats()

    def test_process(self):
        self.test()
        self.finalize_train()
        self.load_stats()
        self.visualize()
        self.save_stats()

    def train(self):
        self.schnet_train_wrapper.train()

    def test(self):
        self.schnet_train_wrapper.test()

    def finalize_train(self):
        [callback.finalise() for callback in self.callbacks]

    def load_stats(self):
        self._cb_epoch_data = {callback: callback.epoch_df for callback in self.callbacks if len(callback.epoch_df) > 0}
        self._cb_batch_data = {callback: callback.batch_df for callback in self.callbacks if len(callback.batch_df) > 0}
        self._cb_check_data = {callback: callback.check_df for callback in self.callbacks if len(callback.check_df) > 0}

        self.schnet_train_wrapper.summarize()
        self.model_data = self.schnet_train_wrapper.model_data

        self.saver.reload_logging()
        self._lg_batch_data = self.saver.load_logger_batch_data
        self._lg_epoch_data = self.saver.load_logger_epoch_data

    def visualize(self):
        training_vis = TrainingVisualisation(axis_scaling=VisualisationUtility.scale_axis_to_zero)
        training_vis.set_epoch_data(self._lg_epoch_data)
        self.figures = training_vis.print_epochs()

    def save_stats(self):
        self.saver.save_saver_logs(self._cb_epoch_data, self._cb_batch_data, self._cb_check_data, self.model_data)

        self.saver.save_epoch_figures(self.figures)

        self.saver.save_logger_logs(self._lg_epoch_data, self._lg_batch_data)

    def get_summary(self):
        model_last_epoch = self._lg_epoch_data.iloc[-1]
        return pd.concat([self.model_data, model_last_epoch])
