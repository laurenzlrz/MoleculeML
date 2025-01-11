import os

import pandas as pd
import schnetpack.transform as trn

from src.general.props.MolProperty import MolProperty
from src.general.props import NNDefaultValue
from src.general.utils import VisualisationUtility
from src.training.SchnetTrainer import SchnetTrainer
from src.training.SchnetNN import SchnetNN, AdditionSchnetNN
from src.training.LoggerCallback import EpochMetricsCallback, LoggerSaverCallback
from src.training.GeometrySchnetDB import GeometrySchnetDB
from src.visualisation.TrainingVisualisation import VisualisationTensorboardLoader, TrainingVisualisation

CONSTRUCTOR_KEY = "constructor"
PREDICTION_KEYS_KEY = "prediction_keys"
ADDITIONAL_INPUT_KEYS_KEY = "additional_input_keys"
NAME_ARG = "name"

INVALID_CONSTRUCTOR = "The 'constructor' must be a callable function or class"
NOT_A_SCHNET_MODEL = "The 'constructor' must be a model class or function"
NO_FREE_DIRECTORY = "No free directory found until break {def_break}"

MODEL_ROOT_FORMAT = "{all_model_root}/{name}_{i}"
COMPARATOR_ROOT_FORMAT = "{all_model_root}/comparisons_{i}"
MODEL_STAT_PATH = "{path}/model_stats.csv"
CALLBACK_PATH = "{path}/metrics"
TRAINER_PATH = "{path}/trainer"
LOGGER_PATH = "{path}/logger"
REAL_LOGGER_PATH = "{path}/logger/logging/version_{i}"  # Subdirectory for the logger from the trainer

DEF_FREQUENCY = NNDefaultValue.DEF_LOG_SAVE_INTERVAL
DEF_BREAK = NNDefaultValue.DEF_DIRECTORY_BREAK

DEF_DB_NAME = "db"

DEF_COMPARISON_PATH = "{path}/comparisons_{i}.csv"


class MultiTrainOrganizer:

    def __init__(self, dataset_name, db_project_root, train_project_root, used_props, batch_size, num_val, num_train):
        self.dataset_name = dataset_name
        self.db_project_root = db_project_root
        self.train_project_root = train_project_root
        self.db = GeometrySchnetDB.load_existing(dataset_name, db_project_root)
        self.comparator = Comparator(self.train_project_root)

        self.transforms = [
            trn.ASENeighborList(cutoff=5.),
            trn.RemoveOffsets(MolProperty.TOTAL_ENERGY.value, remove_mean=True, remove_atomrefs=False),
            trn.CastTo32()
        ]

        self.module = self.db.create_schnet_module(selected_properties=used_props, batch_size=batch_size,
                                                   num_val=num_val, num_train=num_train,
                                                   transforms=self.transforms, split_path=None)
        self.organizers = []

    def new_TrainOrganizer(self, name):
        directory = self.get_free_directory(name)
        train_orga = TrainOrganizer(name, directory)
        train_orga.load_module(self.module, self.db.get_attribute_dimensions())
        self.organizers.append(train_orga)
        return train_orga

    def get_free_directory(self, name):
        for i in range(DEF_BREAK):
            model_dir = MODEL_ROOT_FORMAT.format(all_model_root=self.train_project_root, name=name, i=i)
            if not os.path.exists(model_dir):
                return model_dir
        raise FileExistsError(NO_FREE_DIRECTORY.format(def_break=DEF_BREAK))

    def print_comparison(self):
        self.comparator.save_model_stats(self.organizers)


class Comparator:

    def __init__(self, directory):
        self.directory = directory

    def get_free_directory(self):
        for i in range(DEF_BREAK):
            model_dir = COMPARATOR_ROOT_FORMAT.format(all_model_root=self.directory, i=i)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                return model_dir
        raise FileExistsError(NO_FREE_DIRECTORY.format(def_break=DEF_BREAK))

    def save_model_stats(self, organizers):
        compare_rows = []
        for organizer in organizers:
            model_summary = pd.Series(organizer.get_model_stats())  # df
            model_last_epoch = organizer.get_epoch_stats().iloc[-1]  # series
            compare_row = pd.concat([model_summary, model_last_epoch])  # komisches
            compare_rows.append(compare_row)

        path = self.get_free_directory()

        pd.DataFrame(compare_rows).to_csv(MODEL_STAT_PATH.format(path=path))


class TrainOrganizer:

    def load_module(self, module, propdims):
        self.module = module
        self.propdims = propdims

    def load_model(self, model):
        self.model = model

    def create_schnet_model_from_dict(self, data):
        constructor = data.get(CONSTRUCTOR_KEY)
        if not callable(constructor):
            raise ValueError(INVALID_CONSTRUCTOR)

        constructor_args = {k: v for k, v in data.items() if k != CONSTRUCTOR_KEY}

        # Add units to the properties, because it makes defining the units easier
        # (ideally we would have the models created after the db was loaded)
        constructor_args[ADDITIONAL_INPUT_KEYS_KEY] = {prop: self.propdims[prop]
                                                       for prop in constructor_args[ADDITIONAL_INPUT_KEYS_KEY]}
        constructor_args[PREDICTION_KEYS_KEY] = {prop: self.propdims[prop]
                                                 for prop in constructor_args[PREDICTION_KEYS_KEY]}

        model = constructor(**constructor_args)
        if not isinstance(model, (SchnetNN, AdditionSchnetNN)):
            raise ValueError(NOT_A_SCHNET_MODEL)
        self.model = model

    def create_trainer(self):
        callback_path = CALLBACK_PATH.format(path=self.this_model_root)
        self.logger_saver_callback = LoggerSaverCallback(self.name, callback_path, DEF_FREQUENCY)
        self.epoch_metrics_callback = EpochMetricsCallback(self.name, callback_path, DEF_FREQUENCY)
        callbacks = [self.logger_saver_callback, self.epoch_metrics_callback]

        self.logger_path = LOGGER_PATH.format(path=self.this_model_root)
        self.trainer_path = TRAINER_PATH.format(path=self.this_model_root)
        self.trainer = SchnetTrainer(self.trainer_path, self.logger_path, self.name, callbacks)

    def train(self):
        task = self.model.build_and_return_task()
        self.trainer.train(task, self.module)
        self.logger_saver_callback.save_to_csv()
        self.epoch_metrics_callback.save_to_csv()

    def load_stats(self):
        tensorboard_loader = VisualisationTensorboardLoader()
        current_log_dir = self.newest_logg_dir()
        tensorboard_loader.load_from_file(current_log_dir)
        self.epoch_stats = tensorboard_loader.get_epoch_data()
        self.batch_stats = tensorboard_loader.get_batch_data()

        model_stats = self.model.summary()
        trainer_stats = self.trainer.summarize()
        self.model_stats = {**model_stats, **trainer_stats}
        pd.DataFrame(self.model_stats).to_csv(MODEL_STAT_PATH.format(path=self.this_model_root))

    def visualize(self):
        training_vis = TrainingVisualisation(self.this_model_root, axis_scaling=VisualisationUtility.scale_axis_to_zero)
        training_vis.set_epoch_data(self.epoch_stats)
        training_vis.print_epochs()

    def newest_logg_dir(self):
        for i in range(DEF_BREAK):
            path = REAL_LOGGER_PATH.format(path=self.this_model_root, i=i)
            if not os.path.exists(path):
                return REAL_LOGGER_PATH.format(path=self.this_model_root, i=i - 1)

    def __init__(self, name, this_model_root):
        self.this_model_root = this_model_root
        self.name = name

        self.trainer_path = None
        self.logger_path = None
        self.epoch_metrics_callback = None
        self.logger_saver_callback = None

        self.propdims = None
        self.module = None

        self.model = None
        self.trainer = None

        self.epoch_stats = None
        self.batch_stats = None
        self.model_stats = None

    def get_epoch_stats(self):
        return self.epoch_stats

    def get_batch_stats(self):
        return self.batch_stats

    def get_model_stats(self):
        return self.model_stats

    def get_name(self):
        return self.name


class E2ETraining:

    def __init__(self, dataset_name, db_project_root, train_project_root, used_props, batch_size, num_val, num_train):
        self.multi_train_organizer = MultiTrainOrganizer(dataset_name, db_project_root, train_project_root, used_props,
                                                         batch_size, num_val, num_train)

    def iterate_pipeline(self, nn_models):
        for model in nn_models:
            self.pipeline(model)
        self.multi_train_organizer.print_comparison()

    def pipeline(self, nn_model):
        organizer = self.multi_train_organizer.new_TrainOrganizer(nn_model[NAME_ARG])
        organizer.create_schnet_model_from_dict(nn_model)
        organizer.create_trainer()
        organizer.train()
        organizer.load_stats()
        organizer.visualize()

