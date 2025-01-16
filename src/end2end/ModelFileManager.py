from src.end2end.PathManager import PathManager, LOGGING_SUB_DIR, TRAINING_SUB_DIR, DATA_METRICS_SUB_DIR, \
    FIGURE_SUB_DIR, EPOCH_PREFIX, BATCH_PREFIX, CHECK_PREFIX, MODEL_PREFIX, DEF_LOGGER_NAME, LOGGING_PATH, \
    LOGGING_SUB_NAMES
from src.visualisation.TrainingVisualisation import VisualisationTensorboardLoader


class ModelFileManager:
    """
    Outsourcing the path structure
    Responsible for handling os file transfers on one model training level.
    """

    def __init__(self, root, model_name):
        self.root = root
        self.name = model_name
        self.path_manager = PathManager(root)
        self.path_manager.clear()

        self._logg_dir = self.path_manager.create_subdirectory(LOGGING_SUB_DIR)
        self._train_dir = self.path_manager.create_subdirectory(TRAINING_SUB_DIR)
        self._data_dir = self.path_manager.create_subdirectory(DATA_METRICS_SUB_DIR)
        self._figure_dir = self.path_manager.create_subdirectory(FIGURE_SUB_DIR)

        self.tb_loader = VisualisationTensorboardLoader()

    @property
    def logg_dir(self):
        return self._logg_dir

    @property
    def train_dir(self):
        return self._train_dir

    @property
    def data_dir(self):
        return self._data_dir

    def save_saver_logs(self, epoch_data, batch_data, check_data, model_data):
        path_manager = PathManager(self.data_dir)
        [path_manager.save_df(EPOCH_PREFIX.format(name=callback.logger_name), data)
         for callback, data in epoch_data.items()]
        [path_manager.save_df(BATCH_PREFIX.format(name=callback.logger_name), data)
         for callback, data in batch_data.items()]
        [path_manager.save_df(CHECK_PREFIX.format(name=callback.logger_name), data)
         for callback, data in check_data.items()]

        path_manager.save_df(MODEL_PREFIX.format(self.name), model_data)

    def save_logger_logs(self, epoch_data, batch_data):
        path_manager = PathManager(self.data_dir)
        path_manager.save_df(EPOCH_PREFIX.format(name=DEF_LOGGER_NAME), epoch_data)
        path_manager.save_df(BATCH_PREFIX.format(name=DEF_LOGGER_NAME), batch_data)

    def reload_logging(self):
        path_manager = PathManager(LOGGING_PATH.format(path=self.logg_dir))
        load_path = path_manager.get_highest_version_subdirectory(LOGGING_SUB_NAMES)
        self.tb_loader.load_from_file(load_path)

    def save_epoch_figures(self, figures):
        path_manager = PathManager(self._figure_dir)
        for metric, figure in figures.items():
            name = EPOCH_PREFIX.format(name=metric.value)
            path_manager.save_fig(name, figure)

    @property
    def load_logger_batch_data(self):
        return self.tb_loader.get_batch_data()

    @property
    def load_logger_epoch_data(self):
        return self.tb_loader.get_epoch_data()
