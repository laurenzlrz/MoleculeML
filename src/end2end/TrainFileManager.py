from src.end2end.PathManager import PathManager, SPLIT_FILE_NAME, SPLIT_TYPE, DEF_COMPARE_NAME
from src.end2end.ModelFileManager import ModelFileManager
from src.training.GeometrySchnetDB import GeometrySchnetDB


class TrainFileManager:
    """
    Responsible for handling os file transfers on one training run level.
    Defines the structure of the files needed and created during the training process.
    """

    def __init__(self, root, train_run_name, train_db_path, train_db_name, test_db_path, test_db_name):
        self.root = root
        self.name = train_run_name

        self.train_db_root = train_db_path
        self.train_db_name = train_db_name

        self.test_db_root = test_db_path
        self.test_db_name = test_db_name

        self.path_manager = PathManager(root)
        self._split_file_path = self.path_manager.get_free_version_path(
            SPLIT_FILE_NAME.format(train_run=train_run_name),
            SPLIT_TYPE)

    def load_train_db(self):
        return GeometrySchnetDB.load_existing(self.test_db_name, self.train_db_root)

    def load_test_db(self):
        return GeometrySchnetDB.load_existing(self.test_db_name, self.test_db_root)

    def create_model_run_saver(self, model_name):
        model_dir = self.path_manager.create_version_subdirectory(model_name)
        return ModelFileManager(model_dir, model_name)

    def save_train_comparison(self, compare_df):
        self.path_manager.save_df(DEF_COMPARE_NAME.format(run_name=self.name), compare_df, versionize=True)

    def save_test_comparison(self, compare_df):
        self.path_manager.save_df(DEF_COMPARE_NAME.format(run_name=(self.name + "test")), compare_df, versionize=True)

    @property
    def split_file_path(self):
        return self._split_file_path
