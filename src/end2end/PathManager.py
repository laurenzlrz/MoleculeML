import os
from re import escape, search, match

from src.general.props import NNDefaultValue
from src.general.props.NNDefaultValue import DEF_DIRECTORY_BREAK

PATH_FORMAT = "{path}/{name}"
NO_FREE_DIRECTORY = "No free directory found until break {def_break}"
VERSION_FORMAT = "{name}_{i}{type}"
SUBDIRECTORY_FORMAT = "{all_model_root}/{name}_{i}{type}"
REGEX_TEMPLATE = r"{name}_([\d]+)"
DIR_EXISTS = "Subdirectory {name} already exists"
FILE_NOT_FOUND = "No version found for {name} at path {path}"
CSV_FORMAT = "{path}/{name}.csv"
PNG_FORMAT = "{path}/{name}.png"

DEF_FREQUENCY = NNDefaultValue.DEF_LOG_SAVE_INTERVAL

LOGGING_SUB_DIR = "logger"
TRAINING_SUB_DIR = "training"
DATA_METRICS_SUB_DIR = "data_metrics"

EPOCH_PREFIX = "{name}_epoch"
BATCH_PREFIX = "{name}_batch"
CHECK_PREFIX = "{name}_check"
DEF_COMPARE_NAME = "model_comparison_{run_name}"
MODEL_PREFIX = "model"

LOGGING_PATH = "{path}/logging"
LOGGING_SUB_NAMES = "version"
FIGURE_SUB_DIR = "figures"

CSV_TYPE = ".csv"
PNG_TYPE = ".png"
SPLIT_TYPE = ".npz"

SPLIT_FILE_NAME = "{train_run}_split"
NAME_IN_SUMMARY_COL = "name"

DEF_LOGGER_NAME = "logger"


# TODO: Right now: Creating training package is responsibility of the ModelRun, in the future trainers and models
#  should be passed so that one trainer can train multiple models and vice versa

class PathManager:
    """
    Manages one directory on one level of the file system.
    """

    def __init__(self, root, makedir=True):
        if makedir and not os.path.exists(root):
            os.makedirs(root)
        self.root = root
        self.sub_dirs = {}

    def get_subdirectory(self, name) -> str:
        if PATH_FORMAT.format(path=self.root, name=name) not in self.sub_dirs:
            raise FileNotFoundError(FILE_NOT_FOUND.format(name=name, path=self.root))
        return PATH_FORMAT.format(path=self.root, name=name)

    def create_subdirectory(self, name) -> str:
        directory = PATH_FORMAT.format(path=self.root, name=name)
        if os.path.exists(directory):
            raise FileExistsError(DIR_EXISTS.format(name=name))
        os.makedirs(directory)
        return directory

    def create_version_subdirectory(self, name) -> str:
        directory = self.get_free_version_path(name)
        os.makedirs(directory)
        return directory

    def get_version_subdirectory(self, name) -> str:
        path = self.get_highest_version_subdirectory(name)
        return path

    def get_highest_version_subdirectory(self, name) -> str:
        pattern = REGEX_TEMPLATE.format(name=escape(name))
        subdirs = {int(search(pattern, sub).group(1)): sub for sub in os.listdir(self.root) if match(pattern, sub)}

        if len(subdirs.keys()) == 0:
            raise FileNotFoundError(FILE_NOT_FOUND.format(name=name, path=self.root))

        highest = subdirs[max(subdirs.keys())] if subdirs else None
        return PATH_FORMAT.format(path=self.root, name=highest)

    def get_free_version_name(self, name, file_format="") -> str:
        for i in range(DEF_DIRECTORY_BREAK):
            file_name = VERSION_FORMAT.format(name=name, i=i, type=file_format)
            model_dir = PATH_FORMAT.format(path=self.root, name=file_name)
            if not os.path.exists(model_dir):
                return file_name
        raise FileExistsError(NO_FREE_DIRECTORY.format(def_break=DEF_DIRECTORY_BREAK))

    def get_free_version_path(self, name, file_format="") -> str:
        for i in range(DEF_DIRECTORY_BREAK):
            model_dir = SUBDIRECTORY_FORMAT.format(all_model_root=self.root, name=name, i=i, type=file_format)
            if not os.path.exists(model_dir):
                return model_dir
        raise FileExistsError(NO_FREE_DIRECTORY.format(def_break=DEF_DIRECTORY_BREAK))

    def save_df(self, name, data, versionize=False):
        if versionize:
            path = self.get_free_version_path(name, CSV_TYPE)
            data.to_csv(path)
            return

        path = CSV_FORMAT.format(path=self.root, name=name)
        if os.path.exists(path):
            raise FileExistsError(DIR_EXISTS.format(name=name))
        data.to_csv(path)

    def save_fig(self, name, figure, versionize=False):
        if versionize:
            path = self.get_free_version_path(name, PNG_TYPE)
            figure.savefig(path)
            return

        path = PNG_FORMAT.format(path=self.root, name=name)
        if os.path.exists(path):
            raise FileExistsError(DIR_EXISTS.format(name=name))
        figure.savefig(path)

    def clear(self):
        for sub in self.sub_dirs:
            os.rmdir(sub)


