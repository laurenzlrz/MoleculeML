from src.end2end.BulkRun import TrainRun
from src.end2end.TrainFileManager import TrainFileManager
from src.training.ModelConfig import ModelBuilder


class E2E:

    def __init__(self, models: dict[str, ModelBuilder],
                 root, train_run_name, train_db_path, train_db_name, test_db_path, test_db_name,
                 used_props: list, batch_size, num_val, num_train):
        self.models = models

        self.root = root
        self.train_run_name = train_run_name
        self.db_path = train_db_path
        self.db_name = train_db_name
        self.train_saver = TrainFileManager(root, train_run_name, train_db_path,
                                            train_db_name, test_db_path, test_db_name)

        self.train_run = TrainRun(self.train_saver, used_props, batch_size, num_val, num_train)

    def process(self):
        self.train_run.init_db_manager()
        self.train_run.load_models(self.models)
        self.train_run.process()
        self.train_run.compare()
