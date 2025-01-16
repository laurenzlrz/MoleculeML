from typing import Optional, Dict

import pandas as pd
from schnetpack import transform as trn

from src.end2end.PathManager import NAME_IN_SUMMARY_COL
from src.end2end.ModelRun import ModelRun
from src.end2end.TrainFileManager import TrainFileManager
from src.training.ModelConfig import ModelBuilder
from src.training.SchnetTrainPackage import SchnetTrainPackage


class TrainRun:
    # TODO: Implement multiple modules
    def __init__(self, train_saver: TrainFileManager, used_props, batch_size, num_val, num_train):
        self.saver = train_saver

        self.used_props = used_props
        self.batch_size = batch_size
        self.num_val = num_val
        self.num_train = num_train

        self.model_runs: Optional[Dict[str, ModelRun]] = None
        self.model_Tests = None

        self._train_db_manager = None
        self._test_db_manager = None
        self._train_module = None
        self._test_module = None

    def init_db_manager(self):
        self._train_db_manager = self.saver.load_train_db()
        transforms = [
            trn.ASENeighborList(cutoff=5.),
            trn.CastTo32()
        ]
        self._train_module = self._train_db_manager.create_schnet_module(selected_properties=self.used_props,
                                                                         batch_size=self.batch_size,
                                                                         num_val=self.num_val, num_train=self.num_train,
                                                                         transforms=transforms,
                                                                         split_path=self.saver.split_file_path,
                                                                         pin_memory=True,
                                                                         num_workers=4)

        self._test_db_manager = self.saver.load_test_db()

        transforms = [
            trn.ASENeighborList(cutoff=5.),
            trn.CastTo32()
        ]
        self._test_db_manager = self._test_db_manager.create_schnet_module(selected_properties=self.used_props,
                                                                           batch_size=self.batch_size,
                                                                           num_val=self.num_val,
                                                                           num_train=self.num_train,
                                                                           transforms=transforms,
                                                                           split_path=self.saver.split_file_path,
                                                                           pin_memory=True,
                                                                           num_workers=4)

        print(self._train_db_manager)
        print(self._test_db_manager)

    def load_models(self, builds: dict[str, ModelBuilder]):
        self.model_runs = {}
        self.model_Tests = {}

        for name, build in builds.items():
            build.load(self._train_db_manager)
            train_package = SchnetTrainPackage()
            train_package.set_model(build.build())
            train_package.set_module(self._train_module)
            train_model_saver = self.saver.create_model_run_saver(name + "train")

            test_package = SchnetTrainPackage()
            test_package.set_model(train_package.model)
            test_package.set_module(self._test_module)
            test_model_saver = self.saver.create_model_run_saver(name + "test")

            train_model_run = ModelRun()
            train_model_run.reload_saver(train_model_saver)
            train_model_run.reload_package_and_fill_trainer(train_package)

            test_model_run = ModelRun()
            test_model_run.reload_saver(test_model_saver)
            test_model_run.reload_package_and_fill_trainer(test_package)

            self.model_Tests[name + "test"] = test_model_run
            self.model_runs[name + "train"] = train_model_run

    def process(self):
        for name, model_run in self.model_runs.items():
            print(f"Processing Model Run: {name}")
            model_run.train_process()

        for name, model_run in self.model_Tests.items():
            model_run.test_process()

    def compare(self):
        compare_rows = []
        for name, model_run in self.model_runs.items():
            summary = model_run.get_summary()
            summary[NAME_IN_SUMMARY_COL] = name
            compare_rows.append(summary)

        compare_df = pd.DataFrame(compare_rows)
        self.saver.save_train_comparison(compare_df)

        compare_rows = []
        for name, model_run in self.model_Tests.items():
            summary = model_run.get_summary()
            summary[NAME_IN_SUMMARY_COL] = name
            compare_rows.append(summary)

        compare_df = pd.DataFrame(compare_rows)
        self.saver.save_test_comparison(compare_df)
