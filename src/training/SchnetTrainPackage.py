import pandas as pd


class SchnetTrainPackage:

    def __init__(self):

        self.trainer = None
        self.model = None
        self.module = None
        self._model_data = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def set_model(self, model):
        self.model = model

    def set_module(self, module):
        self.module = module

    def train(self):
        task = self.model.build_and_return_task()
        self.trainer.train(task, self.module)

    def test(self):
        task = self.model.build_and_return_task
        self.trainer.test(task, self.module)

    def summarize(self):
        model_stats = self.model.summary()
        trainer_stats = self.trainer.summarize()
        merged_stats = {**model_stats, **trainer_stats}
        self._model_data = pd.Series(merged_stats)

    @property
    def model_data(self):
        return self._model_data
