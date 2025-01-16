import faulthandler
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from src.pipeline.AtomProcessPipeline import AtomProcessPipeline
from src.pipeline.TrainPipeline import TrainPipeline

if __name__ == "__main__":
    AtomProcessPipeline = AtomProcessPipeline()
    AtomProcessPipeline.process()


if __name__ == "__man__":
    train_pipeline = TrainPipeline()
    train_pipeline.process()
