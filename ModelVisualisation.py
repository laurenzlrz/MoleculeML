from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.general.props.NNMetric import NNMetrics
from src.visualisation.TrainingVisualisation import VisualisationTensorboardLoader, TrainingVisualisation
import src.general.utils.VisualisationUtility as vu

class ModelVisualisation:

    def visualise(self, path):
        return self.extract_tensorboard_scalars(path)

    def extract_tensorboard_scalars(self, path):
        """
        Extrahiert Loss- und Metrik-Verläufe aus TensorBoard-Logs.
        """
        event_acc = EventAccumulator(path)
        event_acc.Reload()

        scalars = {}

        for tag in event_acc.Tags()["scalars"]:
            scalars[tag] = [(s.step, s.value) for s in event_acc.Scalars(tag)]

        return scalars


SAVE_DIR = "data/training/model1_3/logger/logging/version_0"
if __name__ == "__man__":
    vis = ModelVisualisation()
    vis = (vis.visualise(SAVE_DIR))
    print(vis)

if __name__ == "__main__":
    vis = VisualisationTensorboardLoader()
    vis.load_from_file(SAVE_DIR)
    data = vis.get_epoch_data()
    vis = TrainingVisualisation("data/vis", axis_scaling=vu.scale_axis_to_zero)
    vis.set_epoch_data(data)
    vis.print_epochs()
