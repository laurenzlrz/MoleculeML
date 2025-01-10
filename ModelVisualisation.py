import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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

SAVE_DIR = "data/schnet_data/schnet_logs/version_68"
if __name__ == "__main__":
    vis = ModelVisualisation()
    vis = (vis.visualise(SAVE_DIR))
    print(vis)

