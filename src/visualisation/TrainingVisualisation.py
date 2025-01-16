import os

import pandas as pd
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.general import SchnetAdapterStrings
from src.general.props import NNDefaultValue
from src.general.props.NNMetric import NNMetrics
from src.general.utils.Utility import split_string
from functools import reduce

SEPARATOR = "_"
EPOCH_LABEL = "Epoch"

pic_dir = "vis"

class VisualisationTensorboardLoader:

    def __init__(self, measure_metrics=None):
        if measure_metrics is None:
            measure_metrics = NNDefaultValue.DEF_DISPLAY_METRICS
        self.measure_metrics = measure_metrics
        self.batch_data = None
        self.epoch_data = None
        self.event_acc = None

    # Path is the directory of a specific tensorboard log
    def load_from_file(self, dir_path):
        self.event_acc = EventAccumulator(dir_path)
        self.event_acc.Reload()
        self.scalar_keys = self.event_acc.Tags()[SchnetAdapterStrings.SCALAR_TENSORBOARD_KEY]

        self.epochs_step_map = {s.step: s.value
                                for s in self.event_acc.Scalars(SchnetAdapterStrings.EPOCH_TENSORBOARD_KEY)}

        batch_scalar_entries = self.filter_metrics(SchnetAdapterStrings.METRIC_TRAIN_DESCR)
        batch_scalar_frames = self.scalar_to_frame(batch_scalar_entries)
        self.batch_data = self.merge_on_steps(batch_scalar_frames, NNMetrics.STEP)
        self.batch_data[NNMetrics.EPOCH] = self.batch_data[NNMetrics.STEP].map(self.epochs_step_map)

        epoch_scalar_entries = self.filter_metrics(SchnetAdapterStrings.METRIC_VALIDATION_DESCR)
        epoch_scalar_frames = self.scalar_to_frame(epoch_scalar_entries)
        for epoch_scalar_frame in epoch_scalar_frames:
            epoch_scalar_frame[NNMetrics.EPOCH] = epoch_scalar_frame[NNMetrics.STEP].map(self.epochs_step_map)
            epoch_scalar_frame.drop_duplicates(subset=[NNMetrics.EPOCH], inplace=True)
            epoch_scalar_frame.drop(columns=[NNMetrics.STEP], inplace=True)
        self.epoch_data = self.merge_on_steps(epoch_scalar_frames, NNMetrics.EPOCH)

    def scalar_to_frame(self, scalar_entries):
        scalar_dfs = []
        for key, value in scalar_entries.items():
            df = pd.DataFrame({
                NNMetrics.STEP: [s.step for s in value],
                key: [s.value for s in value],
            })
            scalar_dfs.append(df)
        return scalar_dfs

    def merge_on_steps(self, scalar_frames, merge_col):
        batches = reduce(lambda left, right: pd.merge(left, right, on=merge_col, how="outer"), scalar_frames)
        return batches

    def filter_metrics(self, phase):
        scalar_entries = {}
        for nnmetric in self.measure_metrics:
            for log_key in self.scalar_keys:
                phase_key, output_key, metric_key = split_string(log_key, phase, nnmetric.value, SEPARATOR)
                if metric_key is None or output_key is None or phase_key is None:
                    continue

                scalar_entries[nnmetric] = self.event_acc.Scalars(log_key)
        return scalar_entries

    def get_batch_data(self):
        return self.batch_data

    def get_epoch_data(self):
        return self.epoch_data

SUBFOLDER = "{path}/{phase}"
PIC_PATH = "{path}/{stat}.png"

class TrainingVisualisation:

    def __init__(self, axis_scaling=None):
        self.axis_scaling = axis_scaling
        self.batch_data = None
        self.epoch_data = None
        self.model_data = None

    def set_epoch_data(self, epoch_data):
        self.epoch_data = epoch_data

    def set_batch_data(self, batch_data):
        self.batch_data = batch_data

    def set_model_data(self, model_data):
        self.model_data = model_data

    def print_epochs(self) -> dict[NNMetrics, plt.Figure]:
        return {label: self.plot_epoch_figure(label) for label in self.epoch_data.columns if label != NNMetrics.EPOCH}

    def plot_epoch_figure(self, y_col):

        x_axis = self.epoch_data[NNMetrics.EPOCH]
        y_axis = self.epoch_data[y_col]

        fig, ax = plt.subplots()

        # Plot
        ax.plot(x_axis, y_axis, label=y_col.value)
        ax.set_xlabel(EPOCH_LABEL)
        ax.set_ylabel(y_col.value)
        ax.set_title(f'{y_col.value} vs {EPOCH_LABEL}')

        if self.axis_scaling is not None:
            self.axis_scaling(x_axis, y_axis, ax)

        ax.legend()
        ax.grid(True)

        return fig






