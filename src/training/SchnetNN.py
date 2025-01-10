import os

import schnetpack as spk
import schnetpack.transform as trn
import torch
import torchmetrics
from pytorch_lightning import loggers, Trainer
from pytorch_lightning.profilers import PyTorchProfiler

from src.general.MolProperty import MolProperty
from src.general.NNMetrics import NNMetrics
from src.general.NNProperty import NNProperty
from src.training.AugmentedAtomwise import AugmentedAtomwise

import torch.nn as nn
from schnetpack.nn import Dense
from src.training.AdditionOutput import AdditionOutputModule

from src.LoggerCallback import EpochMetricsCallback, LoggerSaverCallback
from torchmetrics import MeanSquaredError, R2Score, MeanAbsoluteError, NormalizedRootMeanSquaredError

DIM_NOT_FLATTENED_MSG = ("Dimensions of the additional input properties are not flattened, "
                         "therefore not possible to use them as input for the neural network.")

SAVE_DIR = "data/schnet_data"
LOG_DIR_NAME = "schnet_logs"
BEST_MODEL_NAME = "best_inference_model"

DEF_CUTOFF = 5.
DEF_ATOM_BASIS_SIZE = 10
DEF_NUM_OF_INTERACTIONS = 1
DEF_RBF_BASIS_SIZE = 10
DEF_LEARNING_RATE = 1e-4
DEF_BEST_SAVES = 1

MAE_LABEL = "MAE"
LEARNING_RATE_LABEL = "lr"
STD_MONITOR = "val_loss"

DEF_METRICS = {NNMetrics.MAE: torchmetrics.MeanAbsoluteError,
               NNMetrics.MSE: torchmetrics.MeanSquaredError,
               NNMetrics.R2: torchmetrics.R2Score,
               NNMetrics.NRMSE: torchmetrics.NormalizedRootMeanSquaredError}


class AdditionSchnetNN:

    def __init__(self, schnetDb, additional_input_keys, prediction_keys, measure_keys, add1, add2, output):
        add_function = lambda x: x[0] + x[1]
        add_module = AdditionOutputModule([add1, add2], output, add_function)
        super(AdditionSchnetNN, self).__init__(schnetDb, additional_input_keys,
                                               prediction_keys, measure_keys, [add_module])


class SchnetNN:

    def __init__(self, schnetDb, additional_input_keys, prediction_keys, measure_keys=None,
                 individual_output_modules=None):
        self.datapath = SAVE_DIR
        self.atom_basis_size = DEF_ATOM_BASIS_SIZE
        self.cut_off = DEF_CUTOFF
        self.prediction_keys = prediction_keys

        dimensions = schnetDb.get_attribute_dimensions()

        # Dict contains the dimensions of the different properties and properties as Strings not the properties as enums
        self.properties_in = {}

        for prop in additional_input_keys & dimensions.keys():
            if len(dimensions[prop]) != 1:
                raise ValueError(DIM_NOT_FLATTENED_MSG)
            self.properties_in[prop.value] = dimensions[prop][0]

        self.prediction_dim = {}
        for prediction_key in self.prediction_keys:
            if len(dimensions[prediction_key]) != 1:
                raise ValueError(DIM_NOT_FLATTENED_MSG)
            self.prediction_dim[prediction_key] = dimensions[prediction_key][0]

        # TODO Checking if keys contradict
        self.output_modules = []
        for prediction_key in self.prediction_keys:
            pred_module = AugmentedAtomwise(atoms_in=self.atom_basis_size,
                                            properties_in=self.properties_in,
                                            n_out=self.prediction_dim[prediction_key],
                                            output_key=prediction_key.value)
            self.output_modules.append(pred_module)

        if individual_output_modules is not None:
            self.output_modules.extend(individual_output_modules)

        # Force example from schnetpack
        """
        pred_energy = spk.atomistic.Atomwise(n_in=self.atom_basis_size, output_key=Property.TOTAL_ENERGY.value)
        pred_forces = spk.atomistic.Forces(energy_key=Property.TOTAL_ENERGY.value, force_key=Property.FORCES.value)
        self.output_modules = [pred_energy, pred_forces]
        """

        # TODO Add Support for 2D positionwise predictions (e.g. forces),
        #  This would mean to leave the aggregation of the atomwise predictions

        self.input_modules = [spk.atomistic.PairwiseDistances()]

        self.schnet_representation = spk.representation.SchNet(
            n_atom_basis=DEF_ATOM_BASIS_SIZE,
            n_interactions=DEF_NUM_OF_INTERACTIONS,
            radial_basis=spk.nn.GaussianRBF(n_rbf=DEF_RBF_BASIS_SIZE, cutoff=self.cut_off),
            cutoff_fn=spk.nn.CosineCutoff(DEF_CUTOFF)
        )

        self.postprocessors = [trn.CastTo64()]

        self.network = spk.model.NeuralNetworkPotential(
            representation=self.schnet_representation,
            input_modules=self.input_modules,
            output_modules=self.output_modules,
            postprocessors=self.postprocessors
        )

        self.output_heads = []

        # If no measure keys are given, the prediction keys are used
        if measure_keys is None:
            measure_keys = self.prediction_keys

        metrics = {metric.value: calc() for metric, calc in DEF_METRICS.items()}

        # TODO Questions how to log for classification tasks, e.g. the accuracy because it is not a scalar value
        for measure_key in measure_keys:
            self.output_heads.append(spk.task.ModelOutput(
                name=measure_key.value,
                loss_fn=torch.nn.MSELoss(),
                loss_weight=1.,
                metrics=metrics)
            )

        self.task = spk.task.AtomisticTask(
            model=self.network,
            outputs=self.output_heads,
            optimizer_cls=torch.optim.AdamW,
            optimizer_args={LEARNING_RATE_LABEL: DEF_LEARNING_RATE}
        )

        self.logger_saver_callback = LoggerSaverCallback(list(DEF_METRICS.keys()))
        self.epoch_metrics_callback = EpochMetricsCallback()

        # TODO Use Tensorboard Logger and implement computational graph logging
        self.logger = loggers.TensorBoardLogger(SAVE_DIR, name=LOG_DIR_NAME)
        self.callbacks = [
            spk.train.ModelCheckpoint(
                model_path=os.path.join(SAVE_DIR, BEST_MODEL_NAME),
                save_top_k=DEF_BEST_SAVES,
                monitor=STD_MONITOR
            ), self.logger_saver_callback, self.epoch_metrics_callback
        ]

        self.trainer = Trainer(
            log_every_n_steps=2,
            callbacks=self.callbacks,
            logger=self.logger,
            default_root_dir=SAVE_DIR,
            max_epochs=3,
        )

        self.transforms = [
            trn.ASENeighborList(cutoff=5.),
            trn.RemoveOffsets(MolProperty.TOTAL_ENERGY.value, remove_mean=True, remove_atomrefs=False),
            trn.CastTo32()
        ]

    def get_transforms(self):
        return self.transforms

    def train(self, data_module):
        self.trainer.fit(self.task, datamodule=data_module)

    def summary(self):
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        optimizer = str(self.task.configure_optimizers())

        summary = {
            NNProperty.MAX_EPOCHS: self.trainer.max_epochs,
            NNProperty.DEVICES: self.trainer.device_ids,
            NNProperty.STRATEGY: self.trainer.strategy,
            NNProperty.PRECISION: self.trainer.precision,
            NNProperty.TOTAL_PARAMETERS: total_params,
            NNProperty.TRAINABLE_PARAMETERS: trainable_params,
            NNProperty.OPTIMIZER: optimizer
        }

        return summary
