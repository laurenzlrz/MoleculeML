import schnetpack as spk
import schnetpack.transform as trn
import torch
import torch.nn as nn
import torchmetrics
from schnetpack.atomistic import Atomwise

from src.general import SchnetAdapterStrings
from src.general.props import NNDefaultValue
from src.general.props.MolProperty import MolProperty
from src.general.props.NNMetric import NNMetrics
from src.general.props.NNProperty import NNProperty
from src.training.AdditionOutput import AdditionOutputModule
from src.training.AugmentedAtomwise import AugmentedAtomwise

DIM_NOT_FLATTENED_MSG = ("Dimensions of the additional input properties are not flattened, "
                         "therefore not possible to use them as input for the neural network.")

STAT_PATH_FORMAT = "{path}/{name}_model_stats.csv"

DEF_CUTOFF = NNDefaultValue.DEF_CUTOFF
DEF_ATOM_BASIS_SIZE = NNDefaultValue.DEF_ATOM_BASIS_SIZE
DEF_NUM_OF_INTERACTIONS = NNDefaultValue.DEF_NUM_OF_INTERACTIONS
DEF_RBF_BASIS_SIZE = NNDefaultValue.DEF_RBF_BASIS_SIZE
DEF_LEARNING_RATE = NNDefaultValue.DEF_LEARNING_RATE

LEARNING_RATE_LABEL = SchnetAdapterStrings.LEARNING_RATE_KEY
MONITOR = SchnetAdapterStrings.NN_PERFORMANCE_KPI

# TODO Implement metrics to enum
DEF_METRICS = {NNMetrics.MAE: torchmetrics.MeanAbsoluteError,
               NNMetrics.MSE: torchmetrics.MeanSquaredError,
               NNMetrics.R2: torchmetrics.R2Score,
               NNMetrics.NRMSE: torchmetrics.NormalizedRootMeanSquaredError}
DEF_LOSS = nn.MSELoss
DEF_OPTIMIZER = torch.optim.AdamW


class SchnetNN:

    def build_input_modules(self):
        self.input_modules = [spk.atomistic.PairwiseDistances()]

    def build_representation(self):
        self.schnet_representation = spk.representation.SchNet(
            n_atom_basis=self.atom_basis_size,
            n_interactions=self.num_of_interactions,
            radial_basis=spk.nn.GaussianRBF(n_rbf=self.rbf_basis_size, cutoff=self.cut_off),
            cutoff_fn=spk.nn.CosineCutoff(self.cut_off)
        )

    def build_output_modules(self):
        self.output_modules = []
        for prediction_key in self.prediction_keys:
            pred_module = spk.atomistic.Atomwise(n_in=self.atom_basis_size, output_key=prediction_key.value)
            """
            pred_module = AugmentedAtomwise(atoms_in=self.atom_basis_size,
                                            properties_in=self.properties_in,
                                            n_out=self.predictions_out[prediction_key],
                                            output_key=prediction_key.value)
            """
            self.output_modules.append(pred_module)

    def build_postprocessors(self):
        self.postprocessors = [trn.CastTo64()]

    def build_network(self):
        self.network = spk.model.NeuralNetworkPotential(
            representation=self.schnet_representation,
            input_modules=self.input_modules,
            output_modules=self.output_modules,
            postprocessors=self.postprocessors
        )

    def build_output_heads(self):
        self.output_heads = []
        metrics = {metric.value: calc() for metric, calc in DEF_METRICS.items()}

        lossweight = 1 / len(self.prediction_keys)
        for measure_key in self.prediction_keys:
            self.output_heads.append(spk.task.ModelOutput(
                name=measure_key.value,
                loss_fn=DEF_LOSS(),
                loss_weight=lossweight,
                metrics=metrics)
            )

    def build_required_transforms(self):
        self.transforms = [
            trn.ASENeighborList(cutoff=5.),
            trn.RemoveOffsets(MolProperty.TOTAL_ENERGY.value, remove_mean=True, remove_atomrefs=False),
            trn.CastTo32()
        ]

    def prop_sanity_check(self):
        # TODO allow variable length prediction (eg. forces)
        # TODO Checking if keys contradict
        self.properties_in = {}
        for prop, shape in self.additional_input_keys.items():
            if len(shape) != 1:
                self.properties_in = None
                raise ValueError(DIM_NOT_FLATTENED_MSG)
            self.properties_in[prop.value] = shape[0]

    def prediction_sanity_check(self):
        self.predictions_out = {}
        for prediction_key, shape in self.prediction_keys.items():
            if len(shape) != 1:
                self.predictions_out = None
                raise ValueError(DIM_NOT_FLATTENED_MSG)
            self.predictions_out[prediction_key] = shape[0]

    # TODO MERGE SCHnetNN and SchnetTrainer
    def __init__(self, additional_input_keys, prediction_keys, atom_basis_size=DEF_ATOM_BASIS_SIZE,
                 num_of_interactions=DEF_NUM_OF_INTERACTIONS, rbf_basis_size=DEF_RBF_BASIS_SIZE, cut_off=DEF_CUTOFF,
                 learning_rate=DEF_LEARNING_RATE):
        self.additional_input_keys = additional_input_keys
        self.prediction_keys = prediction_keys

        self.atom_basis_size = atom_basis_size
        self.num_of_interactions = num_of_interactions
        self.rbf_basis_size = rbf_basis_size
        self.cut_off = cut_off
        self.learning_rate = learning_rate

        # Dict contains the dimensions of the different properties and properties as Strings not the properties as enums
        self.properties_in = None
        # Predictions out can be used to determine the output size of the network
        self.predictions_out = None
        self.input_modules = None
        self.schnet_representation = None
        self.output_modules = None
        self.postprocessors = None
        self.network = None

        self.output_heads = None

        self.transforms = None

        self.task = None

        self.prop_sanity_check()
        self.prediction_sanity_check()

        self.build_input_modules()
        self.build_representation()
        self.build_output_modules()
        self.build_postprocessors()
        self.build_network()

        self.build_output_heads()

        self.build_and_return_task()

    def get_transforms(self):
        return self.transforms

    def build_and_return_task(self):
        self.task = spk.task.AtomisticTask(
            model=self.network,
            outputs=self.output_heads,
            optimizer_cls=DEF_OPTIMIZER,
            optimizer_args={LEARNING_RATE_LABEL: self.learning_rate}
        )
        return self.task

    def summary(self):
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        optimizer = str(self.task.configure_optimizers())
        inputs = [key.value for key in self.additional_input_keys.keys()]
        inputs.append(MolProperty.COORDINATES.value)
        preds = [key.value for key in self.prediction_keys.keys()]

        summary = {
            NNProperty.LEARNING_RATE: self.learning_rate,
            NNProperty.TOTAL_PARAMETERS: total_params,
            NNProperty.TRAINABLE_PARAMETERS: trainable_params,
            NNProperty.OPTIMIZER: optimizer,
            NNProperty.INPUTS: inputs,
            NNProperty.PREDICTIONS: preds,
            NNProperty.OUTPUTS: preds
        }

        return summary


class AdditionSchnetNN(SchnetNN):

    def __init__(self, additional_input_keys, prediction_keys, measure_keys, add1, add2, output,
                 atom_basis_size=DEF_ATOM_BASIS_SIZE, num_of_interactions=DEF_NUM_OF_INTERACTIONS,
                 rbf_basis_size=DEF_RBF_BASIS_SIZE, cut_off=DEF_CUTOFF, learning_rate=DEF_LEARNING_RATE):
        self.measure_keys = measure_keys
        self.add_module = AdditionOutputModule([add1.value, add2.value], output.value, add_function)
        super(AdditionSchnetNN, self).__init__(additional_input_keys, prediction_keys, atom_basis_size,
                                               num_of_interactions, rbf_basis_size, cut_off, learning_rate)

    def build_output_modules(self):
        super(AdditionSchnetNN, self).build_output_modules()
        self.output_modules.append(self.add_module)
        print(self.output_modules)

    def build_output_heads(self):
        self.output_heads = []
        metrics = {metric.value: calc() for metric, calc in DEF_METRICS.items()}

        lossweight = 1 / len(self.measure_keys)
        for measure_key in self.measure_keys:
            self.output_heads.append(spk.task.ModelOutput(
                name=measure_key.value,
                loss_fn=DEF_LOSS(),
                loss_weight=lossweight,
                metrics=metrics)
            )

    def summary(self):
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        optimizer = str(self.task.configure_optimizers())
        inputs = [key.value for key in self.additional_input_keys.keys()]
        inputs.append(MolProperty.COORDINATES.value)
        preds = [key.value for key in self.prediction_keys.keys()]
        estimates = [key.value for key in self.measure_keys]

        summary = {
            NNProperty.LEARNING_RATE: self.learning_rate,
            NNProperty.TOTAL_PARAMETERS: total_params,
            NNProperty.TRAINABLE_PARAMETERS: trainable_params,
            NNProperty.OPTIMIZER: optimizer,
            NNProperty.INPUTS: inputs,
            NNProperty.PREDICTIONS: preds,
            NNProperty.OUTPUTS: estimates
        }

        return summary

def add_function(input_list):
    return input_list[0] + input_list[1]
