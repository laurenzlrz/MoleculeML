import math
from typing import Optional, Union, Sequence, Callable, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack as spk
import schnetpack.nn as snn
from schnetpack.atomistic.atomwise import Atomwise

from src.general.MolProperty import MolProperty
import schnetpack.properties as properties


DEF_ATOMWISE_REPR_KEY = "scalar_representation"


class AugmentedAtomwise(Atomwise):

    def __init__(
            self,
            atoms_in: int,
            properties_in: Optional[Dict[str, int]] = None,
            n_out: int = 1,
            n_hidden: Optional[List[int]] = None,
            activation: Callable = F.silu,
            aggregation_mode: str = "sum",
            output_key: str = "y",
            per_atom_output_key: Optional[str] = None,
            atomwise_representation_key=DEF_ATOMWISE_REPR_KEY
    ):
        """
        Args:
            atoms_in: input dimension of representation
            n_out: output dimension of target property (default: 1)
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers resulting
                in a rectangular network.
                If None, the number of neurons is divided by two after each layer starting
                n_in resulting in a pyramidal network.
            aggregation_mode: one of {sum, avg} (default: sum)
            output_key: the key under which the result will be stored
            per_atom_output_key: If not None, the key under which the per-atom result will be stored
        """
        super(Atomwise, self).__init__()
        self.n_out = n_out
        self.properties_in = properties_in

        self.atomwise_representation_key = atomwise_representation_key
        self.output_key = output_key
        self.per_atom_output_key = per_atom_output_key
        self.model_outputs = [output_key]
        if self.per_atom_output_key is not None:
            self.model_outputs.append(self.per_atom_output_key)

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.outnet = AugmentedAtomwiseNN(atom_in=atoms_in, additional_props=list(properties_in.values()),
                                          n_out=n_out, hidden_sizes=n_hidden, activation=activation)
        self.aggregation_mode = aggregation_mode

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # predict atomwise contributions
        atomwise_representations = inputs[self.atomwise_representation_key]
        input_properties = [inputs[prop] for prop in self.properties_in.keys()]
        elements_per_atom = inputs["_n_atoms"]

        #TODO: Abstract Input batch splitting (idx_m, n_atoms) to a separate function
        y = self.outnet(elements_per_atom=elements_per_atom,
                        atoms=atomwise_representations,
                        additional_properties=input_properties)

        # accumulate the per-atom output if necessary
        if self.per_atom_output_key is not None:
            inputs[self.per_atom_output_key] = y

        # aggregate
        if self.aggregation_mode is not None:
            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            y = snn.scatter_add(y, idx_m, dim_size=maxm)
            y = torch.squeeze(y, -1)

            if self.aggregation_mode == "avg":
                y = y / inputs[properties.n_atoms]

        inputs[self.output_key] = y
        return inputs

    def set_outnet(self, outnet: nn.Module):
        # Condition: outnet Input handles additional Inputs with the same properties
        self.outnet = outnet


INVALID_PROP_TENSOR_FORMAT_MSG = "Invalid format for property tensor. Expects either a 1-dim or 2-dim tensor."



class AugmentedAtomwiseNN(nn.Module):

    def __init__(self,
                 atom_in,
                 additional_props: List[int],
                 scale_factor=1.0,
                 n_out=1,
                 hidden_sizes: Optional[List[int]] = None,
                 activation=F.silu):

        super(AugmentedAtomwiseNN, self).__init__()
        self.atom_in = atom_in
        self.additional_props = additional_props
        self.activation = activation

        self.atom_in_network = nn.Linear(self.atom_in, math.ceil(self.atom_in * scale_factor))
        self.prop_layers = nn.ModuleList([nn.Linear(prop, math.ceil(prop * scale_factor))
                                          for prop in self.additional_props])

        total_in = math.ceil(self.atom_in * scale_factor) + sum([math.ceil(prop * scale_factor)
                                                                 for prop in self.additional_props])

        hidden_layers = nn.ModuleList()
        if hidden_sizes is None:
            previous_size = total_in
            while previous_size // 2 > n_out:
                next_size = previous_size // 2
                hidden_layers.append(nn.Linear(previous_size, next_size))
                previous_size = next_size
            hidden_layers.append(nn.Linear(previous_size, n_out))


        else:
            previous_size = total_in
            for i in range(len(hidden_sizes)):
                hidden_layers.append(nn.Linear(previous_size, hidden_sizes[i]))
                previous_size = hidden_sizes[i]
            hidden_layers.append(nn.Linear(previous_size, n_out))

        self.hidden_layers = hidden_layers

    def forward(self, elements_per_atom, atoms, additional_properties):

        # TODO: change index calculation to rely on idx_m as done in atomwise.py
        def convert_shape(x, prop_size):
            if len(x.shape) == 1:
                # Implies that input belongs to whole molecule and not per atom
                reshaped = x.reshape(-1, prop_size)
                return torch.repeat_interleave(reshaped, elements_per_atom, dim=0)
            if len(x.shape) == 2:
                return x
            raise ValueError(INVALID_PROP_TENSOR_FORMAT_MSG)

        atom_out = self.activation(self.atom_in_network(atoms))
        prop_outs = [self.activation(prop_layer(convert_shape(prop, prop_layer.in_features)))
                     for prop_layer, prop in zip(self.prop_layers, additional_properties)]

        total_in = torch.cat([atom_out] + prop_outs, dim=-1)

        for layer in self.hidden_layers:
            total_in = F.silu(layer(total_in))

        return total_in
