import os

import schnetpack as spk
import schnetpack.transform as trn
import torch
import torchmetrics
from pytorch_lightning import loggers, Trainer

from src.general.Property import Property
from src.training.AugmentedAtomwise import AugmentedAtomwise

SAVE_DIR = "data/schnet_data"

cutoff = 5.
n_atom_basis = 10

import torch.nn as nn
from schnetpack.nn import Dense

class customInputModule(nn.Module):
    def __init__(self, property_key, n_out):
        super().__init__()
        self.property_key = property_key
        self.linear = Dense(in_features=1, out_features=n_out)  # Beispiel: 1D-Input wird transformiert

    def forward(self, inputs):
        property_value = inputs[self.property_key].unsqueeze(-1)
        return self.linear(property_value)

def test_train(module):
    energy_input = customInputModule(Property.TOTAL_ENERGY.value, 2)
    custom_energy_output = AugmentedAtomwise(atoms_in=n_atom_basis, properties_in={Property.OLD_ENERGIES: 1}, n_out=1,
                                             output_key=Property.TOTAL_ENERGY.value)

    pairwise_distance = spk.atomistic.PairwiseDistances()  # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=10, cutoff=cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=1,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )
    pred_total_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=Property.TOTAL_ENERGY.value)

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[custom_energy_output],
        postprocessors=[trn.CastTo64()]
    )

    output_Te = spk.task.ModelOutput(
        name=Property.TOTAL_ENERGY.value,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_Te],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4}
    )

    logger = loggers.TensorBoardLogger(SAVE_DIR, name="schnet_logs")
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(SAVE_DIR, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss"
        )
    ]

    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=SAVE_DIR,
        max_epochs=3,  # for testing, we restrict the number of epochs
    )
    return trainer.fit(task, datamodule=module)
