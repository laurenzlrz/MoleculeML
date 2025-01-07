
import torch
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
import numpy as np

from src.general.Property import Property



class SchnetNN:

    def create_inputs(self, properties, custom_geo_input=None):
        inputs = []
        if custom_geo_input is not None:
            pairwise_distance = custom_geo_input
            inputs.append(pairwise_distance)
        else:
            pairwise_distance = spk.atomistic.PairwiseDistances()
            inputs.append(pairwise_distance)

        for



