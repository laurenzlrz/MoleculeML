from typing import List, Dict

import torch
from torch import nn


class AdditionOutputModule(nn.Module):

    def __init__(self, props_in: List[str], output_key: str, function):
        super(AdditionOutputModule, self).__init__()
        self.props_in = props_in
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.function = function

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs[self.output_key] = self.function([inputs[prop] for prop in self.props_in])
        return inputs
