from typing import List

import torch.nn as nn

import src.nn.activations as acts


def get_act(activation: str):
    if activation in acts.ACT_LIST:
        act = getattr(acts, activation)()
    else:
        act = getattr(nn, activation)()
    return act


class MultiLayerPerceptron(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_neurons: List[int] = [64, 32],
                 hidden_act: str = 'ReLU',
                 out_act: str = 'Identity',
                 dropout_prob: float = 0.0):
        super(MultiLayerPerceptron, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.hidden_act = get_act(hidden_act)
        self.out_act = get_act(out_act)

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            is_last = True if i == len(input_dims) - 1 else False
            self.layers.append(nn.Linear(in_dim, out_dim))
            if is_last:
                self.layers.append(self.out_act)
            else:
                self.layers.append(self.hidden_act)

        if dropout_prob > 0.0:
            self.dropout = nn.Dropout(dropout_prob)

    def forward(self, xs):
        for i, layer in enumerate(self.layers):
            if i != 0 and hasattr(self, 'dropout'):
                xs = self.dropout(xs)
            xs = layer(xs)
        return xs