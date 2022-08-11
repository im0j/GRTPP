import torch
import torch.nn as nn
import torch.nn.functional as F

ACT_LIST = ['Mish']


@torch.jit.script
def mish(input):
    # https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/functional.py
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):

    def forward(self, input):
        return mish(input)