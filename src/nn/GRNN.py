import dgl
import torch as th
import torch.nn as nn
import numpy as np

from torch.nn import functional as F
from torch.autograd import Variable
#
from src.nn.AttnMPNN import AttnMPNN, AttnMPNNCheck, AttnMPNNselect


class GraphNeuralNetworkBlock(nn.Module):

    def __init__(self,
                 node_model,
                 attn_model,
                 node_aggregator,
                 h_dim):
        super(GraphNeuralNetworkBlock, self).__init__()

        # Message Passing neural networks
        self.mpnn = AttnMPNN(node_model, attn_model, 2*h_dim, node_aggregator)

        # self.rnn = nn.GRU(input_size=h_dim, hidden_size=h_dim, batch_first=True)


    def forward(self,
                g: dgl.DGLGraph,
                nf: th.Tensor):

        updated_nf = []
        for i in range(nf.size(1)):
            updated_nf.append(self.mpnn(g, nf[:, i, :]).unsqueeze(dim=1))
        updated_nf = th.cat(updated_nf, dim=1)
        # nh, _ = self.rnn(updated_nf)

        return updated_nf


class GraphNeuralNetworkBlockCheck(nn.Module):

    def __init__(self,
                 node_model,
                 attn_model,
                 node_aggregator,
                 h_dim):
        super(GraphNeuralNetworkBlockCheck, self).__init__()

        # Message Passing neural networks
        self.mpnn = AttnMPNNCheck(node_model, attn_model, 2*h_dim, node_aggregator)

        # self.rnn = nn.GRU(input_size=h_dim, hidden_size=h_dim, batch_first=True)


    def forward(self,
                g: dgl.DGLGraph,
                nf: th.Tensor):

        updated_nf = []
        A = []
        for i in range(nf.size(1)):
            h, attn = self.mpnn(g, nf[:, i, :])
            updated_nf.append(h.unsqueeze(dim=1))
            A.append(attn)
        updated_nf = th.cat(updated_nf, dim=1)
        # nh, _ = self.rnn(updated_nf)

        return updated_nf, A



class GraphNeuralNetworkBlockSelect(nn.Module):

    def __init__(self,
                 node_model,
                 attn_model,
                 node_aggregator,
                 h_dim):
        super(GraphNeuralNetworkBlockSelect, self).__init__()

        # Message Passing neural networks
        self.mpnn = AttnMPNNselect(node_model, attn_model, 2*h_dim, node_aggregator)

        # self.rnn = nn.GRU(input_size=h_dim, hidden_size=h_dim, batch_first=True)


    def forward(self,
                g: dgl.DGLGraph,
                nf: th.Tensor,
                input_events: th.Tensor):

        updated_nf = []
        for i in range(nf.size(1)):

            tg_edge = []

            with th.no_grad():
                tg_src = th.where(input_events[:, i] == 1)[0]

                for c in tg_src:
                    tg_edge.append((g.edges()[0] == c).nonzero().squeeze())

                tg_edge = th.cat(tg_edge, dim=-1)


            updated_nf.append(self.mpnn(g, nf[:, i, :], tg_edge).unsqueeze(dim=1))
        updated_nf = th.cat(updated_nf, dim=1)
        # nh, _ = self.rnn(updated_nf)

        return updated_nf


