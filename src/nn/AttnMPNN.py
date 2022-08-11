import dgl
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from dgl.ops import edge_softmax

from src.graph_utils import get_aggregator


class AttnMPNN(nn.Module):

    def __init__(self,
                 node_model: nn.Module,
                 attn_model: nn.Module,
                 hid_dim: int,
                 node_aggregator: str = 'mean'):
        super(AttnMPNN, self).__init__()
        self.node_model = node_model
        self.attn_model = attn_model
        self.attn_fc = nn.Linear(hid_dim, 1, bias=False)
        self.node_aggr = get_aggregator(node_aggregator)

    def forward(self, g: dgl.DGLGraph, nf: th.tensor):
        """
        :param g: dgl.DGLGraph or dgl.BatchedDGLGraph
        :param nf: node feature; expected dim [#. total nodes x feature dimension]
        :return:
        """

        with g.local_scope():
            g.ndata['x'] = nf
            g.apply_edges(self._msg_func)  # compute edge embedding 'm'
            logits = self.attn_model(g.edata['attn_input'])
            logits = nn.LeakyReLU()(self.attn_fc(logits))
            g.edata['attn'] = edge_softmax(g, logits)
            g.update_all(message_func=dgl.function.src_mul_edge('x', 'attn', 'm'),
                         reduce_func=self.node_aggr)
            unf = self.node_model(th.cat([g.ndata['x'], g.ndata['agg_m']], dim=-1))
            return unf

    @staticmethod
    def _msg_func(edges):
        src_feat = edges.src['x']
        dst_feat = edges.dst['x']
        return {'attn_input': th.cat([src_feat, dst_feat], dim=-1)}




class AttnMPNNCheck(nn.Module):

    def __init__(self,
                 node_model: nn.Module,
                 attn_model: nn.Module,
                 hid_dim: int,
                 node_aggregator: str = 'mean'):
        super(AttnMPNNCheck, self).__init__()
        self.node_model = node_model
        self.attn_model = attn_model
        self.attn_fc = nn.Linear(hid_dim, 1, bias=False)
        self.node_aggr = get_aggregator(node_aggregator)

    def forward(self, g: dgl.DGLGraph, nf: th.tensor):
        """
        :param g: dgl.DGLGraph or dgl.BatchedDGLGraph
        :param nf: node feature; expected dim [#. total nodes x feature dimension]
        :return:
        """

        with g.local_scope():
            g.ndata['x'] = nf
            g.apply_edges(self._msg_func)  # compute edge embedding 'm'
            logits = self.attn_model(g.edata['attn_input'])
            logits = nn.LeakyReLU()(self.attn_fc(logits))
            g.edata['attn'] = edge_softmax(g, logits)
            g.update_all(message_func=dgl.function.src_mul_edge('x', 'attn', 'm'),
                         reduce_func=self.node_aggr)
            unf = self.node_model(th.cat([g.ndata['x'], g.ndata['agg_m']], dim=-1))
            return unf, g.edata['attn'].detach().cpu().numpy()

    @staticmethod
    def _msg_func(edges):
        src_feat = edges.src['x']
        dst_feat = edges.dst['x']
        return {'attn_input': th.cat([src_feat, dst_feat], dim=-1)}



## ATTN APPN for SELECTED EDGES
class AttnMPNNselect(nn.Module):

    def __init__(self,
                 node_model: nn.Module,
                 attn_model: nn.Module,
                 hid_dim: int,
                 node_aggregator: str = 'mean'):
        super(AttnMPNNselect, self).__init__()
        self.node_model = node_model
        self.attn_model = attn_model
        self.attn_fc = nn.Linear(hid_dim, 1, bias=False)
        self.node_aggr = get_aggregator(node_aggregator)

    def forward(self, g: dgl.DGLGraph, nf: th.tensor, tg_edge: th.tensor):
        """
        :param g: dgl.DGLGraph or dgl.BatchedDGLGraph
        :param nf: node feature; expected dim [#. total nodes x feature dimension]
        :return:
        """

        node_length = len(g.nodes())
        edge_length = len(g.edges()[0])

        # node_mask = th.zeros((node_length)).to('cuda')
        # for _n in tg_node:
        #     node_mask[_n] += 1.

        # edge_mask = th.zeros((edge_length)).to('cuda')
        # for _e in tg_edge:
        #     edge_mask[_e] += 1.

        with g.local_scope():
            g.ndata['x'] = nf
            # g.apply_edges(func=self._msg_func, edges=tg_edge)

            # g.send_and_recv(v=tg_node, reduce_func=self._reduce_func)
            g.send_and_recv(tg_edge, message_func=self._msg_func, reduce_func=self._reduce_func)

            # compute edge embedding 'm'
            # logits = self.attn_model(g.edata['attn_input'])
            # logits = nn.LeakyReLU()(self.attn_fc(logits))
            # g.edata['attn'] = edge_softmax(g, logits)*edge_mask
            # g.apply_nodes(message_func=dgl.function.src_mul_edge('x', 'attn', 'm'),
            #              reduce_func=self.node_aggr)
            unf = g.ndata['h']

            return unf

    # @staticmethod
    def _msg_func(self, edges):
        src_feat = edges.src['x']
        dst_feat = edges.dst['x']
        msg = self.attn_model(th.cat([src_feat, dst_feat], dim=-1))
        # attn_val = nn.LeakyReLU()(attn_val)
        # return {'attn_input': th.cat([src_feat, dst_feat], dim=-1)}
        return {'m': msg}

    def _reduce_func(self, nodes):
        msg = nodes.mailbox['m']
        # attn = self.attn_model(msg)
        # weight = th.softmax(msg, dim=1)
        x = nodes.data['x']
        h = self.node_model(th.cat([x, msg.squeeze(1)], dim=-1))

        return {'h': h}