import os
import sys

sys.path.append(os.getcwd() + '/../')
sys.path.append(os.getcwd() + '/..')

from src.nn.MLP import MultiLayerPerceptron as MLP
from src.nn.AttnLayer import MultiHeadAttention
from src.nn.GRNN import GraphNeuralNetworkBlock as GN_Block
from src.nn.GRNN import GraphNeuralNetworkBlockCheck as GN_Block_Check
from src.nn.GRNN import GraphNeuralNetworkBlockSelect as GN_Block_select
from src.utils import *
import dgl

import numpy as np
import math

import torch as th
from torch import nn
from torch.optim import SGD, Adam
from torch.nn import functional as F
from torch.optim import lr_scheduler


class GraphRecurrentTemporalPointProcess(nn.Module):
    def __init__(self, n_class, seq_len, hid_dim, nlayers, node_aggregator, n_samples, lr=2e-3, dropout=0.05, device='cuda'):

        super(GraphRecurrentTemporalPointProcess, self).__init__()

        self.n_class = n_class
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.device = device

        # Position vector, used for temporal encoding
        self.position_vec = th.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / hid_dim) for i in range(hid_dim)],
            device=th.device('cuda'))

        # Embedding Each category.
        self.embedding = nn.Embedding(num_embeddings=n_class, embedding_dim=hid_dim-1)
        self.emb_drop = nn.Dropout(p=dropout)
        self.input_layer = nn.Linear(2 * hid_dim, hid_dim)
            # node model if nf : emb, indicator, h + 1 = h + node.
        self.gnn = nn.ModuleList([GN_Block(node_model=nn.Sequential(nn.Linear(2*hid_dim, hid_dim), nn.ELU(),
                                                      nn.Linear(hid_dim, hid_dim), nn.ELU()),
                             attn_model=nn.Sequential(nn.Linear(2*hid_dim, 2*hid_dim), nn.ELU(),
                                                      nn.Linear(2*hid_dim, 2*hid_dim), nn.ELU()),
                             node_aggregator=node_aggregator, h_dim=hid_dim) for _ in range(nlayers)])

        self.rnn = nn.LSTM(input_size=hid_dim, hidden_size=hid_dim, batch_first=True)

        # Lambda net
        self.mlp = nn.Sequential(nn.Linear(in_features=hid_dim, out_features=hid_dim),
                                 nn.ReLU(),
                                 nn.Linear(in_features=hid_dim, out_features=hid_dim),
                                 nn.Softplus())
        self.mlp_drop = nn.Dropout(p=dropout)

        self.model_linear = nn.Linear(hid_dim, 1)
        self.time_linear = nn.Linear(hid_dim, 1)
        self.time_pred = nn.Linear(n_class, 1)
        self.event_linear = nn.Linear(hid_dim, 1)

        self.set_criterion()
        self.optimizer = Adam(self.parameters(), lr=lr)
        # self.optimizer = SGD(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 10, gamma=0.5)
        self.s = 0

    def set_criterion(self):
        self.intensity_w = nn.Parameter(th.tensor(-0.1, dtype=th.float))
        self.intensity_b = nn.Parameter(th.tensor(0.1, dtype=th.float))

        self.model_criterion = self.RMTPPLoss
        self.time_criterion = self.MSELoss
        self.event_criterion = nn.CrossEntropyLoss()
        # self.event_criterion = LabelSmoothingLoss(0.1, self.n_class)

    def MSELoss(self, pred_time, target_time, mask):
        # pred_time : bs*n_classes
        # target_time: bs*num_classes

        target_time = th.sum(target_time * mask, dim=-1)

        loss = nn.MSELoss()(pred_time.squeeze(), target_time)

        return loss

    def RMTPPLoss(self, pred, gold, mask, none_mask):

        log_lmbda_k = log_lmbda(time_logits=pred, target_time=gold, w=self.intensity_w, b=self.intensity_b, mask=mask)

        int_lmbda_all = integral_lmbda_exact(time_logits=pred, target_time=gold, w=self.intensity_w,
                                             b=self.intensity_b, mask=none_mask,
                                             n_samples=self.n_samples, device=self.device)

        loss = log_lmbda_k - int_lmbda_all

        return -th.mean(loss), -th.mean(loss).item()

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = th.sin(result[:, :, 0::2])
        result[:, :, 1::2] = th.cos(result[:, :, 1::2])
        return result * non_pad_mask


    def forward(self, Graph, input_time, input_events1, event_annotation):

        non_pad_mask = input_events1.ne(0).type(th.float).unsqueeze(dim=-1)
        # temporal encoding
        input_time = self.temporal_enc(input_time, non_pad_mask)
        # node feature
        event_annotation = self.embedding(event_annotation)
        # event_annotation = event_annotation.repeat(1, seq_len)
        nf = self.input_layer(th.cat([event_annotation, input_events1.unsqueeze(dim=2), input_time], dim=-1))
        for gnn_layer in self.gnn:
            nf = gnn_layer(Graph, nf)

        node_h, _ = self.rnn(nf)
        out = node_h[:, -1, :]

        model_logits = self.model_linear(out)
        time_logits = self.time_linear(out)
        event_logits = self.event_linear(out)

        return model_logits, time_logits, event_logits

    def train_batch(self, Graph, time_input, event_input1, time_target, event_target, event_mask, event_anote, none_mask, train=True):

        bs = int(time_input.size(0) / self.n_class)

        model_logits, time_logits, event_logits = self.forward(Graph, time_input, event_input1, event_anote)

        # time_logits: bs * n_class, 1 -> bs, n_class // time target: bs -> bs, n_class // event target: bs -> bs, n_class
        model_logits = model_logits.view(bs, self.n_class, 1)
        time_logits = time_logits.view(bs, self.n_class)
        time_logits = self.time_pred(time_logits)
        # time_logits = time_logits.view(bs, self.n_class)
        event_logits = event_logits.view(bs, self.n_class)
        time_target = time_target.view(bs, self.n_class)
        event_mask = event_mask.view(bs, self.n_class)

        nll, loss1 = self.model_criterion(model_logits, time_target, event_mask, none_mask)
        time_loss = self.time_criterion(time_logits, time_target, event_mask)
        event_loss = self.event_criterion(event_logits, event_target)

        # loss = time_loss
        loss = nll + time_loss*10 + event_loss

        if train:
            loss.backward()
            # nn.utils.clip_grad_norm(self.parameters(), 10)
            # for p in self.parameters():
            #     if (abs(th.max(p.grad) > 200))|(abs(th.max(p.grad)) < 1e-4):
            #         print("BAD_GRAD")

            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.s == 500:
            self.scheduler.step()
            self.s = 0

        self.s += 1

        return nll.item(), time_loss.item() , event_loss.item()


    def predict_time(self, Graph, time_input, event_input1, event_annotation):

        with th.no_grad():
            bs = int(time_input.size(0) / self.n_class)
            model_logits, time_logits, _ = self.forward(Graph, time_input, event_input1, event_annotation)
            time_logits = time_logits.view(bs, self.n_class)
            time_pred = self.time_pred(time_logits)
            # time_logits : bs * n_class, 2 -> bs, n_class, 2
            #
            # aligned_time = th.sum(time_input.view(bs, self.n_class, -1), dim=1)
            # aligned_time = aligned_time[:, 1:] - aligned_time[:, :-1]
            # target_time = th.mean(aligned_time[aligned_time!=0]) * th.ones((time_logits.size(0), time_logits.size(1)), device=self.device)
            #
            # pred_time = bisect_method(time_logits, target_time, self.intensity_b, threshold=1e-4)
            #
            # return pred_time

            return time_pred.squeeze().detach().cpu().numpy()

    def predict_event(self, Graph, time_input, event_input1, event_annotation):

        with th.no_grad():
            bs = int(time_input.size(0) / self.n_class)
            _, _, event_logits = self.forward(Graph, time_input, event_input1, event_annotation)

            # reshape event_logits : bs * n_class -> bs, n_class
            event_logits = event_logits.view(bs, self.n_class).detach().cpu().numpy()

            pred_event = np.argmax(event_logits, axis=-1)

            return pred_event



class GraphRecurrentTemporalPointProcessCheck(nn.Module):
    def __init__(self, n_class, seq_len, hid_dim, nlayers, node_aggregator, n_samples, lr=2e-3, dropout=0.05, device='cuda'):

        super(GraphRecurrentTemporalPointProcessCheck, self).__init__()

        self.n_class = n_class
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.device = device

        # Position vector, used for temporal encoding
        self.position_vec = th.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / hid_dim) for i in range(hid_dim)],
            device=th.device('cuda'))

        # Embedding Each category.
        self.embedding = nn.Embedding(num_embeddings=n_class, embedding_dim=hid_dim-1)
        self.emb_drop = nn.Dropout(p=dropout)
        self.input_layer = nn.Linear(2 * hid_dim, hid_dim)
            # node model if nf : emb, indicator, h + 1 = h + node.
        self.gnn = nn.ModuleList([GN_Block_Check(node_model=nn.Sequential(nn.Linear(2*hid_dim, hid_dim), nn.ELU(),
                                                      nn.Linear(hid_dim, hid_dim), nn.ELU()),
                             attn_model=nn.Sequential(nn.Linear(2*hid_dim, 2*hid_dim), nn.ELU(),
                                                      nn.Linear(2*hid_dim, 2*hid_dim), nn.ELU()),
                             node_aggregator=node_aggregator, h_dim=hid_dim) for _ in range(nlayers)])

        self.rnn = nn.LSTM(input_size=hid_dim, hidden_size=hid_dim, batch_first=True)

        # Lambda net
        self.mlp = nn.Sequential(nn.Linear(in_features=hid_dim, out_features=hid_dim),
                                 nn.ReLU(),
                                 nn.Linear(in_features=hid_dim, out_features=hid_dim),
                                 nn.Softplus())
        self.mlp_drop = nn.Dropout(p=dropout)

        self.model_linear = nn.Linear(hid_dim, 1)
        self.time_linear = nn.Linear(hid_dim, 1)
        self.time_pred = nn.Linear(n_class, 1)
        self.event_linear = nn.Linear(hid_dim, 1)

        self.set_criterion()
        self.optimizer = Adam(self.parameters(), lr=lr)
        # self.optimizer = SGD(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 10, gamma=0.5)
        self.s = 0

    def set_criterion(self):
        self.intensity_w = nn.Parameter(th.tensor(-0.1, dtype=th.float))
        self.intensity_b = nn.Parameter(th.tensor(0.1, dtype=th.float))

        self.model_criterion = self.RMTPPLoss
        self.time_criterion = self.MSELoss
        self.event_criterion = nn.CrossEntropyLoss()
        # self.event_criterion = LabelSmoothingLoss(0.1, self.n_class)

    def MSELoss(self, pred_time, target_time, mask):
        # pred_time : bs*n_classes
        # target_time: bs*num_classes

        target_time = th.sum(target_time * mask, dim=-1)

        loss = nn.MSELoss()(pred_time.squeeze(), target_time)

        return loss

    def RMTPPLoss(self, pred, gold, mask, none_mask):

        log_lmbda_k = log_lmbda(time_logits=pred, target_time=gold, w=self.intensity_w, b=self.intensity_b, mask=mask)

        int_lmbda_all = integral_lmbda_exact(time_logits=pred, target_time=gold, w=self.intensity_w,
                                             b=self.intensity_b, mask=none_mask,
                                             n_samples=self.n_samples, device=self.device)

        loss = log_lmbda_k - int_lmbda_all

        return -th.mean(loss), -th.mean(loss).item()

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = th.sin(result[:, :, 0::2])
        result[:, :, 1::2] = th.cos(result[:, :, 1::2])
        return result * non_pad_mask


    def forward(self, Graph, input_time, input_events1, event_annotation):

        non_pad_mask = input_events1.ne(0).type(th.float).unsqueeze(dim=-1)
        # temporal encoding
        input_time = self.temporal_enc(input_time, non_pad_mask)
        # node feature
        event_annotation = self.embedding(event_annotation)
        # event_annotation = event_annotation.repeat(1, seq_len)
        nf = self.input_layer(th.cat([event_annotation, input_events1.unsqueeze(dim=2), input_time], dim=-1))
        for gnn_layer in self.gnn:
            nf, attn = gnn_layer(Graph, nf)

        node_h, _ = self.rnn(nf)
        out = node_h[:, -1, :]

        model_logits = self.model_linear(out)
        time_logits = self.time_linear(out)
        event_logits = self.event_linear(out)

        return model_logits, time_logits, event_logits, attn

    def train_batch(self, Graph, time_input, event_input1, time_target, event_target, event_mask, event_anote, none_mask, train=True):

        bs = int(time_input.size(0) / self.n_class)

        model_logits, time_logits, event_logits, attn = self.forward(Graph, time_input, event_input1, event_anote)

        # time_logits: bs * n_class, 1 -> bs, n_class // time target: bs -> bs, n_class // event target: bs -> bs, n_class
        model_logits = model_logits.view(bs, self.n_class, 1)
        time_logits = time_logits.view(bs, self.n_class)
        time_logits = self.time_pred(time_logits)
        # time_logits = time_logits.view(bs, self.n_class)
        event_logits = event_logits.view(bs, self.n_class)
        time_target = time_target.view(bs, self.n_class)
        event_mask = event_mask.view(bs, self.n_class)

        nll, loss1 = self.model_criterion(model_logits, time_target, event_mask, none_mask)
        time_loss = self.time_criterion(time_logits, time_target, event_mask)
        event_loss = self.event_criterion(event_logits, event_target)

        # loss = time_loss
        loss = nll + time_loss/100 + event_loss

        if train:
            loss.backward()
            # nn.utils.clip_grad_norm(self.parameters(), 10)
            # for p in self.parameters():
            #     if (abs(th.max(p.grad) > 200))|(abs(th.max(p.grad)) < 1e-4):
            #         print("BAD_GRAD")

            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.s == 500:
            self.scheduler.step()
            self.s = 0

        self.s += 1

        return nll.item(), time_loss.item() , event_loss.item(), attn


    def predict_time(self, Graph, time_input, event_input1, event_annotation):

        with th.no_grad():
            bs = int(time_input.size(0) / self.n_class)
            model_logits, time_logits, _, _ = self.forward(Graph, time_input, event_input1, event_annotation)
            time_logits = time_logits.view(bs, self.n_class)
            time_pred = self.time_pred(time_logits)
            # time_logits : bs * n_class, 2 -> bs, n_class, 2
            #
            # aligned_time = th.sum(time_input.view(bs, self.n_class, -1), dim=1)
            # aligned_time = aligned_time[:, 1:] - aligned_time[:, :-1]
            # target_time = th.mean(aligned_time[aligned_time!=0]) * th.ones((time_logits.size(0), time_logits.size(1)), device=self.device)
            #
            # pred_time = bisect_method(time_logits, target_time, self.intensity_b, threshold=1e-4)
            #
            # return pred_time

            return time_pred.squeeze().detach().cpu().numpy()

    def predict_event(self, Graph, time_input, event_input1, event_annotation):

        with th.no_grad():
            bs = int(time_input.size(0) / self.n_class)
            _, _, event_logits, _ = self.forward(Graph, time_input, event_input1, event_annotation)

            # reshape event_logits : bs * n_class -> bs, n_class
            event_logits = event_logits.view(bs, self.n_class).detach().cpu().numpy()

            pred_event = np.argmax(event_logits, axis=-1)

            return pred_event



class GraphRecurrentTemporalPointProcessAblation1(nn.Module):
    def __init__(self, n_class, seq_len, hid_dim, nlayers, node_aggregator, n_samples, lr=2e-3, dropout=0.05, device='cuda'):

        super(GraphRecurrentTemporalPointProcessAblation1, self).__init__()

        self.n_class = n_class
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.device = device

        # Position vector, used for temporal encoding
        self.position_vec = th.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / hid_dim) for i in range(hid_dim)],
            device=th.device('cuda'))

        # Embedding Each category.
        self.embedding = nn.Embedding(num_embeddings=n_class, embedding_dim=hid_dim-1)
        self.emb_drop = nn.Dropout(p=dropout)
        self.input_layer = nn.Linear(2 * hid_dim, hid_dim)
            # node model if nf : emb, indicator, h + 1 = h + node.

        self.rnn = nn.LSTM(input_size=hid_dim, hidden_size=hid_dim, batch_first=True)

        # Lambda net
        self.mlp = nn.Sequential(nn.Linear(in_features=hid_dim, out_features=hid_dim),
                                 nn.ReLU(),
                                 nn.Linear(in_features=hid_dim, out_features=hid_dim),
                                 nn.Softplus())
        self.mlp_drop = nn.Dropout(p=dropout)

        self.model_linear = nn.Linear(hid_dim, 1)
        self.time_linear = nn.Linear(hid_dim, 1)
        self.time_pred = nn.Linear(n_class, 1)
        self.event_linear = nn.Linear(hid_dim, 1)

        self.set_criterion()
        self.optimizer = Adam(self.parameters(), lr=lr)
        # self.optimizer = SGD(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 10, gamma=0.5)
        self.s = 0

    def set_criterion(self):
        self.intensity_w = nn.Parameter(th.tensor(-0.1, dtype=th.float))
        self.intensity_b = nn.Parameter(th.tensor(0.1, dtype=th.float))

        self.model_criterion = self.RMTPPLoss
        self.time_criterion = self.MSELoss
        self.event_criterion = nn.CrossEntropyLoss()
        # self.event_criterion = LabelSmoothingLoss(0.1, self.n_class)

    def MSELoss(self, pred_time, target_time, mask):
        # pred_time : bs*n_classes
        # target_time: bs*num_classes

        target_time = th.sum(target_time * mask, dim=-1)

        loss = nn.MSELoss()(pred_time.squeeze(), target_time)

        return loss

    def RMTPPLoss(self, pred, gold, mask, none_mask):

        log_lmbda_k = log_lmbda(time_logits=pred, target_time=gold, w=self.intensity_w, b=self.intensity_b, mask=mask)

        int_lmbda_all = integral_lmbda_exact(time_logits=pred, target_time=gold, w=self.intensity_w,
                                             b=self.intensity_b, mask=none_mask,
                                             n_samples=self.n_samples, device=self.device)

        loss = log_lmbda_k - int_lmbda_all

        return -th.mean(loss), -th.mean(loss).item()

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = th.sin(result[:, :, 0::2])
        result[:, :, 1::2] = th.cos(result[:, :, 1::2])
        return result * non_pad_mask


    def forward(self, Graph, input_time, input_events1, event_annotation):

        non_pad_mask = input_events1.ne(0).type(th.float).unsqueeze(dim=-1)
        # temporal encoding
        input_time = self.temporal_enc(input_time, non_pad_mask)
        # node feature
        event_annotation = self.embedding(event_annotation)
        # event_annotation = event_annotation.repeat(1, seq_len)
        nf = self.input_layer(th.cat([event_annotation, input_events1.unsqueeze(dim=2), input_time], dim=-1))

        node_h, _ = self.rnn(nf)
        out = node_h[:, -1, :]

        model_logits = self.model_linear(out)
        time_logits = self.time_linear(out)
        event_logits = self.event_linear(out)

        return model_logits, time_logits, event_logits

    def train_batch(self, Graph, time_input, event_input1, time_target, event_target, event_mask, event_anote, none_mask, train=True):

        bs = int(time_input.size(0) / self.n_class)

        model_logits, time_logits, event_logits = self.forward(Graph, time_input, event_input1, event_anote)

        # time_logits: bs * n_class, 1 -> bs, n_class // time target: bs -> bs, n_class // event target: bs -> bs, n_class
        model_logits = model_logits.view(bs, self.n_class, 1)
        time_logits = time_logits.view(bs, self.n_class)
        time_logits = self.time_pred(time_logits)
        # time_logits = time_logits.view(bs, self.n_class)
        event_logits = event_logits.view(bs, self.n_class)
        time_target = time_target.view(bs, self.n_class)
        event_mask = event_mask.view(bs, self.n_class)

        nll, loss1 = self.model_criterion(model_logits, time_target, event_mask, none_mask)
        time_loss = self.time_criterion(time_logits, time_target, event_mask)
        event_loss = self.event_criterion(event_logits, event_target)

        # loss = time_loss
        loss = nll + time_loss/100 + event_loss

        if train:
            loss.backward()
            # nn.utils.clip_grad_norm(self.parameters(), 10)
            # for p in self.parameters():
            #     if (abs(th.max(p.grad) > 200))|(abs(th.max(p.grad)) < 1e-4):
            #         print("BAD_GRAD")

            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.s == 500:
            self.scheduler.step()
            self.s = 0

        self.s += 1

        return nll.item(), time_loss.item() , event_loss.item()


    def predict_time(self, Graph, time_input, event_input1, event_annotation):

        with th.no_grad():
            bs = int(time_input.size(0) / self.n_class)
            model_logits, time_logits, _ = self.forward(Graph, time_input, event_input1, event_annotation)
            time_logits = time_logits.view(bs, self.n_class)
            time_pred = self.time_pred(time_logits)
            # time_logits : bs * n_class, 2 -> bs, n_class, 2
            #
            # aligned_time = th.sum(time_input.view(bs, self.n_class, -1), dim=1)
            # aligned_time = aligned_time[:, 1:] - aligned_time[:, :-1]
            # target_time = th.mean(aligned_time[aligned_time!=0]) * th.ones((time_logits.size(0), time_logits.size(1)), device=self.device)
            #
            # pred_time = bisect_method(time_logits, target_time, self.intensity_b, threshold=1e-4)
            #
            # return pred_time

            return time_pred.squeeze().detach().cpu().numpy()

    def predict_event(self, Graph, time_input, event_input1, event_annotation):

        with th.no_grad():
            bs = int(time_input.size(0) / self.n_class)
            _, _, event_logits = self.forward(Graph, time_input, event_input1, event_annotation)

            # reshape event_logits : bs * n_class -> bs, n_class
            event_logits = event_logits.view(bs, self.n_class).detach().cpu().numpy()

            pred_event = np.argmax(event_logits, axis=-1)

            return pred_event



class GraphRecurrentTemporalPointProcessAblation2(nn.Module):
    def __init__(self, n_class, seq_len, hid_dim, nlayers, node_aggregator, n_samples, lr=2e-3, dropout=0.05, device='cuda'):

        super(GraphRecurrentTemporalPointProcessAblation2, self).__init__()

        self.n_class = n_class
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.device = device

        # Position vector, used for temporal encoding
        self.position_vec = th.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / hid_dim) for i in range(hid_dim)],
            device=th.device('cuda'))

        # Embedding Each category.
        self.embedding = nn.Embedding(num_embeddings=n_class, embedding_dim=hid_dim-1)
        self.emb_drop = nn.Dropout(p=dropout)
        self.input_layer = nn.Linear(2 * hid_dim, hid_dim)
            # node model if nf : emb, indicator, h + 1 = h + node.
        self.gnn = nn.ModuleList([GN_Block(node_model=nn.Sequential(nn.Linear(2*hid_dim, hid_dim), nn.ELU(),
                                                      nn.Linear(hid_dim, hid_dim), nn.ELU()),
                             attn_model=nn.Sequential(nn.Linear(2*hid_dim, 2*hid_dim), nn.ELU(),
                                                      nn.Linear(2*hid_dim, 2*hid_dim), nn.ELU()),
                             node_aggregator=node_aggregator, h_dim=hid_dim) for _ in range(nlayers)])


        self.attn = MultiHeadAttention(n_head=4, d_model= hid_dim, d_k=hid_dim, d_v=hid_dim,
                                       dropout=dropout, normalize_before=True)
        self.rnn = nn.LSTM(input_size=hid_dim, hidden_size=hid_dim, batch_first=True)

        # Lambda net
        self.mlp = nn.Sequential(nn.Linear(in_features=hid_dim, out_features=hid_dim),
                                 nn.ReLU(),
                                 nn.Linear(in_features=hid_dim, out_features=hid_dim),
                                 nn.Softplus())
        self.mlp_drop = nn.Dropout(p=dropout)

        self.model_linear = nn.Linear(hid_dim, 1)
        self.time_linear = nn.Linear(hid_dim, 1)
        self.time_pred = nn.Linear(n_class, 1)
        self.event_linear = nn.Linear(hid_dim, 1)

        self.set_criterion()
        self.optimizer = Adam(self.parameters(), lr=lr)
        # self.optimizer = SGD(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 10, gamma=0.5)
        self.s = 0

    def set_criterion(self):
        self.intensity_w = nn.Parameter(th.tensor(-0.1, dtype=th.float))
        self.intensity_b = nn.Parameter(th.tensor(0.1, dtype=th.float))

        self.model_criterion = self.RMTPPLoss
        self.time_criterion = self.MSELoss
        self.event_criterion = nn.CrossEntropyLoss()
        # self.event_criterion = LabelSmoothingLoss(0.1, self.n_class)

    def MSELoss(self, pred_time, target_time, mask):
        # pred_time : bs*n_classes
        # target_time: bs*num_classes

        target_time = th.sum(target_time * mask, dim=-1)

        loss = nn.MSELoss()(pred_time.squeeze(), target_time)

        return loss

    def RMTPPLoss(self, pred, gold, mask, none_mask):

        log_lmbda_k = log_lmbda(time_logits=pred, target_time=gold, w=self.intensity_w, b=self.intensity_b, mask=mask)

        int_lmbda_all = integral_lmbda_exact(time_logits=pred, target_time=gold, w=self.intensity_w,
                                             b=self.intensity_b, mask=none_mask,
                                             n_samples=self.n_samples, device=self.device)

        loss = log_lmbda_k - int_lmbda_all

        return -th.mean(loss), -th.mean(loss).item()

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = th.sin(result[:, :, 0::2])
        result[:, :, 1::2] = th.cos(result[:, :, 1::2])
        return result * non_pad_mask


    def forward(self, Graph, input_time, input_events1, event_annotation):

        # non_pad_mask = input_events1.ne(0).type(th.float).unsqueeze(dim=-1)
        non_pad_mask = input_time.ne(0).type(th.float).unsqueeze(dim=-1)
        # temporal encoding
        input_time = self.temporal_enc(input_time, non_pad_mask)
        # node feature
        event_annotation = self.embedding(event_annotation)
        # event_annotation = event_annotation.repeat(1, seq_len)
        nf = self.input_layer(th.cat([event_annotation, input_events1.unsqueeze(dim=2), input_time], dim=-1))
        for gnn_layer in self.gnn:
            nf = gnn_layer(Graph, nf)

        attn_mask = th.triu(non_pad_mask*non_pad_mask.transpose(1, 2)).bool()

        nf, attn = self.attn(nf, nf, nf, attn_mask)

        node_h, _ = self.rnn(nf)
        out = node_h[:, -1, :]

        model_logits = self.model_linear(out)
        time_logits = self.time_linear(out)
        event_logits = self.event_linear(out)

        return model_logits, time_logits, event_logits

    def train_batch(self, Graph, time_input, event_input1, time_target, event_target, event_mask, event_anote, none_mask, train=True):

        bs = int(time_input.size(0) / self.n_class)

        model_logits, time_logits, event_logits = self.forward(Graph, time_input, event_input1, event_anote)

        # time_logits: bs * n_class, 1 -> bs, n_class // time target: bs -> bs, n_class // event target: bs -> bs, n_class
        model_logits = model_logits.view(bs, self.n_class, 1)
        time_logits = time_logits.view(bs, self.n_class)
        time_logits = self.time_pred(time_logits)
        # time_logits = time_logits.view(bs, self.n_class)
        event_logits = event_logits.view(bs, self.n_class)
        time_target = time_target.view(bs, self.n_class)
        event_mask = event_mask.view(bs, self.n_class)

        nll, loss1 = self.model_criterion(model_logits, time_target, event_mask, none_mask)
        time_loss = self.time_criterion(time_logits, time_target, event_mask)
        event_loss = self.event_criterion(event_logits, event_target)

        # loss = time_loss
        loss = nll + time_loss/100 + event_loss

        if train:
            loss.backward()
            # nn.utils.clip_grad_norm(self.parameters(), 10)
            # for p in self.parameters():
            #     if (abs(th.max(p.grad) > 200))|(abs(th.max(p.grad)) < 1e-4):
            #         print("BAD_GRAD")

            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.s == 500:
            self.scheduler.step()
            self.s = 0

        self.s += 1

        return nll.item(), time_loss.item() , event_loss.item()


    def predict_time(self, Graph, time_input, event_input1, event_annotation):

        with th.no_grad():
            bs = int(time_input.size(0) / self.n_class)
            model_logits, time_logits, _ = self.forward(Graph, time_input, event_input1, event_annotation)
            time_logits = time_logits.view(bs, self.n_class)
            time_pred = self.time_pred(time_logits)
            # time_logits : bs * n_class, 2 -> bs, n_class, 2
            #
            # aligned_time = th.sum(time_input.view(bs, self.n_class, -1), dim=1)
            # aligned_time = aligned_time[:, 1:] - aligned_time[:, :-1]
            # target_time = th.mean(aligned_time[aligned_time!=0]) * th.ones((time_logits.size(0), time_logits.size(1)), device=self.device)
            #
            # pred_time = bisect_method(time_logits, target_time, self.intensity_b, threshold=1e-4)
            #
            # return pred_time

            return time_pred.squeeze().detach().cpu().numpy()

    def predict_event(self, Graph, time_input, event_input1, event_annotation):

        with th.no_grad():
            bs = int(time_input.size(0) / self.n_class)
            _, _, event_logits = self.forward(Graph, time_input, event_input1, event_annotation)

            # reshape event_logits : bs * n_class -> bs, n_class
            event_logits = event_logits.view(bs, self.n_class).detach().cpu().numpy()

            pred_event = np.argmax(event_logits, axis=-1)

            return pred_event



class GraphRecurrentTemporalPointProcessAblation3(nn.Module):
    def __init__(self, n_class, seq_len, hid_dim, nlayers, node_aggregator, n_samples, lr=2e-3, dropout=0.05, device='cuda'):

        super(GraphRecurrentTemporalPointProcessAblation3, self).__init__()

        self.n_class = n_class
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.device = device

        # Position vector, used for temporal encoding
        self.position_vec = th.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / hid_dim) for i in range(hid_dim)],
            device=th.device('cuda'))

        # Embedding Each category.
        self.embedding = nn.Embedding(num_embeddings=n_class, embedding_dim=hid_dim-1)
        self.emb_drop = nn.Dropout(p=dropout)
        self.input_layer = nn.Linear(2 * hid_dim, hid_dim)
            # node model if nf : emb, indicator, h + 1 = h + node.
        self.gnn = nn.ModuleList([GN_Block_select(node_model=nn.Sequential(nn.Linear(2*hid_dim, hid_dim), nn.ELU(),
                                                      nn.Linear(hid_dim, hid_dim), nn.ELU()),
                             attn_model=nn.Sequential(nn.Linear(2*hid_dim, 2*hid_dim), nn.ELU(),
                                                      nn.Linear(2*hid_dim, hid_dim), nn.ELU()),
                             node_aggregator=node_aggregator, h_dim=hid_dim) for _ in range(nlayers)])


        self.rnn = nn.LSTM(input_size=hid_dim, hidden_size=hid_dim, batch_first=True)

        # Lambda net
        self.mlp = nn.Sequential(nn.Linear(in_features=hid_dim, out_features=hid_dim),
                                 nn.ReLU(),
                                 nn.Linear(in_features=hid_dim, out_features=hid_dim),
                                 nn.Softplus())
        self.mlp_drop = nn.Dropout(p=dropout)

        self.model_linear = nn.Linear(hid_dim, 1)
        self.time_linear = nn.Linear(hid_dim, 1)
        self.time_pred = nn.Linear(n_class, 1)
        self.event_linear = nn.Linear(hid_dim, 1)

        self.set_criterion()
        self.optimizer = Adam(self.parameters(), lr=lr)
        # self.optimizer = SGD(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 10, gamma=0.5)
        self.s = 0

    def set_criterion(self):
        self.intensity_w = nn.Parameter(th.tensor(-0.1, dtype=th.float))
        self.intensity_b = nn.Parameter(th.tensor(0.1, dtype=th.float))

        self.model_criterion = self.RMTPPLoss
        self.time_criterion = self.MSELoss
        self.event_criterion = nn.CrossEntropyLoss()
        # self.event_criterion = LabelSmoothingLoss(0.1, self.n_class)

    def MSELoss(self, pred_time, target_time, mask):
        # pred_time : bs*n_classes
        # target_time: bs*num_classes

        target_time = th.sum(target_time * mask, dim=-1)

        loss = nn.MSELoss()(pred_time.squeeze(), target_time)

        return loss

    def RMTPPLoss(self, pred, gold, mask, none_mask):

        log_lmbda_k = log_lmbda(time_logits=pred, target_time=gold, w=self.intensity_w, b=self.intensity_b, mask=mask)

        int_lmbda_all = integral_lmbda_exact(time_logits=pred, target_time=gold, w=self.intensity_w,
                                             b=self.intensity_b, mask=none_mask,
                                             n_samples=self.n_samples, device=self.device)

        loss = log_lmbda_k - int_lmbda_all

        return -th.mean(loss), -th.mean(loss).item()

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = th.sin(result[:, :, 0::2])
        result[:, :, 1::2] = th.cos(result[:, :, 1::2])
        return result * non_pad_mask


    def forward(self, Graph, input_time, input_events1, event_annotation):

        non_pad_mask = input_events1.ne(0).type(th.float).unsqueeze(dim=-1)
        # temporal encoding
        input_time = self.temporal_enc(input_time, non_pad_mask)
        # node feature
        event_annotation = self.embedding(event_annotation)
        # event_annotation = event_annotation.repeat(1, seq_len)


        nf = self.input_layer(th.cat([event_annotation, input_events1.unsqueeze(dim=2), input_time], dim=-1))
        for gnn_layer in self.gnn:
            nf = gnn_layer(Graph, nf, input_events1)

        node_h, _ = self.rnn(nf)
        out = node_h[:, -1, :]

        model_logits = self.model_linear(out)
        time_logits = self.time_linear(out)
        event_logits = self.event_linear(out)

        return model_logits, time_logits, event_logits

    def train_batch(self, Graph, time_input, event_input1, time_target, event_target, event_mask, event_anote, none_mask, train=True):

        bs = int(time_input.size(0) / self.n_class)

        model_logits, time_logits, event_logits = self.forward(Graph, time_input, event_input1, event_anote)

        # time_logits: bs * n_class, 1 -> bs, n_class // time target: bs -> bs, n_class // event target: bs -> bs, n_class
        model_logits = model_logits.view(bs, self.n_class, 1)
        time_logits = time_logits.view(bs, self.n_class)
        time_logits = self.time_pred(time_logits)
        # time_logits = time_logits.view(bs, self.n_class)
        event_logits = event_logits.view(bs, self.n_class)
        time_target = time_target.view(bs, self.n_class)
        event_mask = event_mask.view(bs, self.n_class)

        nll, loss1 = self.model_criterion(model_logits, time_target, event_mask, none_mask)
        time_loss = self.time_criterion(time_logits, time_target, event_mask)
        event_loss = self.event_criterion(event_logits, event_target)

        # loss = time_loss
        loss = nll + time_loss/100 + event_loss

        if train:
            loss.backward()
            # nn.utils.clip_grad_norm(self.parameters(), 10)
            # for p in self.parameters():
            #     if (abs(th.max(p.grad) > 200))|(abs(th.max(p.grad)) < 1e-4):
            #         print("BAD_GRAD")

            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.s == 500:
            self.scheduler.step()
            self.s = 0

        self.s += 1

        return nll.item(), time_loss.item() , event_loss.item()


    def predict_time(self, Graph, time_input, event_input1, event_annotation):

        with th.no_grad():
            bs = int(time_input.size(0) / self.n_class)
            model_logits, time_logits, _ = self.forward(Graph, time_input, event_input1, event_annotation)
            time_logits = time_logits.view(bs, self.n_class)
            time_pred = self.time_pred(time_logits)
            # time_logits : bs * n_class, 2 -> bs, n_class, 2
            #
            # aligned_time = th.sum(time_input.view(bs, self.n_class, -1), dim=1)
            # aligned_time = aligned_time[:, 1:] - aligned_time[:, :-1]
            # target_time = th.mean(aligned_time[aligned_time!=0]) * th.ones((time_logits.size(0), time_logits.size(1)), device=self.device)
            #
            # pred_time = bisect_method(time_logits, target_time, self.intensity_b, threshold=1e-4)
            #
            # return pred_time

            return time_pred.squeeze().detach().cpu().numpy()

    def predict_event(self, Graph, time_input, event_input1, event_annotation):

        with th.no_grad():
            bs = int(time_input.size(0) / self.n_class)
            _, _, event_logits = self.forward(Graph, time_input, event_input1, event_annotation)

            # reshape event_logits : bs * n_class -> bs, n_class
            event_logits = event_logits.view(bs, self.n_class).detach().cpu().numpy()

            pred_event = np.argmax(event_logits, axis=-1)

            return pred_event
