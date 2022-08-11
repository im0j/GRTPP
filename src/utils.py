import os
import datetime
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import torch as th
from torch import nn as nn
from torch.nn import functional as F


def mk_dir(data, model):

    today = datetime.date.today()
    date = today.strftime("%y%m%d")
    path = 'result/' + date + '_' + data + '_' + model + '_'

    i = 0
    while 1:
        if os.path.isdir(path + str(i)):
            i += 1
            continue

        path = path + str(i)
        os.mkdir(path)

        return path



def get_non_event_mask(event_target, graph):

    bs = event_target.size(0)
    n_class = len(graph.nodes())
    e_mask = th.zeros([bs, n_class])

    for i in range(e_mask.shape[1]):
        e_mask[:, i] += (event_target.detach().cpu().numpy() == i)

    with th.no_grad():

        n_sample = 10
        for i in range(bs):
            dest = th.LongTensor(np.random.choice(graph.num_nodes(), n_sample, replace=False))
            e_mask[i, dest] += 1


        # bs*n_class, 1
        # for i in range(bs):
        #     src, dest = graph.out_edges(event_target[i])
        #
        #     e_mask[i, dest] += 1.

        e_mask = e_mask.gt(0).float()
    return e_mask.unsqueeze(2)



def log_lmbda(time_logits, target_time, w, b, mask):

    # lmbda = nn.Softplus()(time_logits[:, :, 0] + time_logits[:, :, 1] * target_time + b)
    lmbda = nn.Softplus()(time_logits[:, :, 0] + w * target_time + b)
    # lmbda = nn.Softplus()(time_logits[:, :, 0] + b)
    log_lmbda_all = th.log(lmbda + 1e-6)

    log_lmbda = th.sum(log_lmbda_all * mask, dim=1)

    return log_lmbda


def integral_lmbda_exact(time_logits, target_time, w, b, mask, n_samples=20, device='cuda'):


    bs, n_class, _ = time_logits.size()


    temp_time = target_time.unsqueeze(2) * th.rand([*target_time.size(), n_samples], device=device)
    # lmbda_all = nn.Softplus()((time_logits[:, :, 0]).unsqueeze(2)
    #                           + (time_logits[:, :, 1]).unsqueeze(2)  * temp_time + b) * mask


    # lmbda_all = nn.Softplus()((time_logits[:, :, 0]).unsqueeze(2)
    #                           + w * temp_time + b) / th.sum(mask, 1).unsqueeze(1)

    lmbda_all = nn.Softplus()((time_logits[:, :, 0]).unsqueeze(2)
                              + w * temp_time + b)


    # lmbda_all = nn.Softplus()((time_logits[:, :, 0]).unsqueeze(2) + w * temp_time + b) / th.sum(mask, 1).unsqueeze(1) * mask

    int_lmbda_all = th.sum(lmbda_all, dim=2) / n_samples * target_time
    int_lmbda = th.sum(int_lmbda_all, dim=-1)


    return int_lmbda



def integral_lmbda(time_logits, target_time, b):

    # lmbda_all = nn.Softplus()(
    #     (time_logits[:, :, 0]) + (time_logits[:, :, 1]) * target_time + b)
    # lmbda0_all = nn.Softplus()((time_logits[:, :, 0]) + b)
    # int_lmbda_all = (lmbda_all + lmbda0_all) / 2 * target_time
    # int_lmbda = th.sum(int_lmbda_all, dim=-1)

    lmbda_all = nn.Softplus()(
        (time_logits[:, 0]) + (time_logits[:, 1]) * target_time.squeeze() + b)
    lmbda0_all = nn.Softplus()((time_logits[:, 0]) + b)
    int_lmbda_all = (lmbda_all + lmbda0_all) / 2 * target_time.squeeze()
    int_lmbda = th.sum(int_lmbda_all, dim=-1)

    return int_lmbda
