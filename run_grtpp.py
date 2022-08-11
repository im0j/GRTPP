import sys
import subprocess
import os
sys.path.append('../')
sys.path.append(os.getcwd())


import dgl
import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import DataLoader, TensorDataset
from argparse import ArgumentParser

from src.utils import mk_dir, get_non_event_mask
from src.data_utils import preprocess
from src.graph_utils import read_graph
from src.model.GRTPP import GraphRecurrentTemporalPointProcess

# import wandb




if __name__ == "__main__":

    parser = ArgumentParser()

    # Choose data to load
    parser.add_argument("--data", type=str, default='Reddit')
    parser.add_argument("--graph", type=str, default='Reddit_200graph')
    parser.add_argument("--TE", type=bool, default=True)

    # Choose model to apply
    parser.add_argument("--model", type=str, default='GRTPP')
    parser.add_argument("--epochs", type=int, default=31)
    parser.add_argument("--n_class", type=int, default=100) #NYC TAXI 299 #reddit:100 /#earthquake:162 /#911:69
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=8)


    # Set model hyperparameter
    # parser.add_argument("--node_input_dim", type=int, default=1)
    parser.add_argument("--node_aggregator", type=str, default='sum')
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--hid_dim", type=int, default=32)
    parser.add_argument("--nlayers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Set training options
    parser.add_argument("--lr", type=float, default=2e-3)

    # Set other options
    parser.add_argument("--prt_evry", type=int, default=1)
    parser.add_argument("--save_evry", type=int, default=5)
    parser.add_argument("--early_stop", type=bool, default=True)

    config = parser.parse_args()
    config.device = 'cuda' if th.cuda.is_available() else 'cpu'

    # wandb.init(
    #     project='Wandb-test',
    #     config={'learning_rate': config.lr,
    #             'architecture': config.model,
    #             'dataset': config.data})


    save_path = mk_dir(config.data, config.model)
    # commit = subprocess.check_output("git log --pretty=format:\'%h\' -n 1", shell=True).decode()
    # sys.stdout = open(save_path + '/' + '.log', 'w')
    path = 'data/' + config.data.split('_')[0] + '/' + config.data
    graphpath = 'data/' + config.data.split('_')[0] + '/' + config.graph + '.txt'
    print("config:{}".format(config))

    train_data = pd.read_csv(path + '_train.csv')
    val_data = pd.read_csv(path + '_val.csv')
    graph = read_graph(graphpath, config.n_class).to(config.device)

    ## Preprocess and build train & validation dataloader
    train_time_seq, train_event1_seq, train_time_target, train_event_target = preprocess(
        timeseries=train_data, data=config.data, seq_len=config.seq_len,
        graph=graph, TE=config.TE, device=config.device)

    val_time_seq, val_event1_seq, val_time_target, val_event_target = preprocess(timeseries=val_data,
                                                                                                  data=config.data,
                                                                                                  seq_len=config.seq_len,
                                                                                                  graph=graph,

                                                                                                  TE=config.TE,
                                                                                                  device=config.device)

    train_loader = DataLoader(TensorDataset(train_time_seq, train_event1_seq, train_time_target, train_event_target), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_time_seq, val_event1_seq, val_time_target, val_event_target), batch_size=config.batch_size, shuffle=False)


    model = GraphRecurrentTemporalPointProcess(n_class=config.n_class, seq_len=config.seq_len,
                                               hid_dim=config.hid_dim, nlayers=config.nlayers,
                        node_aggregator=config.node_aggregator,
                         lr=config.lr,  dropout=config.dropout, n_samples=config.n_samples, device=config.device)

    model.to(config.device)

    print("Model Parameters:", sum(p.numel() for p in model.parameters()))

    best_loss = 1e6
    patients = 0
    tol = 20
    savefig = False

    TLoss = []
    TLoss1 = []
    Trmse = []
    Tacc = []
    VLoss = []
    VLoss1 = []
    Vrmse = []
    Vacc = []

    for epoch in range(1, config.epochs):

        model.train()

        range_loss = range_loss1 = range_loss2 = range_int = range_lmbda = 0
        val_range_loss = val_range_loss1 = val_range_loss2 = val_range_int = val_range_lmbda = 0

        train_rmse = 0
        val_rmse = 0
        train_pred = 0
        val_pred = 0


        for time, event1, t_tg, e_tg in train_loader:

            Graph = dgl.batch([graph for _ in range(time.size(0))]).to(config.device)

            bs = int(time.size(0))

            time = time.unsqueeze(1).repeat(1, config.n_class, 1).view(bs*config.n_class, -1)
            event1 = event1.view(bs * config.n_class, config.seq_len - 1)
            t_tg = t_tg.unsqueeze(1).repeat(1, config.n_class).view(-1, 1)

            e_mask = np.zeros([bs, config.n_class])

            for i in range(e_mask.shape[1]):
                e_mask[:, i] += (e_tg.detach().cpu().numpy() == i)

            e_mask = e_mask.reshape(bs * config.n_class, 1)


            event_annot = np.array([[i for i in range(config.n_class)] for _ in range(bs)])
            event_annot = np.repeat(event_annot, config.seq_len - 1)
            event_annot = event_annot.reshape(bs * config.n_class, config.seq_len - 1)

            e_mask = th.FloatTensor(e_mask).to(config.device)
            event_annot = th.LongTensor(event_annot).to(config.device)
            none_mask = get_non_event_mask(e_tg, graph).to(config.device)


            nll, tloss, eloss = model.train_batch(Graph, time, event1, t_tg, e_tg, e_mask, event_annot, none_mask, train=True)
            # nll, tloss, eloss = model.train_batch(Graph, time, event1, t_tg, e_tg, e_mask, event_annot,
            #                                       e_mask.view(bs, config.n_class).unsqueeze(2), train=True)

            train_pred_time = model.predict_time(Graph, time, event1, event_annot)
            train_pred_event = model.predict_event(Graph, time, event1, event_annot)
            time_target = t_tg.detach().cpu().numpy().reshape(-1, config.n_class)[:, 0]
            event_target = e_tg.detach().cpu().numpy()

            range_loss += nll * bs / train_time_seq.size(0)
            range_loss1 += tloss * bs / train_time_seq.size(0)
            range_loss2 += eloss * bs / train_time_seq.size(0)

            train_rmse += np.sum((train_pred_time - time_target) ** 2)
            train_pred += sum(train_pred_event == event_target)



        model.eval()

        for time, event1, t_tg, e_tg in val_loader:

            Graph = dgl.batch([graph for _ in range(time.size(0))]).to(config.device)

            bs = int(time.size(0))

            time = time.unsqueeze(1).repeat(1, config.n_class, 1).view(bs * config.n_class, -1)
            event1 = event1.view(bs * config.n_class, config.seq_len - 1)
            t_tg = t_tg.unsqueeze(1).repeat(1, config.n_class).view(bs * config.n_class, 1)

            e_mask = np.zeros([bs, config.n_class])

            for i in range(e_mask.shape[1]):
                e_mask[:, i] += (e_tg.detach().cpu().numpy() == i)

            e_mask = e_mask.reshape(bs * config.n_class, 1)

            event_annot = np.array([[i for i in range(config.n_class)] for _ in range(bs)])
            event_annot = np.repeat(event_annot, config.seq_len - 1)
            event_annot = event_annot.reshape(bs * config.n_class, config.seq_len - 1)

            e_mask = th.FloatTensor(e_mask).to(config.device)
            event_annot = th.LongTensor(event_annot).to(config.device)
            none_mask = get_non_event_mask(e_tg, graph).to(config.device)


            val_nll, val_tloss, val_eloss = model.train_batch(Graph, time, event1, t_tg, e_tg, e_mask, event_annot,
                                                              none_mask, train=False)

            # val_nll, val_tloss, val_eloss = model.train_batch(Graph, time, event1, t_tg, e_tg, e_mask, event_annot, e_mask.view(bs, config.n_class).unsqueeze(2), train=False)
            val_pred_time = model.predict_time(Graph, time, event1, event_annot)
            val_pred_event = model.predict_event(Graph, time, event1, event_annot)

            time_target = t_tg.detach().cpu().numpy().reshape(-1, config.n_class)[:, 0]
            event_target = e_tg.detach().cpu().numpy()

            val_range_loss += val_nll * bs / val_time_seq.size(0)
            val_range_loss1 += val_tloss * bs / val_time_seq.size(0)
            val_range_loss2 += val_eloss * bs / val_time_seq.size(0)


            val_rmse += np.sum((val_pred_time - time_target) ** 2)
            val_pred += sum(val_pred_event == event_target)


        trmse = np.sqrt(train_rmse / train_time_seq.size(0))
        tacc = train_pred / train_time_seq.size(0)
        vrmse = np.sqrt(val_rmse / val_time_seq.size(0))
        vacc = val_pred / val_time_seq.size(0)

        TLoss.append(range_loss)
        TLoss1.append(range_loss1)
        VLoss.append(val_range_loss)
        VLoss1.append(val_range_loss1)
        Trmse.append(trmse)
        Vrmse.append(vrmse)
        Tacc.append(tacc)
        Vacc.append(vacc)


        if epoch % config.prt_evry == 0:
            print("EPOCH:{}".format(epoch))
            print("Train LL:{}, Train TimeLoss:{}, Train_CE:{}".format(range_loss,
                                                                    range_loss1,
                                                                    range_loss2))
            print("Val LL:{}, Val TimeLoss:{}, Val_CE:{}".format(val_range_loss, val_range_loss1, val_range_loss2))
            print("Train RMSE:{} Accuracy:{}".format(trmse, tacc))
            print("Valid RMSE:{} Accuracy:{}".format(vrmse, vacc))

        if epoch % config.save_evry == 0:
                th.save(model, save_path + '/' + str(epoch) + 'epoch')
                savefig = True



    np.save(save_path + '/Train_Loss', np.array(TLoss))
    np.save(save_path + '/Val_Loss', np.array(VLoss))
    np.save(save_path + '/Trmse', np.array(Trmse))
    np.save(save_path + '/Vrmse', np.array(Vrmse))
    np.save(save_path + '/Tacc', np.array(Tacc))
    np.save(save_path + '/Vacc', np.array(Vacc))

    sys.stdout.close
