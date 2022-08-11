import pickle
import torch as th
import numpy as np


def preprocess(timeseries, data, seq_len, graph, TE=True, device='cpu'):

    s = {'NYCTAXI_distance':3600, 'Reddit': 3600*24, 'earthquake': 3600, '911': 60} # Reddit is already minuite . 3600*24
    scale = s[data]

    TIMES = []
    EVENTS_1hot = []
    TIMES_TARGET = []
    EVENTS_TARGET = []

    if ('Hawkes' in data):

        timeseries_tg = timeseries

        for i in range(len(timeseries_tg)-seq_len):

            ## node feat
            node_eseq = np.zeros((len(graph.nodes()), seq_len-1))

            if TE:
                tseq = np.array(timeseries_tg['T'][i:i + seq_len])
                tseq[np.isin(tseq, [0])] = 1
                tseq = list(tseq)

            else:
                tseq = np.diff(np.array(timeseries_tg['T'][i:i + seq_len]))
                tseq[np.isin(tseq, [0])] = 0.1
                tseq = [0] + list(tseq)

            for j in range(seq_len-1):
                e = int(np.array(timeseries_tg['E'][i:i+seq_len])[j])
                node_eseq[e, j] = 1

            TIMES.append(tseq[:-1])
            EVENTS_1hot.append(th.FloatTensor(node_eseq).unsqueeze(dim=0))
            # EVENTS_1hot.append(th.FloatTensor(node_eseq).unsqueeze(dim=0).to(device))
            TIMES_TARGET.append(tseq[-1]-tseq[-2])
            EVENTS_TARGET.append(timeseries_tg['E'].values[i+seq_len])


    elif 'NYCTAXI' in data:

        ID = set(timeseries['VendorID'])

        for id in ID:

            timeseries_tg = timeseries[timeseries['VendorID'] == id]

            for i in range(len(timeseries_tg)-seq_len):

                ## node feat
                node_eseq = np.zeros((len(graph.nodes()), seq_len-1))

                #time feat
                if TE:
                    tseq = np.array(timeseries_tg['T'][i:i + seq_len])/scale # /24 day /60 hour , /1 minute
                    tseq[np.isin(tseq, [0])] = 1
                    tseq = list(tseq)

                else:
                    tseq = np.diff(np.array(timeseries_tg['T'][i:i+seq_len]))/scale
                    tseq[np.isin(tseq, [0])] = 0.1
                    tseq = [0] + list(tseq)

                for j in range(seq_len-1):
                    e = np.array(timeseries_tg['E'][i:i+seq_len])[j]
                    node_eseq[e, j] = 1

                # if all((np.array(tseq[1:]) - np.array(tseq[:-1])) > .01):
                TIMES.append(tseq[:-1])
                EVENTS_1hot.append(th.FloatTensor(node_eseq).unsqueeze(dim=0).to(device))
                TIMES_TARGET.append(tseq[-1]-tseq[-2])
                EVENTS_TARGET.append(timeseries_tg['E'].values[i + seq_len])


    elif ('Reddit' in data) | ('earthquake' in data) | ('911' in data):

        ID = set(timeseries['ID'])
        ID = list(ID)
        ID.sort()

        for id in ID:

            if (id == 1000) | (id==8000):
                break

            timeseries_tg = timeseries[timeseries['ID'] == id]

            for i in range(len(timeseries_tg) - seq_len):

                ## node feat
                node_eseq = np.zeros((len(graph.nodes()), seq_len - 1))

                if TE:
                    tseq = np.array(timeseries_tg['T'][i:i + seq_len]) / scale
                    tseq[np.isin(tseq, [0])] = 1e-5
                    tseq = list(tseq)

                else:
                    tseq = np.diff(np.array(timeseries_tg['T'][i:i + seq_len])) / scale
                    tseq[np.isin(tseq, [0])] = 0.1
                    tseq = [0] + list(tseq)

                for j in range(seq_len - 1):
                    e = np.array(timeseries_tg['E'][i:i + seq_len])[j]

                    node_eseq[e, j] = 1

                if all((np.array(tseq[1:]) - np.array(tseq[:-1])) > .01):
                    TIMES.append(tseq[:-1])
                    EVENTS_1hot.append(th.FloatTensor(node_eseq).unsqueeze(dim=0).to(device))
                    TIMES_TARGET.append(tseq[-1]-tseq[-2])
                    EVENTS_TARGET.append(timeseries_tg['E'].values[i + seq_len])

    TIMES = th.FloatTensor(TIMES)
    TIMES = TIMES - TIMES[:, 0].unsqueeze(1)
    TIMES = TIMES * (TIMES > 0)
    TIMES[:, 0] += 1e-6
    EVENTS_1hot = th.cat(EVENTS_1hot, dim=0)
    TIMES_TARGET = th.FloatTensor(TIMES_TARGET).to(device)
    EVENTS_TARGET = th.LongTensor(EVENTS_TARGET).to(device)

    return TIMES.to(device), EVENTS_1hot, TIMES_TARGET, EVENTS_TARGET

