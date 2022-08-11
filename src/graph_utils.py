import dgl

AGGR_TYPES = ['sum', 'mean', 'max']



def read_graph(graphpath, n_class):

    E1 = []
    E2 = []

    with open(graphpath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.replace('\n', '')
        e1, e2 = line.split(',')

        E1.append(int(e1))
        E2.append(int(e2))

    g = dgl.graph((E1, E2), num_nodes=n_class)

    return g


def get_aggregator(mode, from_field='m', to_field='agg_m'):
    if mode in AGGR_TYPES:
        if mode == 'sum':
            aggr = dgl.function.sum(from_field, to_field)
        if mode == 'mean':
            aggr = dgl.function.mean(from_field, to_field)
        if mode == 'max':
            aggr = dgl.function.max(from_field, to_field)
    else:
        raise RuntimeError("Given aggregation mode {} is not supported".format(mode))

    return aggr

