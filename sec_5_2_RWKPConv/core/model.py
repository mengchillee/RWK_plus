import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_add
import core.model_utils.pyg_gnn_wrapper as gnn_wrapper 
from core.model_utils.elements import MLP, DiscreteEncoder, Identity, BN
from torch_geometric.nn.inits import reset
from torch_geometric.utils import add_remaining_self_loops


class GNN(nn.Module):
    def __init__(self, nfeat_node, nfeat_edge, nhid, nout, nlayer, gnn_type, dropout=0, pooling='add', bn=BN, res=True):
        super().__init__()
        self.input_encoder = DiscreteEncoder(nhid) if nfeat_node is None else MLP(nfeat_node, nhid, 1)
        self.edge_encoders = nn.ModuleList([DiscreteEncoder(nhid) if nfeat_edge is None else MLP(nfeat_edge, nhid, 1) for _ in range(nlayer)])
                
        self.convs = nn.ModuleList()
        for i in range(nlayer):
            self.convs.append(RWKPConv(nhid, nhid, bias=not bn))

        self.norms = nn.ModuleList([nn.BatchNorm1d(nhid) if bn else Identity() for _ in range(nlayer)])
        self.output_encoder = MLP(nhid, nout, nlayer=2, with_final_activation=False, with_norm=False if pooling=='mean' else True)
        self.nlayer = nlayer

        self.pooling = pooling
        self.dropout = dropout
        self.res = res

    def reset_parameters(self):
        self.input_encoder.reset_parameters()
        self.output_encoder.reset_parameters()
        for edge_encoder, conv, norm in zip(self.edge_encoders, self.convs, self.norms):
            edge_encoder.reset_parameters()
            conv.reset_parameters()
            norm.reset_parameters()
     
    def forward(self, data):
        x = self.input_encoder(data.x.squeeze())

        ori_edge_attr = data.edge_attr 
        if ori_edge_attr is None:
            ori_edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))

        previous_x = x
        for i, (edge_encoder, layer, norm) in enumerate(zip(self.edge_encoders, self.convs, self.norms)):
            edge_attr = edge_encoder(ori_edge_attr) 
            x = layer(x, data.edge_index, edge_attr)
            if i == self.nlayer - 1:
                break

            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                x = x + previous_x 
                previous_x = x

        if self.pooling == 'mean':
            graph_size = scatter(torch.ones_like(x[:,0], dtype=torch.int64), data.batch, dim=0, reduce='add')
            x = scatter(x, data.batch, dim=0, reduce='mean') + self.size_embedder(graph_size)
        else:
            x = scatter(x, data.batch, dim=0, reduce='add')

        x = self.output_encoder(x)
        return x


class RWKPConv(nn.Module):
    def __init__(self, nin, nout, bias=True, max_step=2):
        super().__init__()
        self.max_step = max_step
        self.size_hidden_graphs = nout

        self.features_hidden = nn.Linear(nin, nout, bias=bias)
        self.adj_hidden_layer = nn.Linear(nout, nout, bias=bias)

        self.sigmoid = torch.nn.Sigmoid()

    def reset_parameters(self):
        self.features_hidden.reset_parameters()
        self.adj_hidden_layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        adj = normalize_adj(edge_index, len(x), direction='sym', self_loops=False)

        y0 = self.features_hidden(x)
        y0 = self.sigmoid(y0)

        y = y0.clone()
        for i in range(self.max_step):
            y = torch.spmm(adj, y) 
            y = self.adj_hidden_layer(y)
            y = y0.mul(y)
            if i != self.max_step - 1:
                y = y0.mul(y)

        return y


def normalize_adj(edge_index, num_nodes=None, edge_weight=None, direction='sym', self_loops=True):
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    if self_loops:
        fill_value = 1.
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

    if direction == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif direction == 'row':
        deg_inv = deg.pow_(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    elif direction == 'col':
        deg_inv = deg.pow_(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = edge_weight * deg_inv[col]
    elif direction == 'no':
        pass
    else:
        raise ValueError()

    return torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))