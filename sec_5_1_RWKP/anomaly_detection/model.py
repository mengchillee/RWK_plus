import torch
import torch.nn as nn
from model_feats import graph_convolution
from model_topo_ import RW_GNN


class IGAD(nn.Module):
    def __init__(self, input_dim, degrees_dim, dim_1, f_hidden_dim, f_output_dim, t_hidden_dim, t_output_dim, graph_pooling_type, n_subgraphs, size_subgraphs, max_step, normalize, dropout, color_matching, seed, device):
        super(IGAD, self).__init__()

        self.input_dim = input_dim
        self.degrees_dim = degrees_dim
        self.dim_1 = dim_1
        self.n_subgraphs = n_subgraphs
        self.size_subgraphs = size_subgraphs
        self.max_step = max_step
        self.normalize = normalize
        self.dropout = dropout
        self.device = device

        self.SEAG_features = graph_convolution(input_dim, f_hidden_dim, f_output_dim, graph_pooling_type, device)
        self.SEAG_topo = RW_GNN(degrees_dim, t_hidden_dim, t_output_dim, self.n_subgraphs, self.size_subgraphs, self.max_step, self.normalize, self.dropout, color_matching, seed, device)
        self.mlp_1 = nn.Linear(f_output_dim + t_output_dim, dim_1, bias=True)
        self.mlp_2 = nn.Linear(dim_1, 2, bias=True)
        self.relu = nn.ReLU()

    def forward(self, adj, feats, graph_pool, graph_indicator):
        feats, degrees = feats[:, :self.input_dim], feats[:, self.input_dim:]

        outputs_1 = self.SEAG_features(adj, feats, graph_pool)
        outputs_2 = self.SEAG_topo(adj, degrees, graph_indicator)

        h = torch.cat((outputs_1, outputs_2), dim=1)
        graph_embeddings = self.mlp_2(self.relu(self.mlp_1(h)))

        return graph_embeddings