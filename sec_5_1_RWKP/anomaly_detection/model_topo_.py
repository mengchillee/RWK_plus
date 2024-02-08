import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_scatter import scatter_add
from torch_sparse import spspmm

import numpy as np


class RW_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_subgraphs, size_subgraph, max_step, normalize, dropout, color_matching, seed, device):
        super(RW_GNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_subgraphs = n_subgraphs
        self.size_subgraphs = size_subgraph
        self.max_step = max_step
        self.device = device
        self.normalize = normalize
        self.color_matching = color_matching

        self.Theta_matrix = Parameter(
            torch.FloatTensor(self.n_subgraphs, self.size_subgraphs * (self.size_subgraphs - 1) // 2, 1))
        self.features_hidden = Parameter(torch.FloatTensor(self.n_subgraphs, self.size_subgraphs, hidden_dim))

        self.fc = nn.Linear(input_dim, self.hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim * 2, self.hidden_dim, bias=True)
        self.bn = nn.BatchNorm1d(self.n_subgraphs * self.max_step)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights(seed)

        self.relu = nn.ReLU()
        self.bns = nn.ModuleList(nn.BatchNorm1d(n_subgraphs) for _ in range(max_step))
        self.fc_out = nn.Sequential(
            nn.Linear(self.n_subgraphs * self.max_step, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def init_weights(self, seed):
        torch.manual_seed(seed)
        nn.init.kaiming_normal_(self.Theta_matrix)
        self.features_hidden.data.uniform_(0, 1)

    def forward(self, adj, feats, graph_indicator):
        sampled_matrix = torch.sigmoid(self.Theta_matrix)
        sampled_matrix = sampled_matrix[:, :, 0]
        adj_sampled = torch.zeros(self.n_subgraphs, self.size_subgraphs, self.size_subgraphs).to(self.device)
        idx = torch.triu_indices(self.size_subgraphs, self.size_subgraphs, offset=1)
        adj_sampled[:, idx[0], idx[1]] = sampled_matrix
        adj_sampled = adj_sampled + torch.transpose(adj_sampled, 1, 2)
        
        unique, counts = torch.unique(graph_indicator, return_counts=True)
        n_graphs = unique.size(0)
        n_nodes = adj.shape[0]

        if self.normalize:
            norm = counts.unsqueeze(1).repeat(1, self.n_subgraphs)

        if not self.color_matching:
            E = torch.ones((self.n_subgraphs, self.size_subgraphs, n_nodes), device=self.device)
            I = torch.eye(n_nodes, device=self.device)
            adj_power = adj
            P_power_E = E

            random_walk_results = list()
            for i in range(self.max_step):
                I = torch.spmm(adj_power, I)
                P_power_E = torch.einsum("abc,acd->abd", (adj_sampled, P_power_E))
                res = torch.einsum("abc,cd->abd", (P_power_E, I))
                res = torch.zeros(res.size(0), res.size(1), n_graphs, device=self.device).index_add_(2, graph_indicator, res)
                res = torch.sum(res, dim=1)
                res = torch.transpose(res, 0, 1)
                if self.normalize:
                    res /= norm
                random_walk_results.append(res)
            random_walk_results = torch.cat(random_walk_results, dim=1)
        else:
            x = torch.sigmoid(self.fc(feats))
            z = self.features_hidden

            edge_index = adj.coalesce().indices()
            norm = torch.ones_like(edge_index[0])

            y0 = torch.einsum('ij,klj->ikl', x, z) # (total_num_nodes, num_hidden_graphs, hidden_graph_size)
            y = y0.clone()
            random_walk_results = []
            for i in range(self.max_step):
                y = torch.einsum('ijk,jkl->ijl', y, adj_sampled) # (total_num_nodes, num_hidden_graphs, hidden_graph_size)
                y = scatter_add(y[edge_index[0]] * norm.view(edge_index.size(-1), 1, 1), edge_index[1], dim=0, dim_size=y.size(0))
                y = y0 * y
                t = torch.zeros(n_graphs, y.size(1), y.size(2), device=self.device).index_add_(0, graph_indicator, y)
                t = torch.sum(t, dim=2)
                y = y0 * y
                random_walk_results.append(t)
            random_walk_results = torch.cat(random_walk_results, dim=-1)

        random_walk_results = self.fc_out(random_walk_results)
        return random_walk_results