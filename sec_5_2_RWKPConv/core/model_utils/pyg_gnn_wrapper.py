import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F


class GCNConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.layer = gnn.GCNConv(nin, nout, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)