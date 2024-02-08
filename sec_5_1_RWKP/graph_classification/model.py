import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_scatter import scatter, scatter_add
from torch_sparse import spspmm
import numpy as np
import sys
import time

from utils import generate_batch_train

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class RW_layer(nn.Module):  
    def __init__(self, n_kernels, hidden_dim=None, max_step=1, size_graph_filter=10, color_matching=False, seed=0):
        super(RW_layer, self).__init__()

        self.max_step = max_step
        self.size_graph_filter = size_graph_filter
        self.n_kernels = n_kernels
        self.hidden_dim = hidden_dim
        self.color_matching = color_matching

        self.features_hidden = Parameter(torch.FloatTensor(size_graph_filter, hidden_dim, n_kernels))
        self.adj_hidden = Parameter(torch.FloatTensor((size_graph_filter * (size_graph_filter - 1)) // 2, n_kernels))
  
        self.sigmoid = nn.Sigmoid()

        self.init_weights(seed)
        
    def init_weights(self, seed):
        torch.manual_seed(seed)
        self.adj_hidden.data.uniform_(-1, 1)
        self.features_hidden.data.uniform_(0, 1)

    def forward(self, adj, features, idxs):  
        adj_hidden_norm = torch.zeros(self.size_graph_filter, self.size_graph_filter, self.n_kernels).to(device)
        idx = torch.triu_indices(self.size_graph_filter, self.size_graph_filter, 1)
        adj_hidden_norm[idx[0], idx[1], :] = self.sigmoid(self.adj_hidden)
        adj_hidden_norm = adj_hidden_norm + torch.transpose(adj_hidden_norm, 0, 1)
        
        x = features
        x = torch.cat([x, torch.zeros(1, x.shape[1]).to(device)])
        x = x[idxs] # (#Egonets, Egonet size, D_hid)
        z = self.features_hidden 

        if not self.color_matching:
            zx = torch.einsum("mcn,abc->ambn", (z, x)) # (#G, #Nodes_filter, #Nodes_sub, D_out)
            eye = torch.eye(self.size_graph_filter, device=device)             
            o = torch.einsum("ab,bcd->acd", (eye, z))
            t = torch.einsum("mcn,abc->ambn", (o, x))

            out = []
            for i in range(self.max_step):
                x = torch.einsum("abc,acd->abd", (adj, x))
                z = torch.einsum("abd,bcd->acd", (adj_hidden_norm, z)) # adj_hidden_norm: (Nhid,Nhid,Dout)
                t = torch.einsum("mcn,abc->ambn", (z, x))
                t = torch.mul(zx, t) # (#G, #Nodes_filter, #Nodes_sub, D_out)
                t = torch.mean(t, dim=[1, 2])
                out.append(t)
        else:
            y0 = torch.einsum("abc,mcn->ambn", (x, z)) # (#G, #Nodes_filter, #Nodes_sub, D_out)
            y = y0.clone()
            out = []
            for i in range(self.max_step):
                y = torch.einsum("acb,ambn->acmn", (adj, y))
                y = torch.einsum("acmn,xmn->axcn", (y, adj_hidden_norm))
                y = y0 * y 
                t = torch.mean(y, dim=[1, 2])
                y = y0 * y 
                out.append(t)

        out = sum(out) / len(out)
        return out

class kergnn(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, num_layers=2, size_graph_filter=None, mlp_hidden_dim=None, max_step=1, size_subgraph=None, color_matching=False, seed=0):
        
        super(kergnn, self).__init__()
        self.num_layers = num_layers
        self.fc_in = nn.Linear(input_dim, hidden_dims)

        self.ker_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            self.ker_layers.append(RW_layer(hidden_dims, hidden_dim=hidden_dims, max_step=max_step, size_graph_filter=size_graph_filter, color_matching=color_matching, seed=seed))          
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims))

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dims, mlp_hidden_dim),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, output_dim)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, adj, features, idxs, graph_indicator):
        h = F.relu(self.fc_in(features))
        
        previous_h = h
        for layer in range(self.num_layers):
            h = self.ker_layers[layer](adj, h, idxs)
            h = self.batch_norms[layer](h)
            h = F.relu(h)

            h = h + previous_h 
            previous_h = h

        out = scatter(h, graph_indicator, dim=0, reduce='add')
        out = self.fc_out(out)

        return out