import torch
from core.config import cfg, update_cfg
from core.train_nc import run 
from core.model_nc import GNN
from tqdm import tqdm

import os
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Actor
from torch_geometric.datasets import WikipediaNetwork
from ogb.nodeproppred import PygNodePropPredDataset


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, ratio=(0.6, 0.2, 0.2), seed=0):
    num_nodes = len(data.y)
    trn_idx, test_idx = train_test_split(np.arange(num_nodes),
                                         test_size=ratio[2],
                                         stratify=data.y.cpu().numpy(),
                                         random_state=seed)
    trn_idx, val_idx = train_test_split(trn_idx,
                                        test_size=ratio[1] / (ratio[0] + ratio[1]),
                                        stratify=data.y.cpu().numpy()[trn_idx],
                                        random_state=seed)

    device = torch.device('cuda')
    data.train_mask = index_to_mask(torch.tensor(trn_idx).to(device), size=data.num_nodes)
    data.val_mask = index_to_mask(torch.tensor(val_idx).to(device), size=data.num_nodes)
    data.test_mask = index_to_mask(torch.tensor(test_idx).to(device), size=data.num_nodes)

    return data


def create_dataset(cfg):
    name = cfg.dataset

    transform = T.Compose([T.ToUndirected(), T.NormalizeFeatures()])
    if name in ['arxiv', 'products']:
        graph = PygNodePropPredDataset('ogbn-' + name, 'data')
        graph.data.y = graph.data.y.view(-1)
        data = transform(graph[0])
    elif name in ['cora', 'citeseer', 'pubmed']:
        root_path = '.'
        path = os.path.join(root_path, 'data', name)
        dataset = Planetoid(path, name, transform=transform)
        data = dataset[0]
    elif name in ['chameleon', 'squirrel']:
        preProcDs = WikipediaNetwork(
            root='data/', name=name, geom_gcn_preprocess=False, transform=transform)
        dataset = WikipediaNetwork(
            root='data/', name=name, geom_gcn_preprocess=True, transform=transform)
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index
    elif name in ['actor']:
        dataset = Actor(root='data/film', transform=transform)
        data = dataset[0]
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    data = random_planetoid_splits(data, ratio=(0.6, 0.2, 0.2))

    return data


def create_model(cfg, data):
    model = GNN(data.x.shape[1], None, 
                nhid=cfg.model.hidden_size, 
                nout=data.y.max().item() + 1, 
                nlayer=cfg.model.num_layers, 
                gnn_type=cfg.model.gnn_type, 
                dropout=cfg.train.dropout, 
                pooling=cfg.model.pool,
                res=True)
    return model


def train(data, model, optimizer, device):
    model.train()
    optimizer.zero_grad()
    out = F.log_softmax(model(data), dim=1)[data.train_mask]
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data, model, split, evaluator, device):
    model.eval()
    logits = F.log_softmax(model(data), dim=1)
    mask = data.val_mask if split == 'valid' else data.test_mask
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return acc


if __name__ == '__main__':
    # get config 
    cfg.merge_from_file('train/config/nc.yaml')
    cfg = update_cfg(cfg)
    run(cfg, create_dataset, create_model, train, test)