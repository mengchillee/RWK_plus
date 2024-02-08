from core.config import cfg, update_cfg
from core.train_nc import run 
from core.model_nc import GNN

import pickle
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.data import Data


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data):
    with open('data/TwiBot-22/split.pkl', 'rb') as handle:
        mask_dict = pickle.load(handle)

    trn_idx = mask_dict['train']
    val_idx = mask_dict['val']
    test_idx = mask_dict['test']

    device = torch.device('cuda')
    data.train_mask = index_to_mask(torch.tensor(trn_idx).long().to(device), size=data.num_nodes)
    data.val_mask = index_to_mask(torch.tensor(val_idx).long().to(device), size=data.num_nodes)
    data.test_mask = index_to_mask(torch.tensor(test_idx).long().to(device), size=data.num_nodes)
    
    return data


def create_dataset(cfg):
    name = cfg.dataset

    with open('data/TwiBot-22/edge_index.pkl', 'rb') as handle:
        edge_index = pickle.load(handle)
    with open('data/TwiBot-22/X.pkl', 'rb') as handle:
        X = pickle.load(handle)
    with open('data/TwiBot-22/y.pkl', 'rb') as handle:
        y = pickle.load(handle)

    transform = T.Compose([T.ToUndirected()])
    data = transform(Data(X.float(), edge_index, y=y))
    data = random_planetoid_splits(data)

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
    loss = F.nll_loss(out, data.y[data.train_mask], weight=torch.FloatTensor([1, 5]).to(device))
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data, model, split, evaluator, device):
    model.eval()
    logits = F.log_softmax(model(data), dim=1)
    mask = data.val_mask if split == 'valid' else data.test_mask
    pred = logits[mask].max(1)[1]
    f1 = f1_score(data.y[mask].cpu().numpy(), pred.cpu().numpy())
    return f1


if __name__ == '__main__':
    # get config 
    cfg.merge_from_file('train/config/nc.yaml')
    cfg = update_cfg(cfg)
    run(cfg, create_dataset, create_model, train, test)