import torch
from core.config import cfg, update_cfg
from core.train import run 
from core.model import GNN
from tqdm import tqdm

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


def create_dataset(cfg): 
    dataset = PygGraphPropPredDataset(root='data', name='ogbg-molhiv') 

    split_idx = dataset.get_idx_split() 
    train_dataset = dataset[split_idx["train"]]
    val_dataset = dataset[split_idx["valid"]]
    test_dataset = dataset[split_idx["test"]]

    return train_dataset, val_dataset, test_dataset


def create_model(cfg):
    model = GNN(None, None, 
                nhid=cfg.model.hidden_size, 
                nout=1, 
                nlayer=cfg.model.num_layers, 
                gnn_type=cfg.model.gnn_type, 
                dropout=cfg.train.dropout, 
                pooling=cfg.model.pool,
                res=True)
    return model


def train(train_loader, model, optimizer, device):
    total_loss = 0
    N = 0 
    loss_func = torch.nn.BCEWithLogitsLoss()
    for data in train_loader:
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        optimizer.zero_grad()
        loss = loss_func(model(data), y.float())
        loss.backward()
        total_loss += loss.item() * num_graphs
        optimizer.step()
        N += num_graphs
    return total_loss / N


@torch.no_grad()
def test(loader, model, evaluator, device):
    evaluator = Evaluator(name='ogbg-molhiv')
    y_true, y_pred = [], []
    for data in loader:
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        y_true.append(y)
        y_pred.append(model(data))
    input_dict = {"y_true": torch.cat(y_true), "y_pred": torch.cat(y_pred)}
    result_dict = evaluator.eval(input_dict)
    return result_dict['rocauc']


if __name__ == '__main__':
    # get config 
    cfg.merge_from_file('train/config/molhiv.yaml')
    cfg = update_cfg(cfg)
    run(cfg, create_dataset, create_model, train, test)