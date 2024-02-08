import os 
import json
import time
import argparse
import pickle

import numpy as np
from math import ceil
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from model import kergnn
from utils import *


torch.manual_seed(0)
np.random.seed(0)

# Argument parser
def args_parser():
    parser = argparse.ArgumentParser(description='KerGNNs')

    parser.add_argument('--kernel', default='rw', help='the kernel type')
    parser.add_argument('--iter', type=int, default=0, help='the index of fold in 10-fold validation,0-9')
    parser.add_argument('--test_freq', type=int, default=20,  help='frequency of evaluation on test dataset')
    parser.add_argument('--log_level', default='INFO', help='the level of log file')
    parser.add_argument('--dataset', default='SGC', help='dataset name')
    parser.add_argument('--use_node_labels', action='store_true', default=True, help='whether to use node labels')
    parser.add_argument('--use_node_attri', action='store_true', default=False, help='whether to use node attributes')
    parser.add_argument('--prefix', default='prefix', help='prefix of the file name for record')

    parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')

    parser.add_argument('--size_subgraph', type=int, default=10, help='size of the subgraph')
    parser.add_argument('--hidden_dims', type=int, default=8, help='size of hidden layer of NN')
    parser.add_argument('--num_layers',  type=int, default=1, help='number of model layers')
    parser.add_argument('--k', type=int, default=1, help='use k-hop neighborhood to construct the subgraph ')
    parser.add_argument('--num_mlp_layers', type=int, default=1, help='number of MLP layers')
    parser.add_argument('--mlp_hidden_dim', type=int, default=16, help='hiddem dimension of MLP layers')
    parser.add_argument('--size_graph_filter', type=int, default=6, help='number of hidden graph nodes at each layer')
    parser.add_argument('--no_norm', action='store_true', default=False, help='whether to apply normalization')
    args = parser.parse_args()
    return args

def main(dataset, smaple, use_node_labels):

    args = args_parser()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    save_folder = os.path.join("save", dataset, "cv{}".format(args.iter))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    adj_lst, features_lst, class_labels = load_data_ori(dataset, use_node_labels)

    if sample != -1:
        _, sample_idx = train_test_split(np.arange(len(class_labels)), stratify=class_labels, test_size=sample, random_state=0)
        adj_lst = [adj for i, adj in enumerate(adj_lst) if i in sample_idx]
        features_lst = [feat for i, feat in enumerate(features_lst) if i in sample_idx]
        class_labels = np.array([yy for i, yy in enumerate(class_labels) if i in sample_idx])
    print(Counter(class_labels))
    print(len(class_labels))
    
    N = len(adj_lst)
    features_dim = features_lst[0].shape[1]

    enc = LabelEncoder()
    class_labels = enc.fit_transform(class_labels)
    n_classes = np.unique(class_labels).size
    y = np.array([np.array(class_labels[i]) for i in range(class_labels.size)])

    best_results_cm = []
    best_results_ori = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for _, (train_index, test_index) in enumerate(tqdm(skf.split(np.zeros(N), y))):
        train_index, val_index = train_test_split(train_index, test_size=0.1, stratify=y[train_index], random_state=0)

        n_train = len(train_index)
        n_test = len(test_index)
        n_val = len(val_index)

        # Sampling
        adj_train = [adj_lst[i] for i in train_index]
        features_train = [features_lst[i] for i in train_index]
        y_train = [y[i] for i in train_index]

        adj_test = [adj_lst[i] for i in test_index]
        features_test = [features_lst[i] for i in test_index]
        y_test = [y[i] for i in test_index]

        adj_val = [adj_lst[i] for i in val_index]
        features_val = [features_lst[i] for i in val_index]
        y_val = [y[i] for i in val_index]

        # Create batches
        adj_test, features_test, graph_indicator_test, y_test = generate_batches(adj_test, features_test, y_test, args.batch_size, device, shuffle=True)
        adj_train, features_train, graph_indicator_train, y_train = generate_batches(adj_train, features_train, y_train, args.batch_size, device)  
        adj_val, features_val, graph_indicator_val, y_val = generate_batches(adj_val, features_val, y_val, args.batch_size, device)  

        n_test_batches = ceil(n_test/args.batch_size)
        n_train_batches = ceil(n_train/args.batch_size)
        n_val_batches = ceil(n_val/args.batch_size)
        
        seed = 0
        for color_matching in [True, False]:
            results_dict = []

            subadj_test, subidx_test = generate_sub_features_idx(adj_test, features_test, size_subgrap=args.size_subgraph, k_neighbor=args.k)
            subadj_train, subidx_train = generate_sub_features_idx(adj_train, features_train, size_subgraph=args.size_subgraph, k_neighbor=args.k)
            subadj_val, subidx_val = generate_sub_features_idx(adj_val, features_val, size_subgraph=args.size_subgraph, k_neighbor=args.k)

            for max_step in [2, 3]:

                # Create model
                model = kergnn(features_dim, n_classes, hidden_dims=args.hidden_dims, num_layers=args.num_layers, max_step=max_step, 
                            mlp_hidden_dim=args.mlp_hidden_dim, size_graph_filter=args.size_graph_filter,
                            size_subgraph=size_subgraph, color_matching=color_matching, seed=0).to(device)
                seed += 1

                # set up training
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

                def train(epoch, adj, features, idxs, graph_indicator, y):
                    optimizer.zero_grad()
                    output = model(adj, features, idxs, graph_indicator)
                    loss_train = F.cross_entropy(output, y)
                    loss_train.backward()
                    optimizer.step()
                    return output, loss_train

                def test(adj, features, idxs, graph_indicator, y):
                    output = model(adj, features, idxs, graph_indicator)
                    loss_test = F.cross_entropy(output, y)
                    return output, loss_test

                best_val_acc = 0
                best_val_loss = np.inf
                for epoch in range(args.epochs):
                    start = time.time()
                    # Train for one epoch
                    model.train()
                    train_loss = AverageMeter()
                    train_acc = AverageMeter()
                    for i in range(n_train_batches):
                        output, loss = train(epoch, subadj_train[i], features_train[i], subidx_train[i], graph_indicator_train[i], y_train[i])
                        train_loss.update(loss.item(), output.size(0))
                        train_acc.update(accuracy(output.data, y_train[i].data), output.size(0))

                    # Evaluate on validation set
                    model.eval()
                    val_loss = AverageMeter()
                    val_acc = AverageMeter()
                    for i in range(n_val_batches):
                        output, loss = test(subadj_val[i], features_val[i], subidx_val[i], graph_indicator_val[i], y_val[i])
                        val_loss.update(loss.item(), output.size(0))
                        val_acc.update(accuracy(output.data, y_val[i].data), output.size(0))

                    log_str = "epoch:" + '%03d ' % (epoch + 1) + "train_loss="+ "{:.5f} ".format(train_loss.avg)+ "train_acc="+ "{:.5f} ".format(train_acc.avg) +\
                            "val_acc="+ "{:.5f} ".format(val_acc.avg) # + "time="+ "{:.5f} ".format(time.time() - start)
                    scheduler.step()
                    
                    # Remember best accuracy and save checkpoint
                    is_val_best = val_acc.avg >= best_val_acc
                    if is_val_best:
                        best_val_acc = max(val_acc.avg.item(), best_val_acc)
                        best_val_loss = min(val_loss.avg, best_val_loss)
                        early_stopping_counter = 0
                        best_epoch = epoch + 1
                        torch.save({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                        }, os.path.join(save_folder,'{}_val_model.pth.tar'.format(args.prefix)) )

                val_checkpoint = torch.load(os.path.join(save_folder,'{}_val_model.pth.tar'.format(args.prefix)) )
                epoch = val_checkpoint['epoch']
                model.load_state_dict(val_checkpoint['state_dict'])
                optimizer.load_state_dict(val_checkpoint['optimizer'])

                model.eval()
                test_loss = AverageMeter()
                test_acc = AverageMeter()
                for i in range(n_test_batches):
                    output, loss = test(subadj_test[i], features_test[i], subidx_test[i], graph_indicator_test[i], y_test[i])
                    test_loss.update(loss.item(), output.size(0))
                    test_acc.update(accuracy(output.data, y_test[i].data), output.size(0))
                results_dict.append([best_val_acc, best_val_loss, test_acc.avg.item()])
                
            results_dict = np.array(results_dict)
            max_val_acc = np.max(results_dict[:, 0])
            if len(np.where(results_dict[:, 0] == max_val_acc)[0]) == 1:
                best_idx = np.argmax(results_dict[:, 0])
            else:
                best_idx = np.argmin(results_dict[:, 1])
            if color_matching:
                best_results_cm.append(results_dict[best_idx, 2])
            else:
                best_results_ori.append(results_dict[best_idx, 2])
    print('Test CM Accuracy: %.1f +- %.1f' % (np.mean(best_results_cm) * 100, np.std(best_results_cm) * 100))
    print('Test ORI Accuracy: %.1f +- %.1f' % (np.mean(best_results_ori) * 100, np.std(best_results_ori) * 100))

if __name__ == "__main__":
    datasets = [
        ['MUTAG', -1, True], 
        ['DD', -1, True],
        ['NCI1', -1, True],
        ['PROTEINS', -1, True],
        ['MUTAGEN', -1, True],
        ['TOX21', -1, True],
        ['ENZYMES', -1, False],
        ['IMDB-BINARY', -1, False],
        ['IMDB-MULTI', -1, False],
        ['REDDIT', 0.05, False],
    ]
    for dataset, sample, use_node_labels in datasets:
        print(dataset)
        main(dataset, sample, use_node_labels)