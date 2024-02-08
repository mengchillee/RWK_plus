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
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')

    parser.add_argument('--size_subgraph', type=int, default=6, help='size of the subgraph')
    parser.add_argument('--hidden_dims', type=int, default=8, help='size of hidden layer of NN')
    parser.add_argument('--num_layers',  type=int, default=1, help='number of model layers')
    parser.add_argument('--k', type=int, default=1, help='use k-hop neighborhood to construct the subgraph ')
    parser.add_argument('--num_mlp_layers', type=int, default=1, help='number of MLP layers')
    parser.add_argument('--mlp_hidden_dim', type=int, default=16, help='hiddem dimension of MLP layers')
    parser.add_argument('--size_graph_filter', type=int, default=6, help='number of hidden graph nodes at each layer')
    parser.add_argument('--no_norm', action='store_true', default=False, help='whether to apply normalization')
    args = parser.parse_args()
    return args

def main(args, logger, task, color_matching):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    save_folder = os.path.join("save", args.dataset, "cv{}".format(args.iter))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    adj_lst, features_lst, class_labels = load_data(args.dataset, task, args.use_node_labels, args.use_node_attri)
    
    N = len(adj_lst)
    features_dim = features_lst[0].shape[1]

    n_classes = 1
    y = [np.array(class_labels[i]) for i in range(class_labels.size)]

    with open('datasets/SGC/SGC_split.pkl', 'rb') as handle:
        train_index, val_index, test_index = pickle.load(handle)

    n_train = len(train_index)
    n_test = len(test_index)
    n_val = len(val_index)

    print(n_train, n_val, n_test)

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
    adj_test, features_test, graph_indicator_test, y_test = generate_batches(adj_test, features_test, y_test, args.batch_size, device)
    adj_train, features_train, graph_indicator_train, y_train = generate_batches(adj_train, features_train, y_train, args.batch_size, device)  
    adj_val, features_val, graph_indicator_val, y_val = generate_batches(adj_val, features_val, y_val, args.batch_size, device)  

    n_test_batches = ceil(n_test/args.batch_size)
    n_train_batches = ceil(n_train/args.batch_size)
    n_val_batches = ceil(n_val/args.batch_size)
    
    subadj_test, subidx_test = generate_sub_features_idx(adj_test, features_test, size_subgraph=args.size_subgraph, k_neighbor=args.k)
    subadj_train, subidx_train = generate_sub_features_idx(adj_train, features_train, size_subgraph=args.size_subgraph, k_neighbor=args.k)
    subadj_val, subidx_val = generate_sub_features_idx(adj_val, features_val, size_subgraph=args.size_subgraph, k_neighbor=args.k)
        
    for max_step in [2, 3]:
        # Create model
        model = kergnn(features_dim, n_classes, hidden_dims=args.hidden_dims, num_layers=args.num_layers, max_step=max_step, 
                    mlp_hidden_dim=args.mlp_hidden_dim, size_graph_filter=args.size_graph_filter,
                    size_subgraph=args.size_subgraph, color_matching=color_matching, seed=0).to(device)

        # set up training
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        def train(epoch, adj, features, idxs, graph_indicator, y):
            optimizer.zero_grad()
            output = model(adj, features, idxs, graph_indicator)
            loss_train = (output.reshape(-1) - y).abs().mean()

            loss_train.backward()
            optimizer.step()
            return output, loss_train

        def test(adj, features, idxs, graph_indicator, y):
            output = model(adj, features, idxs, graph_indicator)
            loss_test = (output.reshape(-1) - y).abs().mean()
            return output, loss_test

        best_val_error = np.inf
        best_test_error = np.inf

        for epoch in tqdm(range(args.epochs)):
            start = time.time()

            # Train for one epoch
            model.train()
            train_loss = AverageMeter()
            for i in range(n_train_batches):
                output, loss = train(epoch, subadj_train[i], features_train[i], subidx_train[i], graph_indicator_train[i], y_train[i])
                train_loss.update(loss.item(), output.size(0))

            # Evaluate on validation set
            model.eval()
            val_loss = AverageMeter()
            for i in range(n_val_batches):
                output, loss = test(subadj_val[i], features_val[i], subidx_val[i], graph_indicator_val[i], y_val[i])
                val_loss.update(loss.item(), output.size(0))

            log_str = "epoch:" + '%03d ' % (epoch + 1) + "train_loss="+ "{:.5f} ".format(train_loss.avg) + \
                    "val_loss=" + "{:.5f} ".format(val_loss.avg) + "time="+ "{:.5f} ".format(time.time() - start)

            # Evaluate on test set
            if epoch % args.test_freq == 0: 
                test_loss = AverageMeter()
                for i in range(n_test_batches):
                    output, loss = test(subadj_test[i], features_test[i], subidx_test[i], graph_indicator_test[i], y_test[i])
                    test_loss.update(loss.item(), output.size(0))
                log_str += "test_loss="+ "{:.5f} ".format(test_loss.avg)

            scheduler.step()
            
            # Remember best accuracy and save checkpoint
            is_val_best = val_loss.avg <= best_val_error
            best_val_error = min(val_loss.avg, best_val_error)

            if is_val_best:
                early_stopping_counter = 0
                best_epoch = epoch + 1
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, os.path.join(save_folder,'{}_val_model.pth.tar'.format(args.prefix)) )

        logger.info("best_val_error="+ "{:.5f}".format(best_val_error))
        logger.info("best_epoch="+ "{:d}".format(best_epoch))

        # Print results
        val_checkpoint = torch.load(os.path.join(save_folder,'{}_val_model.pth.tar'.format(args.prefix)) )
        epoch = val_checkpoint['epoch']
        model.load_state_dict(val_checkpoint['state_dict'])
        optimizer.load_state_dict(val_checkpoint['optimizer'])

        model.eval()
        test_loss = AverageMeter()
        for i in range(n_test_batches):
            output, loss = test(subadj_test[i], features_test[i], subidx_test[i], graph_indicator_test[i], y_test[i])
            test_loss.update(loss.item(), output.size(0))
        logger.info("best val model on test dataset loss="+ "{:.5f}".format(test_loss.avg))

if __name__ == "__main__":
    args = args_parser()
    save_folder = os.path.join("save", args.dataset, "cv{}".format(args.iter))
    log_name = args.prefix + '_' + args.dataset + '_{}'.format(args.iter) + '.log'
    logger = get_logger(save_folder, __name__, log_name, level=args.log_level)

    for task in range(4):
        for color_matching in [True, False]:
            print('Task ' + str(task))
            main(args, logger, task, color_matching)