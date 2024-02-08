import argparse
import time
import os
import torch
import torch.nn as nn
import random
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import Counter
from utlis import load_data, generate_batches_, compute_metrics, compute_priors
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from model import IGAD


parser = argparse.ArgumentParser(description='iGAD')

parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')

# feature_part args
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in GNN')
parser.add_argument('--f_hidden_dim', type=int, default=64, help='hidden_dim (features)')
parser.add_argument('--f_output_dim', type=int, default=32, help='output_dim (features)')
parser.add_argument('--graph_pooling_type', type=str, default='average', choices=["sum", "average"], help='the type of graph pooling (sum/average)')

# parser.add_argument('--max_step', type=int, default=3, help='Max length of walks')
parser.add_argument('--hard', type=bool, default=False, help='whether to use ST Gumbel softmax')
parser.add_argument('--normalize', type=bool, default=True, help='whether to use ST Gumbel softmax')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--t_hidden_dim', type=int, default=32, help='hidden_dim (topology)')
parser.add_argument('--t_output_dim', type=int, default=16, help='hidden_dim (topology)')

# mlp args
parser.add_argument('--dim_1', type=int, default=32, help='hidden_dim (all)')

# other args
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--device', type=int, default=0, help='which gpu to use')
parser.add_argument('--n_split', type=int, default=5, help='cross validation')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

def run(dataset, sample):
    print(dataset)
    adj_lst, feats_lst, graph_labels, features_dim = load_data(dataset)
    degrees_dim = feats_lst[0].shape[1] - features_dim
    skf = StratifiedShuffleSplit(args.n_split, test_size=0.2, train_size=0.8, random_state=args.seed)

    if sample != -1:
        _, sample_idx = train_test_split(np.arange(len(graph_labels)), stratify=graph_labels, test_size=sample, random_state=args.seed)
        adj_lst = [adj for i, adj in enumerate(adj_lst) if i in sample_idx]
        feats_lst = [feat for i, feat in enumerate(feats_lst) if i in sample_idx]
        graph_labels = np.array([yy for i, yy in enumerate(graph_labels) if i in sample_idx])
    print(Counter(graph_labels))
    print(len(graph_labels))

    for color_matching in [True]:
        performance_auc_arr = []
        performance_recall_arr = []
        performance_f1_score_arr = []
        performance_c1_recall_arr = []

        Fold_idx = 1
        for eid, (train_index, test_index) in enumerate(tqdm(skf.split(np.zeros(len(adj_lst)), graph_labels))):
            performance_auc = []
            performance_recall = []
            performance_f1_score = []
            performance_c1_recall = []

            train_index, val_index = train_test_split(train_index, test_size=0.1, stratify=graph_labels[train_index], random_state=args.seed)

            adj_train = [adj_lst[i] for i in train_index]
            feats_train = [feats_lst[i] for i in train_index]
            label_train = [graph_labels[i] for i in train_index]

            adj_val = [adj_lst[i] for i in val_index]
            feats_val = [feats_lst[i] for i in val_index]
            label_val = [graph_labels[i] for i in val_index]

            adj_test = [adj_lst[i] for i in test_index]
            feats_test = [feats_lst[i] for i in test_index]
            label_test = [graph_labels[i] for i in test_index]

            num_y_0 = label_train.count(0)
            num_y_1 = label_train.count(1)
            label_priors = compute_priors(num_y_0, num_y_1, device)

            adj_lst_train, feats_lst_train, graph_pool_lst_train, graph_indicator_lst_train, label_lst_train, n_train_batches = generate_batches_(
                                adj_train, feats_train, label_train, args.batch_size, args.graph_pooling_type, device, shuffle=True, seed=args.seed)

            adj_lst_val, feats_lst_val, graph_pool_lst_val, graph_indicator_lst_val, label_lst_val, n_val_batches = generate_batches_(
                    adj_val, feats_val, label_val, args.batch_size, args.graph_pooling_type, device, shuffle=False)

            adj_lst_test, feats_lst_test, graph_pool_lst_test, graph_indicator_lst_test, label_lst_test, n_test_batches = generate_batches_(
                            adj_test, feats_test, label_test, args.batch_size, args.graph_pooling_type, device, shuffle=False)

            best_recall_val_arr = []
            for max_step in [2, 3]:
                for n_subgraphs in [8, 16]:
                    for size_subgraphs in [5, 10]:

                        model = IGAD(features_dim,
                                    degrees_dim,
                                    args.dim_1,
                                    args.f_hidden_dim,
                                    args.f_output_dim,
                                    args.t_hidden_dim,
                                    args.t_output_dim,
                                    args.graph_pooling_type,
                                    n_subgraphs,
                                    size_subgraphs,
                                    max_step,
                                    args.normalize,
                                    args.dropout,
                                    color_matching,
                                    eid,
                                    device).to(device)
                        checkpoint_file = 'models/model_best_' + dataset + '_' + str(max_step) + '.pth.tar'

                        optimizer = optim.Adam(model.parameters(), lr=args.lr)
                        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
                        criterion = nn.CrossEntropyLoss()

                        def train(epoch, adj, feats, graph_pool, graph_indicator, labels):
                            optimizer.zero_grad()
                            outputs = model(adj, feats, graph_pool, graph_indicator)
                            loss = criterion(outputs + label_priors, labels)
                            loss.backward()
                            optimizer.step()
                            return outputs, loss

                        def test(epoch, adj, feats, graph_pool, graph_indicator, labels):
                            outputs = model(adj, feats, graph_pool, graph_indicator)
                            loss = criterion(outputs, labels)
                            return outputs, loss

                        best_recall = 0
                        for epoch in range(1, args.epochs+1):

                            model.train()
                            epoch_loss = 0
                            epoch_time = 0
                            for i in range(0, n_train_batches):
                                start_time = time.time()
                                outputs, loss = train(epoch,
                                                    adj_lst_train[i],
                                                    feats_lst_train[i],
                                                    graph_pool_lst_train[i],
                                                    graph_indicator_lst_train[i],
                                                    label_lst_train[i])
                                end_time = time.time()
                                epoch_time += end_time - start_time
                                epoch_loss += loss.item()

                            model.eval()
                            logits_ = torch.Tensor().to(device)
                            for j in range(0, n_val_batches):
                                outputs, loss = test(epoch,
                                                adj_lst_val[j],
                                                feats_lst_val[j],
                                                graph_pool_lst_val[j],
                                                graph_indicator_lst_val[j],
                                                label_lst_val[j])
                                outputs = nn.functional.softmax(outputs, dim=1)
                                if j == 0:
                                    logits_ = outputs
                                else:
                                    logits_ = torch.cat((logits_, outputs), dim=0)

                            labels_ = torch.cat(label_lst_val, dim=0)
                            auc_val, _, _, recall_val, f1_score_val, _, _, _, _, C1_recall_val, _ = compute_metrics(logits_, labels_)

                            is_best = recall_val >= best_recall
                            if is_best:
                                best_recall = recall_val
                                torch.save({
                                    'epoch': epoch + 1,
                                    'state_dict': model.state_dict(),
                                    'optimizer' : optimizer.state_dict(),
                                }, checkpoint_file)

                            scheduler.step()

                        checkpoint = torch.load(checkpoint_file)
                        model.load_state_dict(checkpoint['state_dict'])

                        model.eval()
                        logits_ = torch.Tensor().to(device)
                        for j in range(0, n_test_batches):
                            outputs, loss = test(epoch,
                                            adj_lst_test[j],
                                            feats_lst_test[j],
                                            graph_pool_lst_test[j],
                                            graph_indicator_lst_test[j],
                                            label_lst_test[j])
                            outputs = nn.functional.softmax(outputs, dim=1)
                            if j == 0:
                                logits_ = outputs
                            else:
                                logits_ = torch.cat((logits_, outputs), dim=0)

                        labels_ = torch.cat(label_lst_test, dim=0)
                        auc_test, accuracy_test, precision_test, recall_test, f1_score_test, \
                        C0_precision_test, C0_recall_test, C0_f1_test, \
                        C1_precision_test, C1_recall_test, C1_f1_test = compute_metrics(logits_, labels_)

                        Fold_idx += 1

                        performance_auc.append(auc_test)
                        performance_recall.append(recall_test)
                        performance_f1_score.append(f1_score_test)
                        performance_c1_recall.append(C1_recall_test)
                        best_recall_val_arr.append(best_recall)

            best_idx = np.argmax(best_recall_val_arr)
            performance_auc_arr.append(performance_auc[best_idx])
            performance_recall_arr.append(performance_recall[best_idx])
            performance_f1_score_arr.append(performance_f1_score[best_idx])
            performance_c1_recall_arr.append(performance_c1_recall[best_idx])

        auc_mean, auc_std = np.mean(performance_auc_arr), np.std(performance_auc_arr)
        recall_mean, recall_std = np.mean(performance_recall_arr), np.std(performance_recall_arr)
        f1_mean, f1_std = np.mean(performance_f1_score_arr), np.std(performance_f1_score_arr)
        c1_recall_mean, c1_recall_std = np.mean(performance_c1_recall_arr), np.std(performance_c1_recall_arr)

        print(color_matching)
        print('auc: %.3f +- %.3f' % (auc_mean, auc_std))
        print('recall: %.3f +- %.3f' % (recall_mean, recall_std))
        print('f1_score: %.3f +- %.3f' % (f1_mean, f1_std))
        print('c1_recall_score: %.3f +- %.3f' % (c1_recall_mean, c1_recall_std))
        print()

def main():
    datasets = [
                ['MCF-7', 0.1], 
                ['MOLT-4', 0.07], 
                ['PC-3', 0.1], 
                ['SW-620', 0.07],
                ['NCI-H23', 0.08],
                ['OVCAR-8', 0.08],
                ['P388', 0.07],
                ['SF-295', 0.08],
                ['SN12C', 0.08],
                ['UACC257', 0.1],
                ['Yeast', 0.035],
            ]
    for dataset, sample in datasets:
        run(dataset, sample)
        print()

main()