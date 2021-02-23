import argparse
import copy
import random
import warnings
from collections import defaultdict
import datetime

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy import sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle as skshuffle
from tqdm import tqdm
from gcc.datasets.data_util import create_node_classification_dataset
from gcc.tasks import build_model

warnings.filterwarnings("ignore")


class NodeClassification(object):
    """Node classification task."""

    def __init__(self, dataset, model, hidden_size, num_shuffle, seed, **model_args):
        self.dataset = dataset
        self.data = create_node_classification_dataset(dataset).data
        self.label_matrix = self.data.y
        self.num_nodes, self.num_classes = self.data.y.shape

        self.model = build_model(model, hidden_size, **model_args)
        self.hidden_size = hidden_size
        self.num_shuffle = num_shuffle
        self.seed = seed

    def get_result(self, preds, labels):
        labels = labels.numpy()
        #print(preds, labels)
        #exit(0)
        res = [[0,0],[0,0]]
        for i in range(len(preds)):
            if labels[i] == 1:
                if preds[i][0] == 1:
                    res[0][0] += 1
                else:
                    res[0][1] += 1
            if labels[i] == 0:
                if preds[i][0] == 0:
                    res[1][1] += 1
                else:
                    res[1][0] += 1
        print(res)
        Acc = 1.0 * (res[0][0] + res[1][1]) / (res[0][0] + res[0][1] + res[1][0] + res[1][1])
        if res[0][0]+res[0][1] != 0:
            precision = 1.0 * res[0][0] / (res[0][0] + res[0][1])
        else:
            precision = 0
        if res[0][0] + res[1][0] != 0:
            recall = 1.0 * res[0][0] / (res[0][0] + res[1][0])
        else:
            recall = 0
        if precision+recall != 0:
            F1 = 2 * precision * recall / (precision + recall)
        else:
            F1 = 0
        return Acc, precision, recall, F1

    def train(self):
        G = nx.Graph()
        G.add_edges_from(self.data.edge_index.t().tolist())
        embeddings = self.model.train(G)

        # Map node2id
        features_matrix = np.zeros((self.num_nodes, self.hidden_size))
        for vid, node in enumerate(G.nodes()):
            features_matrix[node] = embeddings[vid]

        label_matrix = torch.Tensor(self.label_matrix)

        return self._evaluate(features_matrix, label_matrix, self.num_shuffle)

    def _evaluate(self, features_matrix, label_matrix, num_shuffle):
        # shuffle, to create train/test groups
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)
        idx_list = []
        labels = label_matrix.argmax(axis=1).squeeze().tolist()
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        # score each train/test group
        all_results = defaultdict(list)
        Accs, precisions, recalls, F1s = [], [], [], []
        for train_idx, test_idx in idx_list:

            X_train = features_matrix[train_idx]
            y_train = label_matrix[train_idx]

            X_test = features_matrix[test_idx]
            y_test = label_matrix[test_idx]

            #clf = TopKRanker(LogisticRegression(C=1000, class_weight={0:0.25, 1:0.25, 2:0.25, 3:0.25}))
            ppd_dataset = set({"loan", "investor", "debt", "agent", "dx"})
            if self.dataset not in ppd_dataset:
                m = LogisticRegression(C=1000)
            elif self.dataset == "loan":
                m = LogisticRegression(C=1000, class_weight={0:0.11, 1:0.89})
            elif self.dataset == "dx":
                m = LogisticRegression(C=1000, class_weight={0:0.205, 1:0.795})
            elif self.dataset == "agent":
                m = LogisticRegression(C=1000, class_weight={0:0.07, 1:0.93})
            elif self.dataset == "investor":
                m = LogisticRegression(C=1000, class_weight={0:0.12, 1:0.88})
            elif self.dataset == "debt":
                m = LogisticRegression(C=1000, class_weight={0:0.20, 1:0.80})

            clf = TopKRanker(m)          
            #print("ytrain:", y_train, torch.tensor(np.argmax(y_train, 1)))
            #exit(0)
            #print(X_train)
            clf.fit(X_train, torch.tensor(np.argmax(y_train, 1)))
            # find out how many labels should be predicted
            top_k_list = y_test.sum(axis=1).long().tolist()
            
            preds = clf.predict(X_test, top_k_list)
            result = f1_score(y_test, preds, average="micro")
            #print(preds, y_test)
            #print(y_test, preds.todense())
            Acc, precision, recall, F1 = self.get_result(preds.todense().argmax(axis=1), y_test.argmax(axis=1))
            Accs.append(Acc)
            precisions.append(precision)
            recalls.append(recall)
            F1s.append(F1)
            all_results[""].append(result)
        print("Accuracy, precision, recall, F1: " + str(np.mean(Accs)) + " " + str(np.mean(precisions)) + " " + str(np.mean(recalls)) + " " + str(np.mean(F1s))+ " " + str(np.std(Accs, ddof=1)) + " " + str(np.std(precisions, ddof=1)) + " " + str(np.std(recalls, ddof=1)) + " " + str(np.std(F1s, ddof=1)))
        #return dict((f"Micro-F1{train_percent}", sum(all_results[train_percent]) / len(all_results[train_percent]),) for train_percent in sorted(all_results.keys()))
        return Accs, precisions, recalls, F1s


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        #print(X, top_k_list)
        
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = sp.lil_matrix(probs.shape)

        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            #print(labels)
            for label in labels:
                all_labels[i, label] = 1
        #print(all_labels)
        #exit(0)
        return all_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shuffle", type=int, default=10)
    parser.add_argument("--emb-path", type=str, default="")
    args = parser.parse_args()
    
    task = NodeClassification(
        args.dataset,
        args.model,
        args.hidden_size,
        args.num_shuffle,
        args.seed,
        emb_path=args.emb_path,
    )
    
    time = datetime.datetime.now()

    output = open("temp_node_result.file", "a")
    Accs, precisions, recalls, F1s = task.train()
    print(np.mean(Accs), np.mean(precisions), np.mean(recalls), np.mean(F1s), np.std(Accs, ddof=1), np.std(precisions, ddof=1), np.std(recalls, ddof=1), np.std(F1s, ddof=1))
    output.write(args.emb_path + "\n")
    output.write(str(time) + " " + args.dataset + " ")
    output.write(str(np.mean(Accs)) + " "  + str(np.mean(precisions)) + " " + str(np.mean(recalls)) + " " + str(np.mean(F1s)) + " " + str(np.std(Accs, ddof=1)) + " " + str(np.std(precisions, ddof=1)) + " " + str(np.std(recalls, ddof=1)) + " " + str(np.std(F1s, ddof=1)) + "\n")
    output.close()
    