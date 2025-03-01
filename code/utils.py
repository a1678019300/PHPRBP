import pickle

import numpy as np
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, \
    classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import Dataset
import math

def get_RBP_embedding(dataset_name, name):
    RBP_embedding_dir = f'../data/{dataset_name}/rbp_embeddings_labels_{name}.csv'
    RBP_embeddings_labels = pd.read_csv(RBP_embedding_dir, low_memory=False)
    return RBP_embeddings_labels

class RBPDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.long)





def calculate_metrics(true_labels, pred_labels, num_classes):
    cm = np.zeros((num_classes, num_classes))
    for true_label, pred_label in zip(true_labels, pred_labels):
        cm[true_label, pred_label] += 1

    f1_list, mcc_list, precision_list, recall_list, spy_list = [], [], [], [], []
    class_weight = [np.sum(cm[i, :]) for i in range(num_classes)]
    for i in range(num_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = np.sum(cm) - TP - FP - FN

        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        spy = TN / (TN + FP)
        f1 = 2 * recall * precision / (recall + precision)
        mcc = (TP * TN - FP * FN) / np.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN))

        f1_list.append(f1)
        mcc_list.append(mcc)
        precision_list.append(precision)
        recall_list.append(recall)
        spy_list.append(spy)

    acc = accuracy_score(true_labels, pred_labels)
    weighted_f1 = np.sum([a * b if not math.isnan(a) else 0 for a, b in zip(f1_list, class_weight)]) / np.sum(class_weight)
    weighted_mcc = np.sum([a * b if not math.isnan(a) else 0 for a, b in zip(mcc_list, class_weight)]) / np.sum(class_weight)
    weighted_precision = np.sum([a * b if not math.isnan(a) else 0 for a, b in zip(precision_list, class_weight)]) / np.sum(class_weight)
    weighted_recall = np.sum([a * b if not math.isnan(a) else 0 for a, b in zip(recall_list, class_weight)]) / np.sum(class_weight)
    return acc, weighted_f1, weighted_mcc, weighted_precision, weighted_recall
