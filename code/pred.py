import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import CNNModel, data_augmentation
from utils import get_RBP_embedding, RBPDataset, calculate_metrics


parser = argparse.ArgumentParser(description="Main script of PHPRBP.")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset_name', type=str, default='Dataset1')
parser.add_argument('--model_name', type=str, default='PHPRBP')
parser.add_argument('--host_label', type=str, default='Host Genus',
                    help="Host Phylum, Host Class, Host Order, Host Family, Host Genus, Host Species")
parser.add_argument('--batch_size', type=int)
parser.add_argument('--reduction', type=int)
parser.add_argument('--drop_prob', type=float)
parser.add_argument('--epochs', type=int, help='number of epoch for training')

args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
RBP_embeddings_labels_PLM1 = get_RBP_embedding(args.dataset_name, 'ESM2')
RBP_embeddings_labels_PLM2 = get_RBP_embedding(args.dataset_name, 'ProtT5')
column_names_1 = [str(i) for i in range(1, 1281)]
RBP_embeddings_PLM1 = RBP_embeddings_labels_PLM1[column_names_1].values.astype(np.float32)
column_names_2 = [str(i) for i in range(1, 1025)]
RBP_embeddings_PLM2 = RBP_embeddings_labels_PLM2[column_names_2].values.astype(np.float32)

RBP_embeddings_combined = np.concatenate((RBP_embeddings_PLM1, RBP_embeddings_PLM2), axis=1)
RBP_labels = RBP_embeddings_labels_PLM1[args.host_label].values.astype(np.int64)
output_size = len(set((list(RBP_labels))))

save_dir = f'../data/{args.dataset_name}/cross_validation_data'
data_augmentation(RBP_embeddings_combined, RBP_labels, save_dir)
fold_results = []

for fold in range(1, 6):
    print(f'Fold {fold}')
    fold_dir = os.path.join(save_dir, f'fold_{fold}')
    train_embeddings = np.load(os.path.join(fold_dir, 'train_embeddings.npy'))
    train_labels = np.load(os.path.join(fold_dir, 'train_labels.npy'))
    test_embeddings = np.load(os.path.join(fold_dir, 'test_embeddings.npy'))
    test_labels = np.load(os.path.join(fold_dir, 'test_labels.npy'))

    train_dataset = RBPDataset(train_embeddings, train_labels)
    test_dataset = RBPDataset(test_embeddings, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    in_channels1 = 1280
    in_channels2 = 1024
    output_channels = 512
    kernel_size = 3
    reduction = args.reduction
    drop_prob = args.drop_prob

    model = CNNModel(in_channels1=in_channels1, in_channels2=in_channels2, out_channels=output_channels,
                     kernel_size=kernel_size, output_size=output_size, reduction=reduction, drop_prob=drop_prob).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([param for name, param in model.named_parameters()], lr=3e-4, weight_decay=0.00001)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    model.eval()
    final_op = []  # prediction
    hosts = []  # label
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            final_op.extend(predicted.cpu().numpy())
            hosts.extend(labels.cpu().numpy())

    num_classes = output_size
    ACC, F1, MCC, precision, recall = calculate_metrics(hosts, final_op, num_classes)
    print('ACC: %.4f\nF1: %.4f\nMCC: %.4f\nprecision: %.4f\nrecall: %.4f\nspecificity: %.4f' % (
        ACC, F1, MCC, precision, recall))

    fold_results.append({
        'fold': fold,
        'accuracy': ACC,
        'f1': F1,
        'mcc': MCC,
        'precision': precision,
        'recall': recall,
    })

avg_accuracy = np.mean([result['accuracy'] for result in fold_results])
std_accuracy = np.std([result['accuracy'] for result in fold_results])

avg_f1 = np.mean([result['f1'] for result in fold_results])
std_f1 = np.std([result['f1'] for result in fold_results])

avg_mcc = np.mean([result['mcc'] for result in fold_results])
std_mcc = np.std([result['mcc'] for result in fold_results])

avg_precision = np.mean([result['precision'] for result in fold_results])
std_precision = np.std([result['precision'] for result in fold_results])

avg_recall = np.mean([result['recall'] for result in fold_results])
std_recall = np.std([result['recall'] for result in fold_results])

print(f'Average - accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}, '
      f'f1: {avg_f1:.4f} ± {std_f1:.4f}, '
      f'mcc: {avg_mcc:.4f} ± {std_mcc:.4f}, '
      f'precision: {avg_precision:.4f} ± {std_precision:.4f}, '
      f'recall: {avg_recall:.4f} ± {std_recall:.4f}, ')

output_file = f"../result/{args.dataset_name}/{args.model_name}_results.txt"

output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_file, "a") as f:
    f.write(f"Results for model: {args.model_name}\n")

    for result in fold_results:
        f.write(f"Fold {result['fold']} - accuracy: {result['accuracy']:.4f}, "
                f"f1: {result['f1']:.4f}, "
                f"mcc: {result['mcc']:.4f}, "
                f"precision: {result['precision']:.4f}, "
                f"recall: {result['recall']:.4f}\n")

    f.write(f"Average - accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}, "
            f"f1: {avg_f1:.4f} ± {std_f1:.4f}, "
            f"mcc: {avg_mcc:.4f} ± {std_mcc:.4f}, "
            f"precision: {avg_precision:.4f} ± {std_precision:.4f}, "
            f"recall: {avg_recall:.4f} ± {std_recall:.4f}\n\n")