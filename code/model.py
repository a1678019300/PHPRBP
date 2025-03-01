import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import ADASYN

class SEBlock(nn.Module):
    def __init__(self, channel, reduction):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = x.mean(dim=2)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y.expand_as(x)


class CNNModel(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, kernel_size, output_size, reduction, drop_prob):
        super(CNNModel, self).__init__()

        padding = int((kernel_size - 1) / 2)

        self.conv1_ESM = nn.Sequential(
            nn.Conv1d(in_channels=in_channels1, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
        )
        self.conv1_ProtT5 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels2, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
        )

        self.conv2_ESM = nn.Sequential(
            nn.Conv1d(in_channels=out_channels // 2, out_channels=out_channels // 4, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels // 4),
        )
        self.conv2_ProtT5 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels // 2, out_channels=out_channels // 4, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels // 4),
        )
        self.maxPooling = nn.MaxPool1d(kernel_size=2)
        self.drop = nn.Dropout(drop_prob)

        self.se = SEBlock(out_channels // 4, reduction)

        self.fc1 = nn.Linear(out_channels // 4, out_channels // 8)
        self.fc2 = nn.Linear(out_channels // 8, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_ESM = x[:, :1280]
        x_ProtT5 = x[:, 1280:]
        x_ESM = x_ESM.unsqueeze(2)
        x_ProtT5 = x_ProtT5.unsqueeze(2)
        x_ESM = self.conv1_ESM(x_ESM)
        x_ProtT5 = self.conv1_ProtT5(x_ProtT5)

        x_ESM = x_ESM.view(x_ESM.size(0), -1)
        x_ProtT5 = x_ProtT5.view(x_ProtT5.size(0), -1)


        x_ESM = self.maxPooling(x_ESM)
        x_ProtT5 = self.maxPooling(x_ProtT5)

        x_ESM = x_ESM.unsqueeze(2)
        x_ProtT5 = x_ProtT5.unsqueeze(2)
        x_ESM = self.conv2_ESM(x_ESM)
        x_ProtT5 = self.conv2_ProtT5(x_ProtT5)

        x_ESM = x_ESM.view(x_ESM.size(0), -1)
        x_ProtT5 = x_ProtT5.view(x_ProtT5.size(0), -1)
        x_ESM = self.maxPooling(x_ESM)
        x_ProtT5 = self.maxPooling(x_ProtT5)
        x_concat = torch.cat((x_ESM, x_ProtT5), dim=1)
        x_concat = x_concat.unsqueeze(2)

        x_concat = self.drop(x_concat)
        se_output = self.se(x_concat)
        mlp_input = se_output.view(se_output.size(0), -1)
        x = self.relu(self.fc1(mlp_input))
        x = self.fc2(x)
        return x


def data_augmentation(RBP_combined, RBP_labels):

    label_counts = pd.Series(RBP_labels).value_counts()
    mean_count = label_counts.iloc[1:-1].mean()
    labels_to_augment = label_counts[label_counts < mean_count].index
    sampling_strategy = {label: count * 2 for label, count in label_counts.items() if label in labels_to_augment}
    smote = ADASYN(sampling_strategy=sampling_strategy)
    augmented_embeddings, augmented_labels = smote.fit_resample(RBP_combined, RBP_labels)

    return augmented_embeddings, augmented_labels
