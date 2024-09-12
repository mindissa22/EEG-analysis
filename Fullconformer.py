# DATA LOADING
# Data loading
from pathlib import Path
import scipy.io as sio

import numpy as np
import pandas as pd

data_dir = Path.home() / Path("MyData/EEG_ADHD")
# Channel labels and electrode positions
channel_labels = {
    0: "Fp1",
    1: "Fp2",
    2: "F3",
    3: "F4",
    4: "C3",
    5: "C4",
    6: "P3",
    7: "P4",
    8: "O1",
    9: "O2",
    10: "F7",
    11: "F8",
    12: "T7",
    13: "T8",
    14: "P7",
    15: "P8",
    16: "Fz",
    17: "Cz",
    18: "Pz",
}

electrode_positions = {
    "Fp1": (-18, 0.511, 0.95, 0.309, -0.0349, 18, -2, 1),
    "Fp2": (18, 0.511, 0.95, -0.309, -0.0349, -18, -2, 1),
    "F7": (-54, 0.511, 0.587, 0.809, -0.0349, 54, -2, 1),
    "F3": (-39, 0.333, 0.673, 0.545, 0.5, 39, 30, 1),
    "Fz": (0, 0.256, 0.719, 0, 0.695, 0, 44, 1),
    "F4": (39, 0.333, 0.673, -0.545, 0.5, -39, 30, 1),
    "F8": (54, 0.511, 0.587, -0.809, -0.0349, -54, -2, 1),
    "T7": (-90, 0.511, 6.12e-17, 0.999, -0.0349, 90, -2, 1),
    "C3": (-90, 0.256, 4.4e-17, 0.719, 0.695, 90, 44, 1),
    "Cz": (90, 0, 3.75e-33, -6.12e-17, 1, -90, 90, 1),
    "C4": (90, 0.256, 4.4e-17, -0.719, 0.695, -90, 44, 1),
    "T8": (90, 0.511, 6.12e-17, -0.999, -0.0349, -90, -2, 1),
    "P7": (-126, 0.511, -0.587, 0.809, -0.0349, 126, -2, 1),
    "P3": (-141, 0.333, -0.673, 0.545, 0.5, 141, 30, 1),
    "Pz": (180, 0.256, -0.719, -8.81e-17, 0.695, -180, 44, 1),
    "P4": (141, 0.333, -0.673, -0.545, 0.5, -141, 30, 1),
    "P8": (126, 0.511, -0.587, -0.809, -0.0349, -126, -2, 1),
    "O1": (-162, 0.511, -0.95, 0.309, -0.0349, 162, -2, 1),
    "O2": (162, 0.511, -0.95, -0.309, -0.0349, -162, -2, 1),
}

# Sampling Frequency Hz
Sampling_Frequency = 128

# Set the chunk size
chunk_size = 512
def split_into_chunks(df, chunk_size, initial_chunk_number=0):
    # Calculate the number of full chunks
    n_chunks = len(df) // chunk_size
    chunks = []

    # Split into chunks and keep track of the chunk number
    for i in range(n_chunks):
        chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size].copy()  # Get the chunk
        chunk["chunk_number"] = (
            initial_chunk_number + i
        )  # Add the chunk number as a new column
        chunks.append(chunk)

    # Concatenate the chunks back together
    return pd.concat(chunks, ignore_index=True), n_chunks
def load_data(data_dirs):
    data_list = []
    chunked_data_list = []

    chunk_index = 0
    for directory in data_dirs:
        # print(f"Loading data from {directory}")

        for filepath in directory.glob("*.mat"):
            mat = sio.loadmat(filepath)
            key = list(mat.keys())[-1]  # Get the last key (the id of the patient)
            eeg_data = mat[key]

            # Convert the EEG data to a DataFrame
            # Assuming the EEG data is a 2D array (time x channels)
            df = pd.DataFrame(eeg_data)
            df = df.rename(columns=channel_labels)
            # Add a column to identify the source
            df["subject_id"] = key

            # print(f"Loaded data for patient {key}; chunks start at {chunk_index}")
            chucked_df, chunks = split_into_chunks(df, chunk_size, chunk_index)
            chunk_index += chunks

            # Append the DataFrame to the list
            data_list.append(df)
            chunked_data_list.append(chucked_df)

        # Concatenate all DataFrames in the list into a single DataFrame
        full_eeg_df = pd.concat(data_list, ignore_index=True)
        chunked_eeg_df = pd.concat(chunked_data_list, ignore_index=True)

    return full_eeg_df, chunked_eeg_df
adhd_dir1 = data_dir / Path("ADHD_part1")
adhd_dir2 = data_dir / Path("ADHD_part2")
adhd_df, adhd_chunks_df = load_data([adhd_dir1, adhd_dir2])

control_dir1 = data_dir / Path("Control_part1")
control_dir2 = data_dir / Path("Control_part2")
control_df, control_chunks_df = load_data([control_dir1, control_dir2])

#ORGANISATION OF DATA INTO CHUNKS
# Let's put all the data together (we will use the "chunked" data)
adhd_chunks_df["label"] = "ADHD"
control_chunks_df["label"] = "Control"

# we need to "renumber" the chunks for the control data so that they are continuous with the ADHD data
control_chunks_df["chunk_number"] += adhd_chunks_df["chunk_number"].max() + 1

all_data_df = pd.concat([adhd_chunks_df, control_chunks_df], ignore_index=True)

print(all_data_df["chunk_number"].nunique())

chunks_df = (
    all_data_df[["chunk_number", "subject_id", "label"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
chunks_df = pd.get_dummies(chunks_df, columns=["label"])
print(chunks_df.shape)

# Let's reshape the data into a 3D array (chunks x channels x samples)
import einops

raw_data = all_data_df.drop(columns=["subject_id", "label"])
# raw_data.info()

# 1. Extract the data (excluding the 'chunk_number' column)
raw_data = raw_data.iloc[:, :-1].values  # shape will be (#tot_samples, 19)

# 2. Reshape the data into a 3D array
num_chunks = chunks_df.shape[0]
eeg_data = einops.rearrange(
    raw_data,
    "(chunks points) electrodes -> chunks electrodes points",
    chunks=num_chunks,
    points=chunk_size,
)

print(eeg_data.shape)

# PREPARATION OF THE DATA FOR MODEL TRAINING
X = eeg_data
# expand the dimensions to make it compatible with the Conv1D layer in Keras
# the new shape will be (#tot_samples, 1, 19, 512)
X = np.expand_dims(X, axis=1)

y = chunks_df[["label_ADHD", "label_Control"]].values

# Set the random seed for reproducibility
np.random.seed(42)

# 1. Generate a random permutation of the indices
indices = np.random.permutation(len(X))

# 2. Use the first 80% of the indices for the training set
# train_size = int(0.8 * len(X))
train_size = 3695

# 3. Split the indices into train and test sets
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
X_test = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

chunks_train = chunks_df.iloc[train_indices].values
chunks_test = chunks_df.iloc[test_indices].values

print(X_train.shape, y_train.shape, chunks_train.shape)
print(X_test.shape, y_test.shape, chunks_test.shape)

# Transform all labels to proper one-hot encoding (integers instead of booleans) - not needed
# y_train = y_train.astype(int)
# y_test = y_test.astype(int)

Group_train = chunks_train[:, [0, 1]]
Group_test = chunks_test[:, [0, 1]]

label_distr_counts_train = np.sum(y_train, axis=0)
label_distr_counts_test = np.sum(y_test, axis=0)

print(f"Training set: {label_distr_counts_train[0]} ADHD /  {label_distr_counts_train[1]} Control")
print(f"Test set: {label_distr_counts_test[0]} ADHD /  {label_distr_counts_test[1]} Control")

#CONFORMER 
import argparse
import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.backends import cudnn
from einops.layers.torch import Rearrange, Reduce

# Enable deterministic mode for reproducibility
cudnn.benchmark = False
cudnn.deterministic = True

# Convolution module
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out

class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )

class ExP():
    def __init__(self, nsub):
        super(ExP, self).__init__()
        self.batch_size = 72
        self.n_epochs = 2000
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        self.nSub = nsub
        self.start_epoch = 0
        self.root = '/Users/minolidissanayake/Desktop/Keele/Modules/SEM 3/Dataset'

        self.log_write = open(f"./results/log_subject{self.nSub}.txt", "w")
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = Conformer().cuda()
        self.model = nn.DataParallel(self.model)
        self.model = self.model.cuda()

    def get_source_data(self):
        self.total_data = scipy.io.loadmat(self.root + f'A0{self.nSub}T.mat')
        self.train_data = self.total_data['data']  
        self.train_label = self.total_data['label']  

        self.train_data = np.transpose(self.train_data, (2, 1, 0))
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)

        self.allData = self.train_data
        self.allLabel = self.train_label[0]

        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        self.test_tmp = scipy.io.loadmat(self.root + f'A0{self.nSub}E.mat')
        self.test_data = self.test_tmp['data']  
        self.test_label = self.test_tmp['label']  

        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label[0]

        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        # Split the data into training and testing sets
        self.train_size = int(0.8 * len(self.allData))
        indices = np.random.permutation(len(self.allData))
        self.train_indices = indices[:self.train_size]
        self.test_indices = indices[self.train_size:]

        self.allData = self.allData[self.train_indices]
        self.allLabel = self.allLabel[self.train_indices]
        self.testData = self.testData[self.test_indices]
        self.testLabel = self.testLabel[self.test_indices]

        return self.allData, self.allLabel, self.testData, self.testLabel

    def train(self):
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.allData, self.allLabel, self.testData, self.testLabel = self.get_source_data()
        train_data = torch.from_numpy(self.allData).float().cuda()
        train_label = torch.from_numpy(self.allLabel).long().cuda()
        test_data = torch.from_numpy(self.testData).float().cuda()
        test_label = torch.from_numpy(self.testLabel).long().cuda()

        train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.start_epoch, self.n_epochs):
            self.model.train()

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = Variable(data).cuda(), Variable(target).cuda()
                self.optimizer.zero_grad()
                output, logits = self.model(data)
                loss_cls = self.criterion_cls(logits, target)
                loss_cls.backward()
                self.optimizer.step()

            print(f'Epoch {epoch + 1}/{self.n_epochs}, Loss: {loss_cls.item()}')

        torch.save(self.model.state_dict(), f"./results/conformer_subject{self.nSub}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=int, required=True, help='Subject number for training')
    args = parser.parse_args()

    exp = ExP(args.subject)
    exp.train()
