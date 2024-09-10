import argparse
import os
import numpy as np
import math
import glob
import random
import datetime
import time
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn  # Correct import for cudnn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision.transforms import Compose, Resize, ToTensor
from torch import Tensor


# Configure CUDA
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
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
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fitting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
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

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=10, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class ClassificationHead(nn.Module):
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

class Conformer(nn.Module):
    def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size)
        self.transformer_encoder = TransformerEncoder(depth, emb_size)
        self.classification_head = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        return self.classification_head(x)

class ExP():
    def __init__(self, nsub):
        self.batch_size = 72
        self.n_epochs = 2000
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        self.nSub = nsub
        self.start_epoch = 0
        self.root = '/Data/strict_TE/'
        self.log_write = open(f"./results/log_subject{self.nSub}.txt", "w")
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.model = Conformer().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()

    def get_source_data(self):
        self.total_data = scipy.io.loadmat(f'{self.root}A0{self.nSub}T.mat')
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
        self.test_tmp = scipy.io.loadmat(f'{self.root}A0{self.nSub}E.mat')
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
        return self.allData, self.allLabel, self.testData, self.testLabel

    def train(self):
        allData, allLabel, testData, testLabel = self.get_source_data()
        train_loader = DataLoader(dataset=EEGDataset(allData, allLabel), batch_size=self.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(dataset=EEGDataset(testData, testLabel), batch_size=self.batch_size, shuffle=False, num_workers=4)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        for epoch in range(self.start_epoch, self.n_epochs):
            self.model.train()
            epoch_start_time = time.time()
            train_loss = 0
            for i, (data, labels) in enumerate(train_loader):
                data = data.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                _, output = self.model(data)
                loss = self.criterion_cls(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if i % 100 == 0:
                    print(f"Epoch [{epoch}/{self.n_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item()}")
            avg_train_loss = train_loss / len(train_loader)
            print(f'Epoch [{epoch}/{self.n_epochs}] Loss: {avg_train_loss}')
            self.log_write.write(f'Epoch [{epoch}/{self.n_epochs}] Loss: {avg_train_loss}\n')
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), f"./results/model_epoch_{epoch}.pth")
            self.model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for data, labels in test_loader:
                    data = data.cuda()
                    labels = labels.cuda()
                    _, outputs = self.model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                print(f'Test Accuracy: {100 * correct / total}%')
                self.log_write.write(f'Test Accuracy: {100 * correct / total}%\n')

class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

if __name__ == "__main__":
    # Hardcoded subject number
    subject_number = 1  # Replace with your desired subject number
    exp = ExP(subject_number)
    exp.train()

