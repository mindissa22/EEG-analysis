import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from mne import create_info
from mne.io import RawArray
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange

# Preprocess EEG Data
def preprocess_eeg_data(base_path):
    data_files = {
        'Control': [],
        'ADHD': []
    }
    group_labels = {
        'Control': [],
        'ADHD': []
    }

    # Defining parameters
    sfreq = 128  # Sampling frequency
    ch_names = [f'EEG{i+1}' for i in range(19)]  # Placeholder channels

    def load_and_filter_file(file_path):
        data = sio.loadmat(file_path)
        keys = list(data.keys())
        key = keys[-1]
        eeg_data = data[key]

        # Creating MNE RawArray object
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw_data = RawArray(eeg_data.T, info)  # Transposing to match MNE format (channels x samples)

        # Applying the bandpass filter 1-48Hz
        raw_data.filter(l_freq=1.0, h_freq=48.0, fir_design='firwin')

        # Splitting into chunks of 512 points (4 seconds per chunk at 128Hz)
        n_chunks = raw_data.get_data().shape[1] // 512
        eeg_chunks = np.array_split(raw_data.get_data()[:, :n_chunks * 512], n_chunks, axis=1)

        return eeg_chunks

    # Processing each file
    for folder in ['Control_part1', 'Control_part2', 'ADHD_part1', 'ADHD_part2']:
        folder_path = os.path.join(base_path, folder)
        label = 0 if 'Control' in folder else 1  # Control = 0, ADHD = 1
        for file in os.listdir(folder_path):
            if file.endswith('.mat'):
                file_path = os.path.join(folder_path, file)
                try:
                    filtered_chunks = load_and_filter_file(file_path)
                    data_files['Control' if label == 0 else 'ADHD'].extend(filtered_chunks)
                    group_labels['Control' if label == 0 else 'ADHD'].extend([file] * len(filtered_chunks))
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    control_data = np.stack(data_files['Control'], axis=0)
    adhd_data = np.stack(data_files['ADHD'], axis=0)
    control_group_labels = np.array(group_labels['Control'])
    adhd_group_labels = np.array(group_labels['ADHD'])

    return control_data, adhd_data, control_group_labels, adhd_group_labels

# Standardizing data
def standardize_data(train_data, test_data):
    scaler = StandardScaler()
    train_data_reshaped = train_data.reshape(-1, train_data.shape[-1])
    scaler.fit(train_data_reshaped)
    train_data_standardized = scaler.transform(train_data_reshaped).reshape(train_data.shape)
    test_data_standardized = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)
    return train_data_standardized, test_data_standardized

# EEG Dataset class
class EEGDataset(Dataset):
    def __init__(self, data, labels, group_labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.group_labels = group_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.group_labels[idx]

# Conformer Model Components
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 20), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (1, 15), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 20), (1, 10)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        if x.dim() == 3:  # Handle cases where x is 3D
            x = x.unsqueeze(1)  # Adding a dummy channel dimension
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

    def forward(self, x, mask=None):
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)

        # Rearranging tensors to split into multiple heads
        queries = rearrange(queries, "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(keys, "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(values, "b n (h d) -> b h n d", h=self.num_heads)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
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

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.clshead(x)
        return out

class Conformer(nn.Module):
    def __init__(self, emb_size=40, depth=6, n_classes=2):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size)
        self.transformer = TransformerEncoder(depth, emb_size)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.classifier(x)
        return x

# Model Training and evaluation
def train_and_evaluate(model, train_loader, test_loader, group_labels_test, n_epochs=20, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        for data, labels, _ in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate model on test data
        model.eval()
        correct = 0
        total = 0
        group_correct = {group: 0 for group in np.unique(group_labels_test)}
        group_total = {group: 0 for group in np.unique(group_labels_test)}

        with torch.no_grad():
            for data, labels, group_labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for i, group_label in enumerate(group_labels):
                    if predicted[i] == labels[i]:
                        group_correct[group_label] += 1
                    group_total[group_label] += 1

        print(f'Epoch {epoch+1}/{n_epochs}, Accuracy: {100 * correct / total}%')

        for group in group_correct:
            print(f'Group {group}, Accuracy: {100 * group_correct[group] / group_total[group]}%')

# Main function
def main(base_path):
    # Load and preprocess data
    control_data, adhd_data, control_group_labels, adhd_group_labels = preprocess_eeg_data(base_path)

    # Create labels: 0 for Control, 1 for ADHD
    control_labels = np.zeros(control_data.shape[0], dtype=int)
    adhd_labels = np.ones(adhd_data.shape[0], dtype=int)

    # Split data into training and testing sets (80/20 split)
    train_control = control_data[:int(0.8 * len(control_data))]
    test_control = control_data[int(0.8 * len(control_data)):]
    train_adhd = adhd_data[:int(0.8 * len(adhd_data))]
    test_adhd = adhd_data[int(0.8 * len(adhd_data)):]

    train_labels_control = control_labels[:int(0.8 * len(control_labels))]
    test_labels_control = control_labels[int(0.8 * len(control_labels)):]
    train_labels_adhd = adhd_labels[:int(0.8 * len(adhd_labels))]
    test_labels_adhd = adhd_labels[int(0.8 * len(adhd_labels)):]

    train_group_labels_control = control_group_labels[:int(0.8 * len(control_group_labels))]
    test_group_labels_control = control_group_labels[int(0.8 * len(control_group_labels)):]
    train_group_labels_adhd = adhd_group_labels[:int(0.8 * len(adhd_group_labels))]
    test_group_labels_adhd = adhd_group_labels[int(0.8 * len(adhd_group_labels)):]

    train_data = np.concatenate((train_control, train_adhd), axis=0)
    test_data = np.concatenate((test_control, test_adhd), axis=0)
    train_labels = np.concatenate((train_labels_control, train_labels_adhd), axis=0)
    test_labels = np.concatenate((test_labels_control, test_labels_adhd), axis=0)
    train_group_labels = np.concatenate((train_group_labels_control, train_group_labels_adhd), axis=0)
    test_group_labels = np.concatenate((test_group_labels_control, test_group_labels_adhd), axis=0)

    # Standardize the data
    train_data, test_data = standardize_data(train_data, test_data)

    # Create Datasets and Dataloaders
    train_dataset = EEGDataset(train_data, train_labels, train_group_labels)
    test_dataset = EEGDataset(test_data, test_labels, test_group_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model and start training
    model = Conformer()
    train_and_evaluate(model, train_loader, test_loader, test_group_labels)

if __name__ == "__main__":
    base_path = "/Users/minolidissanayake/Desktop/Keele/Modules/SEM 3/Outputs"  
    main(base_path)

