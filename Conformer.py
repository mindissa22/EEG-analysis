import os
import numpy as np
import scipy.io as sio
import mne
from mne import create_info
from mne.io import RawArray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from einops.layers.torch import Rearrange, Reduce
import matplotlib.pyplot as plt

def preprocess_eeg_data(base_path):
    data_files = {
        'Control': [],
        'ADHD': []
    }

    # Defining parameters
    sfreq = 256  # Sampling frequency
    ch_names = [f'EEG{i+1}' for i in range(19)]  # Placeholder channels

    def load_and_filter_file(file_path):
        data = sio.loadmat(file_path)
        keys = list(data.keys())
        print(f"Keys in {file_path}: {keys}")

        # Printing type and shape of each key
        for key in keys:
            print(f"{key}: {type(data[key])}, shape: {data[key].shape if hasattr(data[key], 'shape') else 'N/A'}")
        
        # Assuming EEG data is in the last key
        key = keys[-1]

        if key not in data:
            raise KeyError(f"Expected key '{key}' not found in {file_path}")

        eeg_data = data[key]
        print(f"Data shape: {eeg_data.shape}")

        # Creating MNE RawArray object
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw_data = RawArray(eeg_data.T, info)  # Transposing to match MNE format (channels x samples)
        
        # Applying bandpass filter
        raw_data.filter(l_freq=1.0, h_freq=48.0, fir_design='firwin')
        
        return raw_data.get_data()

    # Processing each file
    for folder in ['Control_part1', 'Control_part2', 'ADHD_part1', 'ADHD_part2']:
        folder_path = os.path.join(base_path, folder)
        label = 0 if 'Control' in folder else 1  # Control = 0, ADHD = 1
        for file in os.listdir(folder_path):
            if file.endswith('.mat'):
                file_path = os.path.join(folder_path, file)
                try:
                    filtered_data = load_and_filter_file(file_path)
                    data_files['Control' if label == 0 else 'ADHD'].append(filtered_data)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Checking if data_files is empty
    if not data_files['Control'] or not data_files['ADHD']:
        raise ValueError("No data found for Control or ADHD. Check the file paths and contents.")

    # Determining the minimum length for consistent data shape
    min_length = min(
        min(d.shape[1] for d in data_files['Control']), 
        min(d.shape[1] for d in data_files['ADHD'])
    )

    # Truncating data to ensure consistent length
    def truncate_or_pad(data_list, length):
        truncated_data = [
            d[:, :length] if d.shape[1] > length 
            else np.pad(d, ((0, 0), (0, length - d.shape[1])), mode='constant') 
            for d in data_list
        ]
        return np.stack(truncated_data, axis=0)

    control_data = truncate_or_pad(data_files['Control'], min_length)
    adhd_data = truncate_or_pad(data_files['ADHD'], min_length)
    
    print(f"Control data shape after preprocessing: {control_data.shape}")
    print(f"ADHD data shape after preprocessing: {adhd_data.shape}")
    
    return control_data, adhd_data


# Standardising data
def standardize_data(train_data, test_data):
    scaler = StandardScaler()
    train_data_reshaped = train_data.reshape(-1, train_data.shape[-1])
    scaler.fit(train_data_reshaped)
    train_data_standardized = scaler.transform(train_data_reshaped).reshape(train_data.shape)
    test_data_standardized = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)
    return train_data_standardized, test_data_standardized

# Defining the EEGDataset class
class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Defining the Conformer model components
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
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x

from einops import rearrange

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
def train_and_evaluate(model, train_loader, test_loader, n_epochs=20, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}')

        # Evaluating on the test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')

# Main function
def main():
    base_path = '/Users/minolidissanayake/Desktop/Keele/Modules/SEM 3/Dataset'  
    control_data, adhd_data = preprocess_eeg_data(base_path)
    
    # Creating labels: 0 for Control, 1 for ADHD
    control_labels = np.zeros(control_data.shape[0])
    adhd_labels = np.ones(adhd_data.shape[0])
    
    # Combining data and labels
    data = np.concatenate((control_data, adhd_data), axis=0)
    labels = np.concatenate((control_labels, adhd_labels), axis=0)
    
    # Standardiding the data
    data_standardized, _ = standardize_data(data, data)
    
    # Splitting data into training and testing sets
    split_index = int(0.8 * len(data_standardized))
    train_data, test_data = data_standardized[:split_index], data_standardized[split_index:]
    train_labels, test_labels = labels[:split_index], labels[split_index:]
    
    # Creating datasets and dataloaders
    train_dataset = EEGDataset(train_data, train_labels)
    test_dataset = EEGDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialising and train the model
    model = Conformer()
    train_and_evaluate(model, train_loader, test_loader)

if __name__ == "__main__":
    main()

