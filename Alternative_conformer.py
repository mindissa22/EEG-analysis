import argparse
import os
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops.layers.torch import Rearrange
from mne import create_info
from mne.io import RawArray

# Preprocessing functions
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

# Model Components
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, x):
        x = x.permute(2, 0, 1)  
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 2, 0)  
        return x


class ResidualAdd(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x):
        return x + self.layer(x)

class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.residual1 = ResidualAdd(self.attention)
        self.ff = FeedForwardBlock(embed_dim, ff_dim, dropout)
        self.residual2 = ResidualAdd(self.ff)
    
    def forward(self, x):
        x = self.residual1(x)
        x = self.residual2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Conformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels=19, out_channels=64)
        self.transformer_encoder = TransformerEncoder(embed_dim=2000, num_layers=4, num_heads=8, ff_dim=128)
        self.classification_head = ClassificationHead(embed_dim=2000, num_classes=num_classes)
    
    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.permute(0, 2, 1)  # Permute for transformer
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.classification_head(x)
        return x

# Experiment Class
class ExP:
    def __init__(self, subject):
        self.subject = subject
        self.base_path = "/Users/minolidissanayake/Desktop/Keele/Modules/SEM 3/Dataset"
        self.num_classes = 2  # Control and ADHD

        # Preprocess and load data
        control_data, adhd_data = preprocess_eeg_data(self.base_path)
        self.train_data, self.test_data = self.split_data(control_data, adhd_data)
        self.train_data, self.test_data = self.standardize_data(self.train_data, self.test_data)
        
        # Create datasets
        self.train_dataset = EEGDataset(self.train_data, [0] * len(control_data) + [1] * len(adhd_data))
        self.test_dataset = EEGDataset(self.test_data, [0] * len(control_data) + [1] * len(adhd_data))

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

        # Model, loss, optimizer
        self.model = Conformer(num_classes=self.num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.7)
        self.loss_fn = nn.CrossEntropyLoss()

    def split_data(self, control_data, adhd_data):
        split_idx_control = int(len(control_data) * 0.8)
        split_idx_adhd = int(len(adhd_data) * 0.8)
        
        train_control = control_data[:split_idx_control]
        test_control = control_data[split_idx_control:]
        train_adhd = adhd_data[:split_idx_adhd]
        test_adhd = adhd_data[split_idx_adhd:]

        train_data = np.concatenate([train_control, train_adhd], axis=0)
        test_data = np.concatenate([test_control, test_adhd], axis=0)

        return train_data, test_data

    def standardize_data(self, train_data, test_data):
        scaler = StandardScaler()
        train_data_reshaped = train_data.reshape(-1, train_data.shape[-1])
        scaler.fit(train_data_reshaped)
        train_data_standardized = scaler.transform(train_data_reshaped).reshape(train_data.shape)
        test_data_standardized = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)
        return train_data_standardized, test_data_standardized

    def train(self, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            for batch_data, batch_labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.loss_fn(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

            # Save model checkpoint
            torch.save(self.model.state_dict(), f'model_epoch_{epoch+1}.pth')

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data, batch_labels in self.test_loader:
                outputs = self.model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        accuracy = correct / total
        print(f'Accuracy: {accuracy * 100:.2f}%')
        return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Conformer model on EEG data.')
    parser.add_argument('subject', type=int, help='Subject number')
    args = parser.parse_args()

    exp = ExP(subject=args.subject)
    exp.train(epochs=10)
    exp.evaluate()