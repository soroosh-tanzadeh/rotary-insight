import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_BiLSTM(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, hidden_size=128, lstm_layers=2):
        super(CNN_BiLSTM, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        
        self.pool  = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.relu  = nn.ReLU()
    
        # BiLSTM layer
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True)
        
        # Fully connected
        self.fc = nn.Linear(hidden_size*2, num_classes)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Prepare for LSTM: [batch, seq_len, features]
        x = x.permute(0, 2, 1)
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Global Average Pooling over sequence dimension
        x = x.mean(dim=1)
       
        # Fully connected
        x = self.fc(x)
        return x