import torch
import torch.nn as nn
import torch.nn.functional as F

class WideResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, widen_factor=2, dropout_rate=0.2):
        super(WideResidualBlock, self).__init__()
        width = out_channels * widen_factor
        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv2 = nn.Conv1d(width, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if stride != 1 or in_channels != out_channels
            else None
        )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate) #  Dropout

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out) #  Dropout 

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Skip Connection
        
        return self.relu(out)


class WRN1D(nn.Module):
    def __init__(self, num_classes=10, widen_factor=2, depth=22, dropout_rate=0.2):
        super(WRN1D, self).__init__()
        self.in_channels = 16
        
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(WideResidualBlock, 16, depth // 6, stride=1, widen_factor=widen_factor, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(WideResidualBlock, 32, depth // 6, stride=2, widen_factor=widen_factor, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(WideResidualBlock, 64, depth // 6, stride=2, widen_factor=widen_factor, dropout_rate=dropout_rate)

        self.avgpool = nn.AdaptiveAvgPool1d(1)  #  Global Average Pooling
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride, widen_factor, dropout_rate):
        layers = [block(self.in_channels, out_channels, stride=stride, widen_factor=widen_factor, dropout_rate=dropout_rate)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1, widen_factor=widen_factor, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x