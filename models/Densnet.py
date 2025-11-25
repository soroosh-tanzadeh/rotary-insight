import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.Conv1d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(growth_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        outputs = [x]
        for layer in self.layers:
            new_out = layer(torch.cat(outputs, dim=1))  # Concatenate previous outputs
            outputs.append(new_out)
        return torch.cat(outputs, dim=1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.pool(x)


class DenseNet1D(nn.Module):
    def __init__(self, num_classes=10, growth_rate=16, num_layers=[4, 4, 4], init_channels=16):
        super(DenseNet1D, self).__init__()
        self.init_conv = nn.Conv1d(1, init_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.init_bn = nn.BatchNorm1d(init_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Dense Block 1
        self.dense1 = DenseBlock(init_channels, growth_rate, num_layers[0])
        in_channels = init_channels + num_layers[0] * growth_rate
        self.trans1 = TransitionLayer(in_channels, in_channels // 2)

        # Dense Block 2
        in_channels = in_channels // 2
        self.dense2 = DenseBlock(in_channels, growth_rate, num_layers[1])
        in_channels += num_layers[1] * growth_rate
        self.trans2 = TransitionLayer(in_channels, in_channels // 2)

        # Dense Block 3
        in_channels = in_channels // 2
        self.dense3 = DenseBlock(in_channels, growth_rate, num_layers[2])
        in_channels += num_layers[2] * growth_rate

        # Final Layers
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten before FC layer
        return self.fc(x)
    