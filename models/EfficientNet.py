import torch
import torch.nn as nn
import torch.nn.functional as F

class MBConv1D(nn.Module):
    """ Mobile Inverted Bottleneck Convolution (MBConv) for 1D signals. """
    def __init__(self, in_channels, out_channels, expansion=6, stride=1):
        super(MBConv1D, self).__init__()
        hidden_dim = in_channels * expansion
        self.use_residual = in_channels == out_channels and stride == 1

        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_residual:
            out += identity  # Skip Connection
        return self.relu(out)
    
class EfficientNet1D(nn.Module):
    """ EfficientNet1D model using MBConv blocks. """
    def __init__(self, num_classes=10):
        super(EfficientNet1D, self).__init__()
        self.init_conv = nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.init_bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)

        self.mbconv1 = MBConv1D(32, 64, expansion=6, stride=2)
        self.mbconv2 = MBConv1D(64, 128, expansion=6, stride=2)
        self.mbconv3 = MBConv1D(128, 256, expansion=6, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.relu(x)

        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)