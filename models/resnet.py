import torch
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet, Bottleneck
import torch.nn as nn


class ResNetClassfier(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetClassfier, self).__init__()
        self.resize = transforms.Resize((64, 64))
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, padding="same")
        self.resent = ResNet(
            block=Bottleneck,
            layers=[1, 1, 1, 1],
            norm_layer=nn.BatchNorm2d,
            num_classes=num_classes,
        )

    def forward(self, x):
        x = x.squeeze(1)
        x = torch.stft(
            x,
            n_fft=512,
            win_length=512,
            hop_length=512,
            center=False,
            return_complex=True,
        )
        spectrogram = torch.abs(x)
        x = self.resize(spectrogram)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        return self.resent(x)


if __name__ == "__main__":
    model = ResNetClassfier()
    x = torch.randn(32, 1, 2048)
    y = model(x)
    print(y.shape)
