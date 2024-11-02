import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, batch_norm=True):
        super(Bottleneck, self).__init__()
        self.batch_norm = batch_norm

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        # Shortcut layer
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            if self.batch_norm:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * self.expansion)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                )

    def forward(self, x):
        out = self.conv1(x)

        if self.batch_norm:
            out = self.bn1(out)

        out = torch.relu(out)
        out = self.conv2(out)

        if self.batch_norm:
            out = self.bn2(out)

        out = torch.relu(out)
        out = self.conv3(out)
        if self.batch_norm:
            out = self.bn3(out)

        out += self.shortcut(x)
        out = torch.relu(out)
        return out