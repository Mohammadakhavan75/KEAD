import torch
import torch.nn as nn

# Define the ResNet Block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, batch_norm=True):
        super(BasicBlock, self).__init__()
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if self.batch_norm:
            self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut layer
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            if self.batch_norm:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * out_channels)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                )

    def forward(self, x):

        out = self.conv1(x)

        if self.batch_norm:
            out = self.bn1(out)

        out = torch.relu(out)
        out = self.conv2(out)

        if self.batch_norm:
            out = self.bn2(out)

        out += self.shortcut(x)
        out = torch.relu(out)
        return out