import torch
import torch.nn as nn
from models.basic_block import BasicBlock
from models.bottel_neck import Bottleneck

# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, img_size=32, batch_norm=True, fc_available=False, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.fc_available = fc_available
        self.batch_norm = batch_norm
        self.img_size = img_size
        if img_size == 224:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # for 224 resolution
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # for 224 resolution
        elif img_size == 32:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            raise NotImplementedError("img_size is not implemented: {}".format(img_size))


        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(64)



        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.batch_norm))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, batch_norm=self.batch_norm))
        return nn.Sequential(*layers)

    def forward(self, x):
        out_list = []
        out = self.conv1(x)
        out_list.append(out)
        if self.batch_norm:
            out = self.bn1(out)
            out_list.append(out)
        if self.img_size==224:
            out = self.maxpool(out)
            out_list.append(out)
        out = torch.relu(out)
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = self.avgpool(out)
        out_list.append(out)
        out = out.view(out.size(0), -1)
        out_list.append(out)

        if self.fc_available:
            out = self.fc(out)

        return out, out_list

def ResNet18(img_size=32, batch_norm=True, fc_available=False, num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], img_size=img_size,  batch_norm=batch_norm, fc_available=fc_available, num_classes=num_classes)

# Function to instantiate ResNet-50
def ResNet50(img_size=32, batch_norm=True, fc_available=False, num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], img_size=img_size,  batch_norm=batch_norm, fc_available=fc_available, num_classes=num_classes)
