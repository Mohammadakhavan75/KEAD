import torch
import torch.nn as nn
from models.basic_block import BasicBlock
from models.bottel_neck import Bottleneck

# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, img_size=32, batch_norm=True, classification_head=False, num_classes=10, proj_head=False, proj_dim=128):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.classification_head = classification_head
        self.batch_norm = batch_norm
        self.img_size = img_size
        self.proj_dim = proj_dim
        self.proj_head = proj_head


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

        if self.classification_head:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.proj_head:
            self.projection_head = nn.Sequential(
            nn.Linear(512 * block.expansion, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )


    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.batch_norm))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, batch_norm=self.batch_norm))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        if self.img_size == 224:
            out = self.maxpool(out)
        out = torch.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        
        feat = out.view(out.size(0), -1)

        if self.proj_head:
            proj_out = self.projection_head(feat)
            
            return proj_out, feat

        if self.classification_head:
            cls_pred = self.fc(feat)
            
            return cls_pred, feat

        return feat, None

def ResNet18(img_size=32, batch_norm=True, classification_head=False, num_classes=10, proj_head=False, proj_dim=128):
    return ResNet(BasicBlock, [2, 2, 2, 2], img_size=img_size,  batch_norm=batch_norm, classification_head=classification_head, num_classes=num_classes, proj_head=proj_head, proj_dim=proj_dim)

# Function to instantiate ResNet-50
def ResNet50(img_size=32, batch_norm=True, classification_head=False, num_classes=10, proj_head=False, proj_dim=128):
    return ResNet(Bottleneck, [3, 4, 6, 3], img_size=img_size,  batch_norm=batch_norm, classification_head=classification_head, num_classes=num_classes, proj_head=proj_head, proj_dim=proj_dim)
