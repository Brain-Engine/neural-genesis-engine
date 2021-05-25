import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.conv1 = nn.Conv2d(self.in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, self.expansion * self.planes, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.expansion * self.planes))

    def forward(self, inputs):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(inputs)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.conv1 = nn.Conv2d(self.in_planes, self.planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.planes)
        self.conv2 = nn.Conv2d(self.planes, self.planes, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.conv3 = nn.Conv2d(self.planes, self.expansion * self.planes, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(self.expansion * self.planes)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, self.expansion * self.planes, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.expansion * self.planes)
            )

    def forward(self, inputs):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(inputs)
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    def __init__(self, block="BasicBlock", blocks_list=None, half=True, in_channel=3):
        super(ResNetBackbone, self).__init__()
        self.in_planes = 64
        block = BasicBlock if block == "BasicBlock" else Bottleneck
        self.blocks_list = [1, 1, 1, 1] if blocks_list is None else Bottleneck
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        if half:
            self.layer1 = self._make_layer(block, 32, self.blocks_list[0], stride=1)
            self.layer2 = self._make_layer(block, 64, self.blocks_list[1], stride=2)
            self.layer3 = self._make_layer(block, 128, self.blocks_list[2], stride=2)
            self.layer4 = self._make_layer(block, 256, self.blocks_list[3], stride=2)
        else:
            self.layer1 = self._make_layer(block, 64, self.blocks_list[0], stride=1)
            self.layer2 = self._make_layer(block, 128, self.blocks_list[1], stride=2)
            self.layer3 = self._make_layer(block, 256, self.blocks_list[2], stride=2)
            self.layer4 = self._make_layer(block, 512, self.blocks_list[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layer_list = []
        for stride in strides:
            layer_list.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layer_list)

    def compute_shape(self, shape, batch_size=1, data_type=torch.float32):
        inputs = torch.ones((batch_size, *shape), dtype=data_type)
        out = self.forward(inputs)
        return out.shape[1:]

    def forward(self, inputs):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
