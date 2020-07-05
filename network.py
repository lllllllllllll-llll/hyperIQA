import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import math

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class res_diy(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(res_diy, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.avgpool(layer4)

        return layer1, layer2, layer3, layer4, x

resnet = models.resnet50(pretrained=True)
res = res_diy(Bottleneck, [3, 4, 6, 3])

pretrained_dict = resnet.state_dict()
model_dict = res.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
res.load_state_dict(model_dict)

class hyperIQA(nn.Module):
    def __init__(self):
        super(hyperIQA, self).__init__()

        self.res = res

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)

        self.x1_conv = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=3, padding=0)
        self.x1_fc = nn.Linear(256, 512)
        self.x2_conv = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=3, padding=0)
        self.x2_fc = nn.Linear(256, 256)
        self.x3_conv = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, stride=3, padding=0)
        self.x3_fc = nn.Linear(256, 128)
        self.x4_conv = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=3, padding=0)
        self.x4_fc = nn.Linear(256, 64)

        self.LocalAM1 = nn.Conv2d(in_channels=256*56*56, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.LocalAM2 = nn.Conv2d(in_channels=512*28*28, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.LocalAM3 = nn.Conv2d(in_channels=1024*14*14, out_channels=256, kernel_size=1, stride=1, padding=0)

        self.fc1 = nn.Linear(2816, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, input):
        input = input.view(-1, input.size(-3), input.size(-2), input.size(-1))
        layer1, layer2, layer3, layer4, x = self.res(input)

        batchsize = x.size(0)

        x = x.squeeze(3).squeeze(2)

        y1 = torch.reshape(layer1, (batchsize, 256*56*56, 1, 1))
        y1 = self.LocalAM1(y1)
        y1 = y1.squeeze(3).squeeze(2)

        y2 = torch.reshape(layer2, (batchsize, 512*28*28, 1, 1))
        y2 = self.LocalAM2(y2)
        y2 = y2.squeeze(3).squeeze(2)

        y3 = torch.reshape(layer3, (batchsize, 1024*14*14, 1, 1))
        y3 = self.LocalAM3(y3)
        y3 = y3.squeeze(3).squeeze(2)

        x0 = self.conv1(layer4)
        x0 = self.conv2(x0)
        x0 = self.conv3(x0)

        x1_w = self.x1_conv(x0)
        x1_w = torch.reshape(x1_w, (batchsize, 2*2*128, 1, 1))
        x1_w = x1_w.squeeze(3).squeeze(2)

        x1_fc = self.gap(x0)
        x1_fc = x1_fc.squeeze(3).squeeze(2)
        x1_b = self.x1_fc(x1_fc)

        x2_w = self.x2_conv(x0)
        x2_w = torch.reshape(x2_w, (batchsize, 2 * 2 * 64, 1, 1))
        x2_w = x2_w.squeeze(3).squeeze(2)
        x2_fc = self.gap(x0)
        x2_fc = x2_fc.squeeze(3).squeeze(2)
        x2_b = self.x2_fc(x2_fc)

        x3_w = self.x3_conv(x0)
        x3_w = torch.reshape(x3_w, (batchsize, 2 * 2 * 32, 1, 1))
        x3_w = x3_w.squeeze(3).squeeze(2)
        x3_fc = self.gap(x0)
        x3_fc = x3_fc.squeeze(3).squeeze(2)
        x3_b = self.x3_fc(x3_fc)

        x4_w = self.x4_conv(x0)
        x4_w = torch.reshape(x4_w, (batchsize, 2 * 2 * 16, 1, 1))
        x4_w = x4_w.squeeze(3).squeeze(2)
        x4_fc = self.gap(x0)
        x4_fc = x4_fc.squeeze(3).squeeze(2)
        x4_b = self.x4_fc(x4_fc)

        fc = torch.cat((y1, y2, y3, x), 1)

        fc1 = self.fc1(fc)
        fc1 = torch.mul(fc1, x1_w)
        fc1 = torch.add(fc1, x1_b)

        fc2 = self.fc2(fc1)
        fc2 = torch.mul(fc2, x2_w)
        fc2 = torch.add(fc2, x2_b)

        fc3 = self.fc3(fc2)
        fc3 = torch.mul(fc3, x3_w)
        fc3 = torch.add(fc3, x3_b)

        fc4 = self.fc4(fc3)
        fc4 = torch.mul(fc4, x4_w)
        fc4 = torch.add(fc4, x4_b)

        q = self.fc5(fc4)

        return q

