import torch.nn as nn
import torch.nn.functional as F
import torch
import math


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'G': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'AC256', 'M', 512, 512, 512, 512, 'AC512', 'M',
          512, 512, 512, 512, 'AC512', 'M'],
    'H': [64, 64, 'AS64', 'M', 128, 128, 'AS128', 'M', 256, 256, 256, 256, 'AC256', 'M', 512, 512, 512, 512, 'AC512', 'M',
          512, 512, 512, 512, 'AC512', 'M'],
    'I': [64, 64, 'AS64', 'M', 128, 128, 'AS128', 'M', 256, 256, 256, 256, 'AS256', 'M', 512, 512, 512, 512, 'AS512', 'M',
          512, 512, 512, 512, 'AS512', 'M'],
    'J': [64, 64, 'AS64', 'M', 128, 128, 'AS128', 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
    'K': [64, 64, 'AS64', 'M', 128, 128, 'AS128', 'M', 256, 256, 256, 256, 'AC256', 'AS256', 'M',
          512, 512, 512, 512, 'AC512', 'AS512', 'M', 512, 512, 512, 512, 'AC512', 'AS512', 'M'],
    'L': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'AC256', 'AS256', 'M',
          512, 512, 512, 512, 'AC512', 'AS512', 'M', 512, 512, 512, 512, 'AC512', 'AS512', 'M'],
    'M': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'CR256', 'M',
          512, 512, 512, 512, 'CR512', 'M', 512, 512, 512, 512, 'CR512', 'M'],
    'N': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'AC512', 'AS512', 'M'],
    'O': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'SC512', 'M'],
    'P': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'SeqCS512', 'M'],

}


def hw_flattern(x):
    return x.view(x.size()[0], x.size()[1], -1)


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['J'], batch_norm=True), **kwargs)

    return model


class ChannelAttention(nn.Module):
    def __init__(self, c, reduction_ratio=8, gamma=1.0):  # 8
        super(ChannelAttention, self).__init__()
        self.conv1 = nn.Conv2d(c, c//int(reduction_ratio), kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(c, c//int(reduction_ratio), kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(c, c, kernel_size=1, stride=1, bias=False)

        # self.sqrt_dk = math.sqrt(c//int(reduction_ratio))
        self.gamma = gamma

    def forward(self, x):
        f = self.conv1(x)   # [bs,c',h,w]
        g = self.conv2(x)   # [bs,c',h,w]
        h = self.conv3(x)   # [bs,c,h,w]

        f = hw_flattern(f)
        f = torch.transpose(f, 1, 2)                # [bs,N,c']
        g = hw_flattern(g)                          # [bs,c',N]
        h = hw_flattern(h)                          # [bs,c,N]

        # s = torch.matmul(f, g) / self.sqrt_dk       # [bs,N,N]
        s = torch.matmul(f, g)
        beta = F.softmax(s, dim=1)

        o = torch.matmul(h, beta)                   # [bs,c, N]

        o = o.view(x.shape)
        x = self.gamma * o + x
        # x = F.sigmoid(o) * x

        return x


class SpatialAttention(nn.Module):
    def __init__(self, c, reduction_ratio=4, gamma=1.0):  # 4
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(c, c, kernel_size=2*reduction_ratio-1, stride=reduction_ratio, padding=reduction_ratio-1, bias=False)
        self.conv2 = nn.Conv2d(c, c, kernel_size=2*reduction_ratio-1, stride=reduction_ratio, padding=reduction_ratio-1, bias=False)
        self.conv3 = nn.Conv2d(c, c, kernel_size=1, stride=1, bias=False)

        self.gamma = gamma

    def forward(self, x):
        f = self.conv1(x)  # [bs,c,h',w']
        g = self.conv2(x)  # [bs,c,h',w']
        h = self.conv3(x)  # [bs,c,h,w]

        f = hw_flattern(f)              # [bs,c,N']
        g = hw_flattern(g)              # [bs,c,N']
        g = torch.transpose(g, 1, 2)    # [bs,N',c]

        s = torch.matmul(f, g)          # [bs,c,c]
        beta = F.softmax(s, dim=1)

        h = hw_flattern(h)              # [bs,c,N]
        h = torch.transpose(h, 1, 2)    # [bs,N,c]

        o = torch.matmul(h, beta)       # [bs,N,c]
        o = torch.transpose(o, 1, 2)    # [bs,c,N]

        o = o.view(x.shape)
        x = self.gamma * o + x
        # x = F.sigmoid(o) * x

        return x


class ChannelReduction(nn.Module):
    def __init__(self, c):
        super(ChannelReduction, self).__init__()
        self.conv1 = nn.Conv2d(c, 1, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(c, c, kernel_size=1, stride=1)
        self.trash = nn.Conv2d(c, c, kernel_size=1, stride=1)
        # self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        attn = self.conv1(x)
        attn = attn.view(x.size(0), -1)
        attn = F.softmax(attn, dim=1)
        attn = attn.view(x.size(0), 1, -1)
        attn = torch.transpose(attn, 1, 2).contiguous()

        x_flattern = x.view(x.size(0), x.size(1), -1)
        x2 = torch.matmul(x_flattern, attn)
        x2 = x2.unsqueeze(3)

        x3 = self.conv2(x2)
        x3 = F.sigmoid(x3)

        # x4 = x + x3
        # x = F.relu(self.bn(x4))
        x = x * x3

        return x


class ParallelSpatialChannelAttention(nn.Module):
    def __init__(self, c):
        super(ParallelSpatialChannelAttention, self).__init__()
        self.sa = SpatialAttention(c)
        self.ca = ChannelAttention(c)
        self.conv = nn.Conv2d(c, c, 1, 1)

    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(x)

        x = sa + ca
        x = self.conv(x)

        return x


class SequentialChannelSpatialAttention(nn.Module):
    def __init__(self, c):
        super(SequentialChannelSpatialAttention, self).__init__()
        self.gamma = 1.0

        self.reduction_c = 8
        self.conv1 = nn.Conv2d(c, c // int(self.reduction_c), kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(c, c // int(self.reduction_c), kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(c, c, kernel_size=1, stride=1)

        self.reduction_s = 4
        self.conv4 = nn.Conv2d(c, c, kernel_size=2 * self.reduction_s - 1, stride=self.reduction_s,
                               padding=self.reduction_s - 1)
        self.conv5 = nn.Conv2d(c, c, kernel_size=2 * self.reduction_s - 1, stride=self.reduction_s,
                               padding=self.reduction_s - 1)
        self.conv6 = nn.Conv2d(c, c, kernel_size=1, stride=1)

    def forward(self, x):
        x_ori = x
        f = self.conv1(x)  # [bs,c',h,w]
        g = self.conv2(x)  # [bs,c',h,w]
        h = self.conv3(x)  # [bs,c,h,w]

        f = hw_flattern(f)
        f = torch.transpose(f, 1, 2)  # [bs,N,c']
        g = hw_flattern(g)  # [bs,c',N]
        h = hw_flattern(h)  # [bs,c,N]

        s = torch.matmul(f, g)
        beta = F.softmax(s, dim=1)

        o = torch.matmul(h, beta)  # [bs,c, N]

        o = o.view(x.shape)

        x = o

        f = self.conv4(x)  # [bs,c,h',w']
        g = self.conv5(x)  # [bs,c,h',w']
        h = self.conv6(x)  # [bs,c,h,w]

        f = hw_flattern(f)  # [bs,c,N']
        g = hw_flattern(g)  # [bs,c,N']
        g = torch.transpose(g, 1, 2)  # [bs,N',c]

        s = torch.matmul(f, g)  # [bs,c,c]
        beta = F.softmax(s, dim=1)

        h = hw_flattern(h)  # [bs,c,N]
        h = torch.transpose(h, 1, 2)  # [bs,N,c]

        o = torch.matmul(h, beta)  # [bs,N,c]
        o = torch.transpose(o, 1, 2)  # [bs,c,N]

        o = o.view(x.shape)
        x = self.gamma * o + x_ori

        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif isinstance(v, str) and v.startswith('AC'):
            channel_num = int(v[2:])
            layers += [ChannelAttention(channel_num)]
        elif isinstance(v, str) and v.startswith('AS'):
            channel_num = int(v[2:])
            layers += [SpatialAttention(channel_num)]
        elif isinstance(v, str) and v.startswith('CR'):
            channel_num = int(v[2:])
            layers += [ChannelReduction(channel_num)]
        elif isinstance(v, str) and v.startswith('SC'):
            channel_num = int(v[2:])
            layers += [ParallelSpatialChannelAttention(channel_num)]
        elif isinstance(v, str) and v.startswith('SeqCS'):
            channel_num = int(v[5:])
            layers += [SequentialChannelSpatialAttention(channel_num)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, num_classes=2, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
