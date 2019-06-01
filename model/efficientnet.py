import torch
import torch.nn as nn
import torch.nn.functional as F

from .swish import Swish


class SqeezeExcitation(nn.Module):
    def __init__(self, inplanes, se_ratio):
        super(SqeezeExcitation, self).__init__()
        hidden_dim = int(inplanes*se_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(inplanes, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, inplanes, bias=False)
        self.swish = Swish()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.avg_pool(x).view(x.size(0), -1)
        out = self.fc1(out)
        out = self.swish(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2).unsqueeze(3)
        out = x * out.expand_as(x)
        return out


class Bottleneck(nn.Module):
    def __init__(self,inplanes, planes, kernel_size, stride, expand, se_ratio, prob=1.0):
        super(Bottleneck, self).__init__()
        if expand == 1:
            self.conv2 = nn.Conv2d(inplanes*expand, inplanes*expand, kernel_size=kernel_size, stride=stride,
                                   padding=kernel_size//2, groups=inplanes*expand, bias=False)
            self.bn2 = nn.BatchNorm2d(inplanes*expand, momentum=0.99, eps=1e-3)
            self.se = SqeezeExcitation(inplanes*expand, se_ratio)
            self.conv3 = nn.Conv2d(inplanes*expand, planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes, momentum=0.99, eps=1e-3)
        else:
            self.conv1 = nn.Conv2d(inplanes, inplanes*expand, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(inplanes*expand, momentum=0.99, eps=1e-3)
            self.conv2 = nn.Conv2d(inplanes*expand, inplanes*expand, kernel_size=kernel_size, stride=stride,
                                   padding=kernel_size//2, groups=inplanes*expand, bias=False)
            self.bn2 = nn.BatchNorm2d(inplanes*expand, momentum=0.99, eps=1e-3)
            self.se = SqeezeExcitation(inplanes*expand, se_ratio)
            self.conv3 = nn.Conv2d(inplanes*expand, planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes, momentum=0.99, eps=1e-3)

        self.swish = Swish()
        self.correct_dim = (stride == 1) and (inplanes == planes)
        self.prob = torch.Tensor([prob])

    def forward(self, x):
        if self.training:
            if not torch.bernoulli(self.prob):
                # drop
                return x

        if hasattr(self, 'conv1'):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.swish(out)
        else:
            out = x

        out = self.conv2(out) # depth wise conv
        out = self.bn2(out)
        out = self.swish(out)

        out = self.se(out)


        out = self.conv3(out)
        out = self.bn3(out)

        if self.correct_dim:
            out += x

        return out


class MBConv(nn.Module):
    def __init__(self, inplanes, planes, repeat, kernel_size, stride, expand, se_ratio, sum_layer, count_layer=None, pl=0.5):
        super(MBConv, self).__init__()
        layer = []

        # not drop(stchastic depth)
        layer.append(Bottleneck(inplanes, planes, kernel_size, stride, expand, se_ratio))

        for l in range(1, repeat):
            if count_layer is None:
                layer.append(Bottleneck(planes, planes, kernel_size, 1, expand, se_ratio))
            else:
                # stochastic depth
                prob = 1.0 - (count_layer + l) / sum_layer * (1 - pl)
                layer.append(Bottleneck(planes, planes, kernel_size, 1, expand, se_ratio, prob=prob))

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        out = self.layer(x)
        return out


class Upsample(nn.Module):
    def __init__(self, scale):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)


class Flatten(nn.Module):
    def __init(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)


class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000, width_coef=1., depth_coef=1., scale=1.,
                 dropout_ratio=0.2, se_ratio=0.25, stochastic_depth=False, pl=0.5):

        super(EfficientNet, self).__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        expands = [1, 6, 6, 6, 6, 6, 6]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        depth = depth_coef
        width = width_coef


        channels = [round(x*width) for x in channels] # [int(x*width) for x in channels]
        repeats = [round(x*depth) for x in repeats] # [int(x*width) for x in repeats]

        sum_layer = sum(repeats)

        self.upsample = Upsample(scale)
        self.swish = Swish()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0], momentum=0.99, eps=1e-3))

        if stochastic_depth:
            # stochastic depth
            self.stage2 = MBConv(channels[0], channels[1], repeats[0], kernel_size=kernel_sizes[0],
                                 stride=strides[0], expand=expands[0], se_ratio=se_ratio, sum_layer=sum_layer,
                                 count_layer=sum(repeats[:0]), pl=pl)
            self.stage3 = MBConv(channels[1], channels[2], repeats[1], kernel_size=kernel_sizes[1],
                                 stride=strides[1], expand=expands[1], se_ratio=se_ratio, sum_layer=sum_layer,
                                 count_layer=sum(repeats[:1]), pl=pl)
            self.stage4 = MBConv(channels[2], channels[3], repeats[2], kernel_size=kernel_sizes[2],
                                 stride=strides[2], expand=expands[2], se_ratio=se_ratio, sum_layer=sum_layer,
                                 count_layer=sum(repeats[:2]), pl=pl)
            self.stage5 = MBConv(channels[3], channels[4], repeats[3], kernel_size=kernel_sizes[3],
                                 stride=strides[3], expand=expands[3], se_ratio=se_ratio, sum_layer=sum_layer,
                                 count_layer=sum(repeats[:3]), pl=pl)
            self.stage6 = MBConv(channels[4], channels[5], repeats[4], kernel_size=kernel_sizes[4],
                                 stride=strides[4], expand=expands[4], se_ratio=se_ratio, sum_layer=sum_layer,
                                 count_layer=sum(repeats[:4]), pl=pl)
            self.stage7 = MBConv(channels[5], channels[6], repeats[5], kernel_size=kernel_sizes[5],
                                 stride=strides[5], expand=expands[5], se_ratio=se_ratio, sum_layer=sum_layer,
                                 count_layer=sum(repeats[:5]), pl=pl)
            self.stage8 = MBConv(channels[6], channels[7], repeats[6], kernel_size=kernel_sizes[6],
                                 stride=strides[6], expand=expands[6], se_ratio=se_ratio, sum_layer=sum_layer,
                                 count_layer=sum(repeats[:6]), pl=pl)
        else:
            self.stage2 = MBConv(channels[0], channels[1], repeats[0], kernel_size=kernel_sizes[0],
                                 stride=strides[0], expand=expands[0], se_ratio=se_ratio, sum_layer=sum_layer)
            self.stage3 = MBConv(channels[1], channels[2], repeats[1], kernel_size=kernel_sizes[1],
                                 stride=strides[1], expand=expands[1], se_ratio=se_ratio, sum_layer=sum_layer)
            self.stage4 = MBConv(channels[2], channels[3], repeats[2], kernel_size=kernel_sizes[2],
                                 stride=strides[2], expand=expands[2], se_ratio=se_ratio, sum_layer=sum_layer)
            self.stage5 = MBConv(channels[3], channels[4], repeats[3], kernel_size=kernel_sizes[3],
                                 stride=strides[3], expand=expands[3], se_ratio=se_ratio, sum_layer=sum_layer)
            self.stage6 = MBConv(channels[4], channels[5], repeats[4], kernel_size=kernel_sizes[4],
                                 stride=strides[4], expand=expands[4], se_ratio=se_ratio, sum_layer=sum_layer)
            self.stage7 = MBConv(channels[5], channels[6], repeats[5], kernel_size=kernel_sizes[5],
                                 stride=strides[5], expand=expands[5], se_ratio=se_ratio, sum_layer=sum_layer)
            self.stage8 = MBConv(channels[6], channels[7], repeats[6], kernel_size=kernel_sizes[6],
                                 stride=strides[6], expand=expands[6], se_ratio=se_ratio, sum_layer=sum_layer)

        self.stage9 = nn.Sequential(
                            nn.Conv2d(channels[7], channels[8], kernel_size=1, bias=False),
                            nn.BatchNorm2d(channels[8], momentum=0.99, eps=1e-3),
                            Swish(),
                            nn.AdaptiveAvgPool2d((1, 1)),
                            Flatten(),
                            nn.Dropout(p=dropout_ratio),
                            nn.Linear(channels[8], num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='sigmoid')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.upsample(x)
        x = self.swish(self.stage1(x))
        x = self.swish(self.stage2(x))
        x = self.swish(self.stage3(x))
        x = self.swish(self.stage4(x))
        x = self.swish(self.stage5(x))
        x = self.swish(self.stage6(x))
        x = self.swish(self.stage7(x))
        x = self.swish(self.stage8(x))
        logit = self.stage9(x)

        return logit



def efficientnet_b0(num_classes=1000):
    return EfficientNet(num_classes=num_classes, width_coef=1.0, depth_coef=1.0, scale=1.0,dropout_ratio=0.2, se_ratio=0.25)

def efficientnet_b1(num_classes=1000):
    return EfficientNet(num_classes=num_classes, width_coef=1.0, depth_coef=1.1, scale=240/224, dropout_ratio=0.2, se_ratio=0.25)

def efficientnet_b2(num_classes=1000):
    return EfficientNet(num_classes=num_classes, width_coef=1.1, depth_coef=1.2, scale=260/224., dropout_ratio=0.3, se_ratio=0.25)

def efficientnet_b3(num_classes=1000):
    return EfficientNet(num_classes=num_classes, width_coef=1.2, depth_coef=1.4, scale=300/224, dropout_ratio=0.3, se_ratio=0.25)

def efficientnet_b4(num_classes=1000):
    return EfficientNet(num_classes=num_classes, width_coef=1.4, depth_coef=1.8, scale=380/224, dropout_ratio=0.4, se_ratio=0.25)

def efficientnet_b5(num_classes=1000):
    return EfficientNet(num_classes=num_classes, width_coef=1.6, depth_coef=2.2, scale=456/224, dropout_ratio=0.4, se_ratio=0.25)

def efficientnet_b6(num_classes=1000):
    return EfficientNet(num_classes=num_classes, width_coef=1.8, depth_coef=2.6, scale=528/224, dropout_ratio=0.5, se_ratio=0.25)

def efficientnet_b7(num_classes=1000):
    return EfficientNet(num_classes=num_classes, width_coef=2.0, depth_coef=3.1, scale=600/224, dropout_ratio=0.5, se_ratio=0.25)

def test():
    x = torch.FloatTensor(64, 3, 224, 224)
    model = EfficientNet(num_classes=100, width_coef=1.0, depth_coef=1.0, scale=1.0,dropout_ratio=0.2,
                         se_ratio=0.25, stochastic_depth=True)
    logit = model(x)
    print(logit.size())

if __name__ == '__main__':
    test()
