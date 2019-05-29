from collections import OrderedDict

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ncf, use_bias=False, mix=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ncf, ncf, kernel_size=3, stride=1, padding=1, bias=use_bias)),
            ('bn', nn.InstanceNorm2d(ncf)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ncf, ncf, kernel_size=3, stride=1, padding=1, bias=use_bias)),
            ('bn', nn.InstanceNorm2d(ncf)),
        ]))
        self.relu = nn.ReLU(inplace=True)

        if mix:
            self.mix_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ncf, int(ncf/2), kernel_size=3, stride=1, padding=1, bias=use_bias)),
            ('bn', nn.InstanceNorm2d(ncf)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.mix = mix

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x
        out = self.relu(out)
        if self.mix:
            return self.mix_conv(out)
        else:
            return out


class ResGenerator(nn.Module):
    def __init__(self, ngf, num_resblock):
        super(ResGenerator, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('pad', nn.ReflectionPad2d(3)),
            ('conv', nn.Conv2d(3, ngf, kernel_size=7, stride=1, padding=0, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf * 2)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.conv3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf * 4)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.p_conv1 = nn.Sequential(OrderedDict([
            ('pad', nn.ReflectionPad2d(3)),
            ('conv', nn.Conv2d(18 * 2, ngf, kernel_size=7, stride=1, padding=0, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.p_conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf * 2)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.p_conv3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf * 4)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.num_resblock = num_resblock
        for i in range(num_resblock):
            if i == 0:
                res = ResBlock(ngf * 8, use_bias=True, mix=True)
            else:
                res = ResBlock(ngf * 4, use_bias=True, mix=False)
            setattr(self, 'res' + str(i + 1), res)

        self.deconv3 = nn.Sequential(OrderedDict([
            ('deconv',
             nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf * 2)),
            ('relu', nn.ReLU(True))
        ]))
        self.deconv2 = nn.Sequential(OrderedDict([
            ('deconv',
             nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf)),
            ('relu', nn.ReLU(True))
        ]))
        self.deconv1 = nn.Sequential(OrderedDict([
            ('pad', nn.ReflectionPad2d(3)),
            ('conv', nn.Conv2d(ngf, 3, kernel_size=7, stride=1, padding=0, bias=False)),
            ('tanh', nn.Tanh())
        ]))

    def forward(self, im, pose, num_ouput=None):
        x = self.conv1(im)
        x = self.conv2(x)
        x = self.conv3(x)

        p = self.p_conv1(pose)
        p = self.p_conv2(p)
        p = self.p_conv3(p)

        x = torch.cat([x, p], dim=1)

        for i in range(self.num_resblock):
            res = getattr(self, 'res' + str(i + 1))
            x = res(x)
            if not self.training and (i == num_ouput):
                break
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        return x


class PatchDiscriminator(nn.Module):
    def __init__(self, ndf):
        super(PatchDiscriminator, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(6, ndf, kernel_size=4, stride=2, padding=1, bias=False)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf * 2)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.conv3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf * 4)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.conv4 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=0, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf * 8)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.dis = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)),
        ]))

    def forward(self, x, y):
        x = self.conv1(torch.cat([x, y], dim=1))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        dis = self.dis(x).squeeze()

        return dis