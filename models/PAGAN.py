from collections import OrderedDict

import torch
import torch.nn as nn

from models.PNGAN import ResBlock


class PABlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.res1 = ResBlock(num_channels)
        self.res2 = ResBlock(num_channels)
        self.conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(num_channels * 2)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.conv(x)
        return x


class AU(nn.Module):
    def __init__(self, num_channels, softmax_dim=0, is_first=False, use_bias=False):
        super().__init__()
        self.is_first = is_first
        self.line_emb = nn.Linear(1, num_channels, bias=False)
        self.softmax = nn.Softmax(dim=softmax_dim)
        self.conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(num_channels * 2, num_channels, kernel_size=3, stride=1, padding=1, bias=use_bias)),
            ('bn', nn.InstanceNorm2d(num_channels * 2)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.deconv = nn.Sequential(OrderedDict([
            ('deconv',
             nn.ConvTranspose2d(
                 (num_channels * 2) if not is_first else num_channels,
                 int(num_channels * 0.5), kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(int(num_channels * 0.5))),
            ('relu', nn.ReLU(True))
        ]))

    def forward(self, attr, img_f, pose_f, u=None):
        # attr = attr.view(attr.size(0), -1, 1)  # bs*27*1
        # semantic_attr_t = self.line_emb(attr)  # bs*27*nc
        # img_f = img_f.view(img_f.size(0), img_f.size(1), -1)  # bs*nc*(nh*nw)
        # semantic_attr = torch.transpose(semantic_attr_t, 1, 2)
        #
        # tm = torch.matmul(semantic_attr_t, img_f)
        # fattn = torch.matmul(semantic_attr, self.softmax(tm))
        # fattn = fattn.view(pose_f.size())

        x = torch.cat([img_f, pose_f], dim=1)
        x = self.conv(x)
        if self.is_first:
            x = self.deconv(x)
        else:
            x = self.deconv(torch.cat([x, u], dim=1))
        return x


class PAGenerator(nn.Module):
    def __init__(self, pose_input_channels=18*2, num_block=3, ngf=64):
        super(PAGenerator, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('pad', nn.ReflectionPad2d(3)),
            ('conv', nn.Conv2d(3, ngf, kernel_size=7, stride=1, padding=0, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.pa1 = PABlock(ngf)
        self.pa2 = PABlock(ngf * 2)
        self.pa3 = PABlock(ngf * 4)

        self.conv_p = nn.Sequential(OrderedDict([
            ('pad', nn.ReflectionPad2d(3)),
            ('conv', nn.Conv2d(pose_input_channels, ngf, kernel_size=7, stride=1, padding=0, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.pa1_p = PABlock(ngf)
        self.pa2_p = PABlock(ngf * 2)
        self.pa3_p = PABlock(ngf * 4)

        self.au1 = AU(ngf * 8, is_first=True)

        self.au2 = AU(ngf * 4)
        self.au3 = AU(ngf * 2)
        print(self.au3)

        self.deconv = nn.Sequential(OrderedDict([
            ('pad', nn.ReflectionPad2d(3)),
            ('conv', nn.Conv2d(ngf, 3, kernel_size=7, stride=1, padding=0, bias=False)),
            ('tanh', nn.Tanh())
        ]))

    def forward(self, img, pose):
        attr = None
        v1 = self.pa1(self.conv(img))
        v2 = self.pa2(v1)
        v3 = self.pa3(v2)

        s1 = self.pa1_p(self.conv_p(pose))
        s2 = self.pa2_p(s1)
        s3 = self.pa3_p(s2)

        u1 = self.au1(attr, v3, s3)
        u2 = self.au2(attr, v2, s2, u1)
        u3 = self.au3(attr, v1, s1, u2)

        return self.deconv(u3)
