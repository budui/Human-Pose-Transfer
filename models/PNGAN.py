from collections import OrderedDict

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ncf, use_bias=False):
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

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x
        out = self.relu(out)

        return out


class ResGenerator(nn.Module):
    def __init__(self, ngf, num_resblock):
        super(ResGenerator, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('pad', nn.ReflectionPad2d(3)),
            ('conv', nn.Conv2d(3 + 18, ngf, kernel_size=7, stride=1, padding=0, bias=True)),
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

        self.num_resblock = num_resblock
        for i in range(num_resblock):
            setattr(self, 'res' + str(i + 1), ResBlock(ngf * 4, use_bias=True))

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
        x = torch.cat((im, pose), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        for i in range(self.num_resblock):
            res = getattr(self, 'res' + str(i + 1))
            x = res(x)
            if not self.training and (i == num_ouput):
                break
            print(x.size())
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        return x


class DCDiscriminator(nn.Module):
    def __init__(self, ndf, num_att):
        super(DCDiscriminator, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False)),
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
            ('conv', nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf * 8)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.conv5 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf * 8)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.conv6 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf * 8)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.dis = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf * 8, 1, kernel_size=1, stride=1, padding=0, bias=False)),
        ]))
        self.att = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(7 * 3 * ndf * 8, 1024)),
            ('relu', nn.ReLU(True)),
            ('fc2', nn.Linear(1024, num_att))
        ]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        dis = self.dis(x)
        # print (x.size())
        x = x.view(x.size(0), -1)
        # print (x.size())
        att = self.att(x)

        return dis, att


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


class PABlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.res1 = ResBlock(num_channels)
        self.res2 = ResBlock(num_channels)
        self.conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(num_channels, num_channels*2, kernel_size=3, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(num_channels*2)),
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
            ('conv', nn.Conv2d(num_channels*2, num_channels, kernel_size=3, stride=1, padding=1, bias=use_bias)),
            ('bn', nn.InstanceNorm2d(num_channels*2)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.deconv = nn.Sequential(OrderedDict([
            ('deconv',
             nn.ConvTranspose2d(
                 (num_channels * 2) if not is_first else num_channels,
                 int(num_channels*0.5), kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(int(num_channels*0.5))),
            ('relu', nn.ReLU(True))
        ]))

    def forward(self, attr, img_f, pose_f, u=None):
        print(attr.size(), img_f.size())
        attr = attr.view(attr.size(0), -1, 1) # bs*27*1
        semantic_attr_t = self.line_emb(attr) # bs*27*nc
        img_f = img_f.view(img_f.size(0), img_f.size(1), -1) # bs*nc*(nh*nw)
        semantic_attr = torch.transpose(semantic_attr_t, 1, 2)

        tm = torch.matmul(semantic_attr_t, img_f)
        fattn = torch.matmul(semantic_attr, self.softmax(tm))
        fattn = fattn.view(pose_f.size())

        x = torch.cat([fattn, pose_f], dim=1)
        x = self.conv(x)
        if self.is_first:
            x = self.deconv(x)
        else:
            x = self.deconv(torch.cat([x, u], dim=1))
        return x


class PAGenerator(nn.Module):
    def __init__(self, pose_input_channels=18, num_block=3, ngf=64):
        super(PAGenerator, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('pad', nn.ReflectionPad2d(3)),
            ('conv', nn.Conv2d(3, ngf, kernel_size=7, stride=1, padding=0, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.pa1 = PABlock(ngf)
        self.pa2 = PABlock(ngf*2)
        self.pa3 = PABlock(ngf*4)

        self.conv_p = nn.Sequential(OrderedDict([
            ('pad', nn.ReflectionPad2d(3)),
            ('conv', nn.Conv2d(pose_input_channels, ngf, kernel_size=7, stride=1, padding=0, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.pa1_p = PABlock(ngf)
        self.pa2_p = PABlock(ngf * 2)
        self.pa3_p = PABlock(ngf * 4)

        self.au1 = AU(ngf*8, is_first=True)

        self.au2 = AU(ngf*4)
        self.au3 = AU(ngf*2)
        print(self.au3)

        self.deconv = nn.Sequential(OrderedDict([
            ('pad', nn.ReflectionPad2d(3)),
            ('conv', nn.Conv2d(ngf, 3, kernel_size=7, stride=1, padding=0, bias=False)),
            ('tanh', nn.Tanh())
        ]))

    def forward(self, img, pose, attr):
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


def _test():
    attr = torch.randn([2, 27])
    rg = ResGenerator(64, 9)
    x = torch.randn([2, 3, 128, 64])
    p = torch.randn([2, 18, 128, 64])
    y = rg(x, p)
    print(y.size())


if __name__ == '__main__':
    _test()