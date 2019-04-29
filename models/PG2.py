import torch
import torch.nn as nn
import torch.nn.functional


class G1BasicBlock(nn.Module):
    def __init__(self, is_last_block, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, in_decoder=False):
        super(G1BasicBlock, self).__init__()
        self.is_last_block = is_last_block
        self.in_decoder = in_decoder
        self.block_2conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding),
            nn.ReLU()
        )
        if not self.is_last_block:
            if self.in_decoder:
                self.block_1conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                    nn.ReLU()
                )
            else:
                self.block_1conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                    nn.ReLU()
                )

    def forward(self, x):
        res = x
        x = self.block_2conv(x)
        x = x + res
        if self.is_last_block:
            return x
        else:
            if self.in_decoder:
                x = nn.functional.interpolate(x, scale_factor=2)
                output = self.block_1conv(x)
                return output
            else:
                output = self.block_1conv(x)
                return output, x


class G1(nn.Module):
    def __init__(self, in_channels, hidden_num=128, repeat_num=6, middle_z_dim=128, half_width=False):
        super(G1, self).__init__()
        self.hidden_num = hidden_num
        self.repeat_num = repeat_num
        self.middle_z_dim = middle_z_dim
        self.half_width = half_width

        self.min_feat_map_h = 8
        self.min_feat_map_w = int(self.min_feat_map_h / 2) if self.half_width else self.min_feat_map_h

        self.middle_feature_dim = self.min_feat_map_h * self.min_feat_map_w * self.repeat_num * self.hidden_num

        self.first_block = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.encoder_blocks = []
        for i in range(self.repeat_num):
            setattr(self, "en_block_{}".format(i),
                    G1BasicBlock(i == self.repeat_num - 1, self.hidden_num * (i + 1), self.hidden_num * (i + 2)))
            self.encoder_blocks.append(getattr(self, "en_block_{}".format(i)))

        self.fc1 = nn.Linear(self.middle_feature_dim, self.middle_z_dim)

        self.fc2 = nn.Linear(self.middle_z_dim, self.min_feat_map_h * self.min_feat_map_w * self.hidden_num)

        skip_connection_channels = [(self.repeat_num - i) * self.hidden_num for i in range(self.repeat_num)]
        decoder_blocks_channels = [(self.repeat_num - i) * self.hidden_num for i in range(self.repeat_num)]
        decoder_blocks_channels[0] = self.hidden_num
        in_channels_list = list(map(sum, zip(skip_connection_channels, decoder_blocks_channels)))

        self.decoder_blocks = []
        for i in range(self.repeat_num):
            setattr(self, "de_block_{}".format(i),
                    G1BasicBlock(i == self.repeat_num - 1,
                                 in_channels=in_channels_list[i],
                                 out_channels=self.hidden_num * (self.repeat_num - i - 1),
                                 in_decoder=True))
            self.decoder_blocks.append(getattr(self, "de_block_{}".format(i)))

        self.last_conv = nn.Conv2d(in_channels_list[-1], 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.first_block(x)
        skip_connection_list = []
        for i, block in enumerate(self.encoder_blocks):
            if i != self.repeat_num - 1:
                # not last res block
                x, skip_conn_feat = block(x)
                skip_connection_list.append(skip_conn_feat)
            else:
                x = block(x)
                skip_connection_list.append(x)

        x = x.view(-1, self.middle_feature_dim)
        x = self.fc1(x)
        z = x
        x = self.fc2(z)
        x = x.view(-1, self.hidden_num, self.min_feat_map_h, self.min_feat_map_w)

        for i, block in enumerate(self.decoder_blocks):
            block_input = torch.cat([x, skip_connection_list[- 1 - i]], dim=1)
            x = block(block_input)
        output = self.last_conv(x)
        return output


class G2BasicBlock(nn.Module):
    def __init__(self, is_last_block, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, in_decoder=False):
        super(G2BasicBlock, self).__init__()
        self.is_last_block = is_last_block
        self.in_decoder = in_decoder
        self.block_2conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU()
        )
        if not self.is_last_block and not self.in_decoder:
            self.block_1conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size, 2, padding),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.block_2conv(x)
        if self.is_last_block:
            return x
        else:
            if self.in_decoder:
                x = nn.functional.interpolate(x, scale_factor=2)
                return x
            else:
                output = self.block_1conv(x)
                return output, x


class G2(nn.Module):
    def __init__(self, in_channels, hidden_num=128, repeat_num=4, skip_connect=0):
        super(G2, self).__init__()
        self.hidden_num = hidden_num
        self.repeat_num = repeat_num
        self.skip_connect = skip_connect

        self.first_block = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.encoder_blocks = []
        for i in range(self.repeat_num):
            block_in_channels = self.hidden_num * i if i > 0 else self.hidden_num
            setattr(self, "en_block_{}".format(i),
                    G2BasicBlock(i == self.repeat_num - 1, block_in_channels, self.hidden_num * (i + 1)))
            self.encoder_blocks.append(getattr(self, "en_block_{}".format(i)))

        skip_connection_channels = [self.hidden_num * self.repeat_num] + \
                                   [self.hidden_num] * (self.repeat_num - 1 - self.skip_connect) + \
                                   [0] * self.skip_connect
        decoder_blocks_channels = [(self.repeat_num - i) * self.hidden_num for i in range(self.repeat_num)]
        in_channels_list = list(map(sum, zip(skip_connection_channels, decoder_blocks_channels)))

        self.decoder_blocks = []
        for i in range(self.repeat_num):
            setattr(self, "de_block_{}".format(i),
                    G2BasicBlock(i == self.repeat_num - 1,
                                 in_channels=in_channels_list[i],
                                 out_channels=self.hidden_num,
                                 in_decoder=True))
            self.decoder_blocks.append(getattr(self, "de_block_{}".format(i)))

        self.last_conv = nn.Conv2d(self.hidden_num, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.first_block(x)
        skip_connection_list = []
        for i, block in enumerate(self.encoder_blocks):
            if i != self.repeat_num - 1:
                # not last res block
                x, skip_conn_feat = block(x)
                if i > self.skip_connect - 1:
                    skip_connection_list.append(skip_conn_feat)
            else:
                x = block(x)
                skip_connection_list.append(x)

        for i, block in enumerate(self.decoder_blocks):
            if i < self.repeat_num - self.skip_connect:
                block_input = torch.cat([x, skip_connection_list[-1 - i]], dim=1)
            else:
                block_input = x
            x = block(block_input)
        output = self.last_conv(x)
        return output


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# TODO 这部分结构不知道怎么设计，目前靠调参数使得最后输出为1，但是作者的代码是加入FC的，多试试。
class Discriminator(nn.Module):
    def __init__(self, in_channels=6, base_channels=64):
        super(Discriminator, self).__init__()
        self.base_channels = base_channels
        kernel_size = 5
        padding = 2
        stride = 2
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(8*4*8*base_channels, 1)
        self.sigmoid = nn.Sigmoid()

        self.main.apply(weights_init)

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 8*4*8*self.base_channels)
        return self.sigmoid(self.fc(x))


class NDiscriminator(nn.Module):
    def __init__(self, in_channels=6, base_channels=64):
        super(NDiscriminator, self).__init__()
        nf = base_channels
        nc = in_channels
        self.main = nn.Sequential(
            # input is (nc) x 128 x 64
            nn.Conv2d(in_channels=nc, out_channels=nf, kernel_size=4, stride=(4, 2), padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (nf) x 32 x 32
            nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (nf*2) x 16 x 16
            nn.Conv2d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (nf*4) x 8 x 8
            nn.Conv2d(in_channels=nf * 4, out_channels=nf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (nf*8) x 4 x 4
            nn.Conv2d(in_channels=nf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.main.apply(weights_init)

    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1)


if __name__ == '__main__':
    g2 = G2(3 + 3, hidden_num=64, repeat_num=3, skip_connect=1)
    print(g2)


