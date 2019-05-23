import torch
import torch.nn as nn
import torch.nn.functional

NUM_KEY_POINT = 18
NUM_POSE_DIM = NUM_KEY_POINT * 2 + NUM_KEY_POINT
NUM_MIDDLE_CHANNELS = 128
NUM_ROI_FEATURE_DIM = 32
NUM_BG_FEATURE_DIM = 128
NUM_FC_MIDDLE_Z_DIM = 512


class _ResLinear(nn.Module):
    def __init__(self, num_features=512):
        super(_ResLinear, self).__init__()
        self.linear1 = nn.Linear(num_features, num_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_features, num_features)

    def forward(self, x):
        res = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return res + x


class _ResBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(_ResBasicBlock, self).__init__()

        self.block_2conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding),
            nn.ReLU()
        )

        self.block_1conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        res = x
        x = self.block_2conv(x)
        x = x + res

        output = self.block_1conv(x)

        return output


class _ResAdvBlock(nn.Module):
    def __init__(self, is_last_block, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, in_decoder=False):
        super(_ResAdvBlock, self).__init__()
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


class AppearanceExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=NUM_MIDDLE_CHANNELS, kernel_size=3, stride=1, padding=1):
        super(AppearanceExtractor, self).__init__()
        self.main_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.main_block(x)
        return x


class PoseDecoder(nn.Module):
    def __init__(self, num_input=32, num_features=NUM_FC_MIDDLE_Z_DIM, num_linear=4):
        super(PoseDecoder, self).__init__()
        self.input_linear = nn.Linear(num_input, num_features)
        self.fc_res = nn.ModuleList([_ResLinear(num_features) for _ in range(num_linear)])
        self.output_linear = nn.Linear(num_features, NUM_POSE_DIM)

    def forward(self, x):
        x = self.input_linear(x)
        for l in self.fc_res:
            x = l(x)
        x = self.output_linear(x)
        return x.view(-1, NUM_KEY_POINT, 3)


class PoseEncoder(nn.Module):
    def __init__(self, num_output=32, num_features=NUM_FC_MIDDLE_Z_DIM, num_linear=4):
        super(PoseEncoder, self).__init__()
        self.num_linear = num_linear

        self.input_linear = nn.Linear(NUM_POSE_DIM, num_features)
        self.fc_res = nn.ModuleList([_ResLinear(num_features) for _ in range(num_linear)])
        self.output_linear = nn.Linear(num_features, num_output)

    def forward(self, key_points):
        key_points = key_points.view(-1, NUM_POSE_DIM)
        x = self.input_linear(key_points)
        for l in self.fc_res:
            x = l(x)
        x = self.output_linear(x)
        return x


class BGEncoder(nn.Module):
    def __init__(self, num_res, output_dim=NUM_BG_FEATURE_DIM):
        super(BGEncoder, self).__init__()
        self.num_res = num_res
        self.first_conv = nn.Sequential(
            nn.Conv2d(NUM_MIDDLE_CHANNELS, NUM_MIDDLE_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.ResConvs = nn.ModuleList([
            _ResBasicBlock(NUM_MIDDLE_CHANNELS * i, NUM_MIDDLE_CHANNELS * (i + 1))
            for i in range(1, num_res)
        ])
        self.fc = nn.Linear(num_res * NUM_MIDDLE_CHANNELS * 8 * 4, output_dim)

    def forward(self, x):
        x = self.first_conv(x)
        for res_c in self.ResConvs:
            x = res_c(x)
        x = x.view(-1, self.num_res * NUM_MIDDLE_CHANNELS * 8 * 4)
        x = self.fc(x)
        return x


class FGEncoder(nn.Module):
    def __init__(self, num_res, output_dim=NUM_ROI_FEATURE_DIM):
        super(FGEncoder, self).__init__()
        self.num_res = num_res
        self.first_conv = nn.Sequential(
            nn.Conv2d(NUM_MIDDLE_CHANNELS, NUM_MIDDLE_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.ResConvs = nn.ModuleList([
            _ResBasicBlock(NUM_MIDDLE_CHANNELS * i, NUM_MIDDLE_CHANNELS * (i + 1))
            for i in range(1, num_res)
        ])
        self.fc = nn.Linear(num_res * NUM_MIDDLE_CHANNELS * 3 * 3, output_dim)

    def forward(self, x):
        x = self.first_conv(x)
        for res_c in self.ResConvs:
            x = res_c(x)
        x = x.view(-1, self.num_res * NUM_MIDDLE_CHANNELS * 3 * 3)
        x = self.fc(x)
        return x


class AppearanceReconstructor(nn.Module):
    def __init__(self, in_channels=7 * NUM_ROI_FEATURE_DIM + NUM_BG_FEATURE_DIM,
                 num_res=6, middle_z_dim=64, half_width=True):
        super(AppearanceReconstructor, self).__init__()
        self.hidden_num = NUM_MIDDLE_CHANNELS
        self.repeat_num = num_res
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
                    _ResAdvBlock(i == self.repeat_num - 1, self.hidden_num * (i + 1), self.hidden_num * (i + 2)))
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
                    _ResAdvBlock(i == self.repeat_num - 1,
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


class MappingFunc(nn.Module):
    def __init__(self, k_dim, middle_dim=NUM_FC_MIDDLE_Z_DIM, num_middle_fc=4):
        super(MappingFunc, self).__init__()
        self.in_linear = nn.Linear(k_dim, middle_dim)
        self.middle_linears = nn.ModuleList([
            _ResLinear(middle_dim)
            for _ in range(num_middle_fc)
        ])
        self.out_linear = nn.Linear(middle_dim, k_dim)

    def forward(self, z):
        x = self.in_linear(z)
        for l in self.middle_linears:
            x = l(x)
        x = self.out_linear(x)
        return x


def _test():
    rbb = AppearanceExtractor()
    input_img = torch.ones([2, 3, 128, 64])
    output_img = rbb(input_img)
    print("AppearanceExtractor output size:{}".format(output_img.size()))

    bge = BGEncoder(5)
    print("BGEncoder output size:{}".format(bge(output_img).size()))

    fge = FGEncoder(5)
    roi = torch.ones([2, 128, 48, 48])
    print("FGEncoder output size:{}".format(fge(roi).size()))

    c = torch.randn([3, 18, 2])
    v = torch.randn([3, 18])
    kpe = PoseEncoder()
    kpd = PoseDecoder()
    kpe_o = kpe(torch.cat([c, v.view(-1, 18, 1)], dim=-1))
    print("PoseEncoder output size:{}".format(kpe_o.size()))
    print("PoseDecoder output size:{}".format(kpd(kpe_o).size()))


if __name__ == '__main__':
    _test()
