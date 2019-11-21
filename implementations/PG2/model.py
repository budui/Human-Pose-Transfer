import torch
import torch.nn as nn
import torch.nn.functional as functional

from helper import weights_init


class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.main(x) + x


class _G1EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last_block=True):
        """
        :param is_last_block: last block in encoder DO NOT down sampling.
        """
        super().__init__()
        self.res = ResBlock(in_channels)

        self.is_last_block = is_last_block
        if not self.is_last_block:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.res(x)
        return x, self.conv(x) if not self.is_last_block else x


class _G1DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last_block=True):
        """
        :param is_last_block: last block in encoder DO NOT Downsampling.
        """
        super().__init__()
        self.res = ResBlock(in_channels)

        self.is_last_block = is_last_block
        if not self.is_last_block:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.ReLU()
            )

    def forward(self, x, skip_connection):
        x = torch.cat([x, skip_connection], dim=1)
        x = self.res(x)
        if not self.is_last_block:
            x = functional.interpolate(x, scale_factor=2)
            return self.conv(x)
        else:
            return x


class Generator1(nn.Module):
    def __init__(self, in_channels, num_repeat, middle_features_dim, channels_base, image_size=(128, 64)):
        super().__init__()
        self.num_repeat = num_repeat
        self.channels_base = channels_base

        self.start_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels_base, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.encoder = nn.ModuleList([
            _G1EncoderBlock(channels_base * (idx + 1), channels_base * (idx + 2), idx == num_repeat - 1)
            for idx in range(num_repeat)
        ])

        self.encoder_output_size = self.cal_encoder_output_size(image_size, num_repeat)
        middle_space_size = self.encoder_output_size[0] * self.encoder_output_size[1]
        self.down_fc = nn.Linear(middle_space_size * num_repeat * channels_base, middle_features_dim)
        self.up_fc = nn.Linear(middle_features_dim, middle_space_size * channels_base)

        skip_connection_channels = [(num_repeat - i) * channels_base for i in range(num_repeat)]
        decoder_blocks_channels = [(num_repeat - i) * channels_base for i in range(num_repeat)]
        decoder_blocks_channels[0] = channels_base
        in_channels_list = list(map(sum, zip(skip_connection_channels, decoder_blocks_channels)))

        self.decoder = nn.ModuleList([
            _G1DecoderBlock(in_channels_list[idx], (num_repeat - idx - 1) * channels_base, idx == num_repeat - 1)
            for idx in range(num_repeat)
        ])

        self.end_conv = nn.Conv2d(in_channels_list[-1], 3, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def cal_encoder_output_size(image_size, num_repeat):
        """
        calculate the height*width of encoder output features.
        every decoder block (except last block): output_tensor_height = input_tensor_height / 2
        this is only correct when image size can be divided by the power of 2,
        like 128(Market1501), 256(DeepFashion)
        """
        return int(image_size[0] / (2 ** (num_repeat - 1))), int(image_size[1] / (2 ** (num_repeat - 1)))

    def forward(self, condition_img, target_pose):
        x = torch.cat([condition_img, target_pose], dim=1)
        x = self.start_conv(x)

        skip_connection_list = []
        for i, block in enumerate(self.encoder):
            skip_conn_feat, x = block(x)
            skip_connection_list.append(skip_conn_feat)

        x = x.view(x.size(0), -1)
        x = self.down_fc(x)
        z = x
        x = self.up_fc(z)
        x = x.view(x.size(0), self.channels_base, self.encoder_output_size[0], -1)

        for i, block in enumerate(self.decoder):
            x = block(x, skip_connection_list[- 1 - i])

        return self.end_conv(x)


class _G2EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last_block=True):
        """
        :param is_last_block: last block in encoder DO NOT down sampling.
        """
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU()
        )
        self.is_last_block = is_last_block
        if not self.is_last_block:
            self.conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 2, 1),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.main(x)
        return x, self.conv(x) if not self.is_last_block else x


class _G2DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last_block=True):
        """
        :param is_last_block: last block in encoder DO NOT Downsampling.
        """
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU()
        )

        self.is_last_block = is_last_block

    def forward(self, x):
        x = self.main(x)
        if not self.is_last_block:
            x = functional.interpolate(x, scale_factor=2)
        return x


class Generator2(nn.Module):
    def __init__(self, in_channels, channels_base, num_repeat, num_skip_out_connect=0, weight_init_way=None):
        super().__init__()
        self.start_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels_base, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.num_repeat = num_repeat
        self.num_skip_out_connect = num_skip_out_connect
        self.encoder = nn.ModuleList([
            _G2EncoderBlock(
                channels_base * idx if idx > 0 else channels_base,
                channels_base * (idx + 1),
                idx == num_repeat - 1
            )
            for idx in range(num_repeat)
        ])

        skip_connection_channels = [channels_base * num_repeat] + \
                                   [channels_base] * (num_repeat - 1 - num_skip_out_connect) + \
                                   [0] * num_skip_out_connect
        decoder_blocks_channels = [(num_repeat - i) * channels_base for i in range(num_repeat)]
        in_channels_list = list(map(sum, zip(skip_connection_channels, decoder_blocks_channels)))

        self.decoder = nn.ModuleList([
            _G2DecoderBlock(
                in_channels_list[idx],
                channels_base,
                idx == num_repeat - 1
            )
            for idx in range(num_repeat)
        ])

        self.end_conv = nn.Conv2d(channels_base, 3, kernel_size=3, stride=1, padding=1)

        if weight_init_way is not None:
            self.apply(weights_init.select(weight_init_way))

    def forward(self, condition_image, stage1_image):
        x = torch.cat([condition_image, stage1_image], dim=1)
        x = self.start_conv(x)

        skip_connection_list = []
        for i, block in enumerate(self.encoder):
            skip_conn_feat, x = block(x)
            if i > self.num_skip_out_connect - 1:
                skip_connection_list.append(skip_conn_feat)

        for i, block in enumerate(self.decoder):
            if i < self.num_repeat - self.num_skip_out_connect:
                block_input = torch.cat([x, skip_connection_list[-1 - i]], dim=1)
            else:
                block_input = x
            x = block(block_input)

        return self.end_conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, channel_base=64, market_or_DF=True, weight_init_way=None):
        super(Discriminator, self).__init__()
        self.base_channels = channel_base
        kernel_size = 5
        # TODO: Ma use "SAME"-padding in tf code
        padding = 2
        stride = 2
        layers = [
            nn.Conv2d(in_channels, channel_base, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channel_base, channel_base * 2, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(channel_base * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channel_base * 2, channel_base * 4, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(channel_base * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channel_base * 4, channel_base * 8, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(channel_base * 8),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if not market_or_DF:
            layers += [
                nn.Conv2d(channel_base * 8, channel_base * 8, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(channel_base * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.main = nn.Sequential(*layers)
        self.fc = nn.Linear(8 * 8 * channel_base * (4 if market_or_DF else 8), 1)

        if weight_init_way is not None:
            self.apply(weights_init.select(weight_init_way))

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
