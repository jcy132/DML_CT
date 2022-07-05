import torch
from torch import nn, Tensor


def initialize_weights(net_l, scale=1.):
    if not isinstance(net_l, (list, tuple)):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)


class SqueezeExcitation(nn.Module):
    def __init__(self, num_features: int, reduction):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=num_features, out_channels=num_features // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_features // reduction, out_channels=num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * self.layers(inputs)


class ResidualDenseBlockCA(nn.Module):
    def __init__(self, num_features: int, growth_rate: int, reduction: int, num_layers: int,
                 padding_mode='zeros', use_sn=False):
        super().__init__()
        assert num_layers > 0, 'Number of convolution layers in the RDB must be a positive integer.'
        conv_args = dict(kernel_size=3, padding=1, padding_mode=padding_mode)
        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=growth_rate, **conv_args)
        # self.conv2 = nn.Conv2d(in_channels=in_channels + growth_rate, out_channels=growth_rate, **conv_args)
        # self.conv3 = nn.Conv2d(in_channels=in_channels + 2 * growth_rate, out_channels=growth_rate, **conv_args)
        # self.conv4 = nn.Conv2d(in_channels=in_channels + 3 * growth_rate, out_channels=growth_rate, **conv_args)
        # self.conv5 = nn.Conv2d(in_channels=in_channels + 4 * growth_rate, out_channels=in_channels, **conv_args)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.channel_attention = SqueezeExcitation(num_features, reduction)

        nets = list()
        for idx in range(num_layers):
            num_in = num_features + idx * growth_rate
            num_out = growth_rate if idx < (num_layers - 1) else num_features
            conv = nn.Conv2d(in_channels=num_in, out_channels=num_out, **conv_args)
            if use_sn:
                conv = nn.utils.spectral_norm(conv)
            nets.append(conv)

        # nets = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.channel_attention]
        initialize_weights(self.channel_attention, scale=0.1)
        initialize_weights(nets, scale=0.1)
        self.nets = nn.ModuleList(nets)

    def forward(self, inputs: Tensor):
        # out1 = self.lrelu(self.conv1(inputs))
        # out2 = self.lrelu(self.conv2(torch.cat([inputs, out1], dim=1)))  # All concatenation is on the channel axis.
        # out3 = self.lrelu(self.conv3(torch.cat([inputs, out1, out2], dim=1)))
        # out4 = self.lrelu(self.conv4(torch.cat([inputs, out1, out2, out3], dim=1)))
        # out5 = self.lrelu(self.conv4(torch.cat([inputs, out1, out2, out3, out4], dim=1)))
        # return self.channel_attention(out5) + inputs

        out = None  # Placeholder, keeps pylint happy.
        outs = [inputs]  # Initialize the DenseNet inputs.
        for net in self.nets:
            out = self.lrelu(net(torch.cat(outs, dim=1)))  # Concatenating all previous outputs.
            outs.append(out)  # Include output into output list.

        return inputs + self.channel_attention(out)


class RRDBCA(nn.Module):
    def __init__(self, num_features: int, growth_rate: int, reduction: int, num_blocks: int, num_layers: int,
                 padding_mode='zeros', use_sn=False):
        super().__init__()
        assert num_blocks > 0, 'Number of RDB blocks must be a positive integer.'
        layers = num_blocks * \
            [ResidualDenseBlockCA(num_features, growth_rate, reduction, num_layers, padding_mode, use_sn)]
        self.layers = nn.Sequential(*layers)
        self.channel_attention = SqueezeExcitation(num_features, reduction)
        initialize_weights(self.channel_attention, scale=0.1)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.channel_attention(self.layers(inputs))


class RRDBNetCA(nn.Module):
    def __init__(self, in_channels: int, num_features: int, growth_rate: int, reduction: int,
                 num_blocks: int, num_layers: int, padding_mode='zeros', use_sn=False):
        super().__init__()
        conv_kwargs = dict(kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv_first = nn.Conv2d(in_channels=in_channels, out_channels=num_features, **conv_kwargs)
        self.rrdb = RRDBCA(num_features, growth_rate, reduction, num_blocks, num_layers, padding_mode, use_sn)
        self.conv_mid = nn.Conv2d(in_channels=num_features, out_channels=num_features, **conv_kwargs)
        self.conv_last = nn.Conv2d(in_channels=num_features, out_channels=in_channels, **conv_kwargs)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        if use_sn:
            self.conv_first = nn.utils.spectral_norm(self.conv_first)
            self.conv_mid = nn.utils.spectral_norm(self.conv_mid)
            self.conv_last = nn.utils.spectral_norm(self.conv_last)

    def forward(self, inputs):
        features = self.conv_first(inputs)
        features += self.conv_mid(self.rrdb(features))
        return self.conv_last(self.lrelu(features))


# No channel attention here.
class ResidualDenseBlock(nn.Module):
    def __init__(self, num_features: int, growth_rate: int, num_layers: int, padding_mode='zeros', use_sn=False):
        super().__init__()
        assert num_layers > 0, 'Number of convolution layers in the RDB must be a positive integer.'
        conv_args = dict(kernel_size=3, padding=1, padding_mode=padding_mode)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)  # Negative slope fixed at 0.2.

        nets = list()
        for idx in range(num_layers):
            num_in = num_features + idx * growth_rate
            num_out = growth_rate if idx < (num_layers - 1) else num_features
            conv = nn.Conv2d(in_channels=num_in, out_channels=num_out, **conv_args)
            if use_sn:
                conv = nn.utils.spectral_norm(conv)
            nets.append(conv)

        initialize_weights(nets, scale=0.1)  # Weight scaling fixed at 0.1.
        self.nets = nn.ModuleList(nets)  # Necessary for registering parameters in model.

    def forward(self, inputs: Tensor):
        out = None  # Placeholder, keeps pylint happy.
        outs = [inputs]  # Initialize the DenseNet inputs.
        for net in self.nets:
            out = self.lrelu(net(torch.cat(outs, dim=1)))  # Concatenating all previous outputs in the channel axis.
            outs.append(out)  # Include output into output list.

        return inputs + 0.1 * out  # Residual scaling fixed at 0.1, different from ESRGAN.


class RRDB(nn.Module):
    def __init__(self, num_features: int, growth_rate: int, num_blocks: int,
                 num_layers: int, padding_mode='zeros', use_sn=False):
        super().__init__()
        assert num_blocks > 0, 'Number of RDB blocks must be a positive integer.'
        layers = num_blocks * [ResidualDenseBlock(num_features, growth_rate, num_layers, padding_mode, use_sn)]
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + 0.1 * self.layers(inputs)  # Residual scaling fixed at 0.1, different from ESRGAN.


class RRDBNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_features: int, growth_rate: int,
                 num_blocks: int, num_layers: int, padding_mode='zeros', use_sn=False, residual_bool=False):
        super().__init__()
        conv_kwargs = dict(kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv_first = nn.Conv2d(in_channels=in_channels, out_channels=num_features, **conv_kwargs)
        self.rrdb = RRDB(num_features, growth_rate, num_blocks, num_layers, padding_mode, use_sn)
        self.conv_mid = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=1, padding=0)
        self.conv_last = nn.Conv2d(in_channels=num_features, out_channels=out_channels, **conv_kwargs)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        if use_sn:
            self.conv_first = nn.utils.spectral_norm(self.conv_first)
            self.conv_mid = nn.utils.spectral_norm(self.conv_mid)
            self.conv_last = nn.utils.spectral_norm(self.conv_last)

        self.residual_bool = residual_bool

    def forward(self, inputs, layers=[], encode_only=False, residual_bool=False):
        if encode_only:
            feats = []

            features = self.conv_first(inputs)
            feat1 = self.rrdb(features)

            if residual_bool:
                feat2 = features + self.conv_mid(feat1)
            else:
                feat2 = self.conv_mid(feat1)

            if 1 in layers:
                feats.append(feat1)
            if 2 in layers:
                feats.append(feat2)
            return feats


        else:
            features = self.conv_first(inputs)
            if residual_bool:
                features += self.conv_mid(self.rrdb(features))
            else:
                features = self.conv_mid(self.rrdb(features))
            return self.conv_last(self.lrelu(features))



