import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm



class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        chans = in_channels // reduction
        # I don't know if the Conv2d Layers are more efficient than the Linear layers for attention.
        # It does simplify the code, however.
        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Conv2d(in_channels=in_channels, out_channels=chans, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=chans, out_channels=in_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.layer(tensor)

class Identity(nn.Module):
    def __init__(self, a):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class NLayerDiscriminatorSN(nn.Module):
    """Defines a PatchGAN discriminator with spectral normalization. Also has 3x3 kernel."""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=Identity, reduction=16):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        '''
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        '''
        use_bias=True
        kw = 3
        padw = 1
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                        kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                SqueezeExcitation(in_channels=ndf * nf_mult, reduction=reduction)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                    kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            SqueezeExcitation(in_channels=ndf * nf_mult, reduction=reduction)
        ]

        # output 1 channel prediction map
        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]
        self.model = nn.Sequential(*sequence)

    def forward(self, input: torch.Tensor):
        """Standard forward."""
        return self.model(input)
