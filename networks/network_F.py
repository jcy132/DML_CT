import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np


##############
# Network
##############


class PatchSampleF(nn.Module):
    def __init__(self, input_nc, nc=256, use_mlp=False, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.gpu_ids = gpu_ids
        self.mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])

    def forward(self, x, num_patches=64, patch_ids=None):

        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        feat_reshape = x.permute(0, 2, 3, 1).flatten(1, 2)
        if num_patches > 0:
            if patch_ids is not None:
                patch_id = patch_ids
            else:
                patch_id = torch.randperm(feat_reshape.shape[1], device=x[0].device)
                patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
            x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
        else:
            x_sample = feat_reshape
            patch_id = []

        if self.use_mlp:
            x_sample = self.mlp(x_sample)

        x_sample = self.l2norm(x_sample)

        if num_patches == 0:
            x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])

        return x_sample, patch_id


class ConvF(nn.Module):
    def __init__(self, input_nc, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(ConvF, self).__init__()
        self.l2norm = Normalize(2)
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.gpu_ids = gpu_ids
        self.conv = nn.Sequential(*[nn.Conv2d(input_nc, self.nc, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(self.nc, self.nc, 4, 2, 1)])

    def forward(self, x, num_patches=64, patch_ids=None):

        feat = self.conv(x)
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1,2)        #(B, HxW, C)

        if num_patches > 0:
            if patch_ids is not None:
                patch_id = patch_ids
            else:
                patch_id = torch.randperm(feat_reshape.shape[1], device=x[0].device)
                patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
            feat_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])         #(BXHXW, C)
        else:
            feat_sample = feat_reshape
            patch_id = []

        feat_sample = self.l2norm(feat_sample)

        return feat_sample, patch_id


class PyramidalF(nn.Module):
    def __init__(self, input_nc, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PyramidalF, self).__init__()
        self.l2norm = Normalize(2)
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.gpu_ids = gpu_ids
        self.layer1 = nn.Sequential(*[nn.Conv2d(input_nc, self.nc, 1, 1, 0), nn.ReLU()])
        self.prj1 = nn.Sequential(*[nn.Conv2d(self.nc, self.nc, 1, 1, 0)])
        self.prj2 = nn.Sequential(*[nn.Conv2d(self.nc, self.nc*4, 8, 8, 0)])

    def forward(self, x, num_patches=64, patch_ids=None):

        feat = self.layer1(x)
        feat1 = self.prj1(feat)
        feat2 = self.prj2(feat)

        feat1_reshape = feat1.permute(0, 2, 3, 1).flatten(1,2)        #(B, HxW, C)
        feat2_reshape = feat1.permute(0, 2, 3, 1).flatten(1,2)        # bug but good result

        if num_patches > 0:
            if patch_ids is not None:
                patch_id1 = patch_ids[0]
                patch_id2 = patch_ids[1]
            else:
                patch_id1 = torch.randperm(feat1_reshape.shape[1], device=x[0].device)
                patch_id2 = torch.randperm(feat2_reshape.shape[1], device=x[0].device)

                patch_id1 = patch_id1[:int(min(num_patches, patch_id1.shape[0]))]  # .to(patch_ids.device)
                patch_id2 = patch_id2[:int(min(num_patches, patch_id2.shape[0]))]

                patch_ids = []
                patch_ids.append(patch_id1)
                patch_ids.append(patch_id2)

            feat1_sample = feat1_reshape[:, patch_id1, :].flatten(0, 1)  # reshape(-1, x.shape[1])         #(BXHXW, C)
            feat2_sample = feat2_reshape[:, patch_id2, :].flatten(0, 1)

        else:
            feat1_sample = feat1_reshape
            feat2_sample = feat2_reshape

        feat1_sample = self.l2norm(feat1_sample)
        feat2_sample = self.l2norm(feat2_sample)

        feat_sample = []
        feat_sample.append(feat1_sample)
        feat_sample.append(feat2_sample)

        return feat_sample, patch_ids


class ConvF_down(nn.Module):
    def __init__(self, input_nc, down_level, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(ConvF_down, self).__init__()
        self.l2norm = Normalize(2)
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.gpu_ids = gpu_ids
        self.down_level = down_level

        convKernel_width = 2**down_level
        self.conv = nn.Sequential(*[nn.Conv2d(input_nc, self.nc, 1, 1, 0), nn.ReLU(), nn.Conv2d(self.nc, self.nc, convKernel_width, convKernel_width, 0)])

    def forward(self, x, num_patches=64, patch_ids=None):

        feat = self.conv(x)
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1,2)        #(B, HxW, C)

        if num_patches > 0:
            if patch_ids is not None:
                patch_id = patch_ids
            else:
                patch_id = torch.randperm(feat_reshape.shape[1], device=x[0].device)
                patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
            feat_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])         #(BXHXW, C)
        else:
            feat_sample = feat_reshape
            patch_id = []

        feat_sample = self.l2norm(feat_sample)

        return feat_sample, patch_id



class directCLR(nn.Module):
    def __init__(self, input_nc, gpu_ids=[], use_conv=False):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(directCLR, self).__init__()
        self.l2norm = Normalize(2)
        # self.nc = nc  # hard-coded
        self.mlp_init = False
        self.gpu_ids = gpu_ids
        self.conv_1 = nn.Sequential(*[nn.Conv2d(input_nc, input_nc, 1, 1, 0)])
        self.use_conv = use_conv
        # self.conv_2 = nn.Sequential(*[nn.Conv2d(input_nc, self.nc, 1, 1, 0), nn.ReLU(), nn.Conv2d(self.nc, self.nc, 1, 1, 0)])

    def forward(self, x, num_patches=64, patch_ids=None):
        _, C, _, _ = x.shape
        if self.use_conv:
            feat = x + self.conv_1(x)
            feat = feat[:, :C//2, :, :]
        else:
            feat = x[:, :C//2, :, :]
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1,2)        #(B, HxW, C)

        if num_patches > 0:
            if patch_ids is not None:
                patch_id = patch_ids
            else:
                patch_id = torch.randperm(feat_reshape.shape[1], device=x[0].device)
                patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
            feat_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])         #(BXHXW, C)
        else:
            feat_sample = feat_reshape
            patch_id = []

        feat_sample = self.l2norm(feat_sample)

        return feat_sample, patch_id


class ConvF_down_pool(nn.Module):
    def __init__(self, input_nc, down_level, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(ConvF_down_pool, self).__init__()
        self.l2norm = Normalize(2)
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.gpu_ids = gpu_ids
        self.down_level = down_level

        convKernel_width = 2**down_level
        if self.down_level == 0:
            self.conv = nn.Sequential(*[nn.Conv2d(input_nc, self.nc, 1, 1, 0), nn.ReLU(), nn.Conv2d(self.nc, self.nc, 1, 1, 0)])
        else:
            self.conv = nn.Sequential(*[nn.Conv2d(input_nc, self.nc, 1, 1, 0), nn.ReLU(), nn.AvgPool2d((convKernel_width, convKernel_width)), nn.Conv2d(self.nc, self.nc, 1, 1, 0)])

    def forward(self, x, num_patches=64, patch_ids=None):

        feat = self.conv(x)
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1,2)        #(B, HxW, C)

        if num_patches > 0:
            if patch_ids is not None:
                patch_id = patch_ids
            else:
                patch_id = torch.randperm(feat_reshape.shape[1], device=x[0].device)
                patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
            feat_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])         #(BXHXW, C)
        else:
            feat_sample = feat_reshape
            patch_id = []

        feat_sample = self.l2norm(feat_sample)
        feat_sample = self.l2norm(feat_sample)

        return feat_sample, patch_id


class PyramidalF_down(nn.Module):
    def __init__(self, input_nc, down_level, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PyramidalF_down, self).__init__()
        self.l2norm = Normalize(2)
        self.nc = nc  # hard-coded
        self.down_level = down_level
        self.mlp_init = False
        self.gpu_ids = gpu_ids
        self.layer1 = nn.Sequential(*[nn.Conv2d(input_nc, self.nc, 1, 1, 0), nn.ReLU()])

        for i in range(down_level+1):       # i -> 0 ~ down_level
            convKernel_width = 2**i
            setattr(self, 'prj_%d'%i, nn.Sequential(*[nn.Conv2d(self.nc, self.nc*convKernel_width, convKernel_width, convKernel_width, 0)]))

    def get_feats(self, input):
        feats = []
        for i in range(self.down_level + 1):
            prj = getattr(self, 'prj_%d' % i)
            feats.append(prj(input))

        feats_reshape = []
        for i in range(feats.__len__()):
            feat_reshape = feats[i].permute(0, 2, 3, 1).flatten(1, 2)  # (B, HxW, C)
            feats_reshape.append(feat_reshape)
        return feats_reshape

    def forward(self, x, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []

        feat = self.layer1(x)
        feats_reshape = self.get_feats(feat)
        for i in range(self.down_level+1):
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[i]
                else:
                    patch_id = torch.randperm(feats_reshape[i].shape[1], device=x[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
                feat_sample = feats_reshape[i][:, patch_id, :].flatten(0, 1)

            else:
                feat_sample = feats_reshape[i]
                patch_id = []

            feat_sample = self.l2norm(feat_sample)
            return_feats.append(feat_sample)
            return_ids.append(patch_id)

        return return_feats, return_ids


################################
# Functions
################################

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>
