import torch
from collections import OrderedDict

from networks.RRDBmodel import RRDBNet
from networks.dis_sn import NLayerDiscriminatorSN
from networks.network_F import ConvF_down_pool

def create_network(opt):
    in_ch=1
    net = RRDBNet(in_ch, in_ch, num_features=opt.ngf, growth_rate=8, num_blocks=3, num_layers=5, padding_mode='replicate', use_sn=True, residual_bool=opt.gen_residual)
    return net


def load_network(net, load_path, map_device, saved_module=True):
    state_dict = torch.load(load_path, map_location=map_device)
    if saved_module:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

    else:
        net.load_state_dict(state_dict)

    return net.to(map_device)


def create_dis(opt):
    in_ch = 1
    net = NLayerDiscriminatorSN(in_ch, ndf=opt.ndf, n_layers=opt.dis_depth)
    return net

def create_F(opt):
    netF_nc = opt.netF_nc
    net = ConvF_down_pool(input_nc=32, down_level=opt.netF_down, gpu_ids=opt.gpu_ids, nc=netF_nc)
    return net