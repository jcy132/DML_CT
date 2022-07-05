
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler



def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    ## GPU Device
    if len(gpu_ids)>0:
        net.to(gpu_ids[0])
        net = nn.DataParallel(net, gpu_ids)

    ## weight initialize
    init_weights(net, init_type, init_gain = init_gain)
    return net


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def get_scheduler(optimizer, epoch_count, epoch_decay, decay_policy):
    if decay_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0-max(0, epoch+epoch_count - epoch_decay) / float(epoch_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda_rule)
    elif decay_policy == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_decay], last_epoch=-1, gamma=0.5)
    return scheduler