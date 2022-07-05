import torch
import torch.nn as nn
import itertools
from collections import OrderedDict

from util.util import crop_patch
from util.util_model import init_net, get_scheduler

from networks._network_manager import create_network, create_dis, create_F
from networks.patchMetric import PatchMetricLoss

###########################################
# function
###########################################

class GANLoss(nn.Module):
    def __init__(self, gan_mode):
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target = torch.ones(prediction.shape).to(prediction.device)
        else:
            target = torch.zeros(prediction.shape).to(prediction.device)
        return target

    def __call__(self, prediction, target_is_real):
        target = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target)
        return loss


###########################################
# model
###########################################

class cutGAN_model():
    def __init__(self, opt):
        self.opt = opt
        self.in_ch = 1
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))

        self.criterionGAN = GANLoss(opt.gan_mode)
        self.criterionIdt = nn.L1Loss()

        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        self.criterionMetric = []
        for i, nce_layer in enumerate(self.nce_layers):
            self.criterionMetric.append(PatchMetricLoss(opt=opt).to(self.device))

        self.loss_names = ['G_A2B', 'D_B', 'idt_B']
        if opt.lambda_Metric>0.0:
            self.loss_names.append('Metric')
        for name in self.loss_names:
            setattr(self, 'loss_'+name, 0)


    def set_network(self):
        in_ch = self.in_ch

        # Get Network
        self.genA2B = create_network(self.opt)
        self.disB = create_dis(self.opt)

        # initialize Network
        self.genA2B = init_net(self.genA2B, self.opt.init_type, self.opt.init_gain, self.opt.gpu_ids)
        self.disB = init_net(self.disB, self.opt.init_type, self.opt.init_gain, self.opt.gpu_ids)

        self.netFs = []
        F_params=[]
        for i, n in enumerate (self.nce_layers):
            netF = create_F(self.opt)
            netF = init_net(netF, self.opt.init_type, self.opt.init_gain, self.opt.gpu_ids)
            self.netFs.append(netF)
            F_params.append(netF.parameters())

        # Load Network
        if self.opt.conti_train:
            self.load_network()

        self.optim_G = torch.optim.Adam(self.genA2B.parameters(), lr=self.opt.g_lr, betas=(0.5, 0.999))
        self.optim_D_B = torch.optim.Adam(self.disB.parameters(), lr=self.opt.d_lr, betas=(0.5, 0.999))
        self.optim_F = torch.optim.Adam(itertools.chain(*F_params), lr=self.opt.g_lr, betas=(0.5, 0.999))
        self.optimizers = [self.optim_G, self.optim_D_B, self.optim_F]
        self.schedulers = [get_scheduler(optimizer, epoch_count=1, epoch_decay=self.opt.epoch_decay, decay_policy=self.opt.decay_policy) for optimizer in self.optimizers]

    def load_network(self, saved_module=False, model=None):
        if model==None:
            load_list = ['genA2B', 'disB']
        else:
            load_list = [model]
        for name in load_list:
            load_path = getattr(self.opt, name+'_load_path')
            net = getattr(self, name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            state_dict = torch.load(load_path, map_location=self.device)
            if saved_module:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]
                    new_state_dict[name] = v

                net.load_state_dict(new_state_dict)

            else:
                net.load_state_dict(state_dict)

    def load_network_test(self, saved_module=False, model=None):
        if model==None:
            load_list = ['genA2B']
        else:
            load_list = [model]
        for name in load_list:
            load_path = getattr(self.opt, name+'_load_path')
            net = getattr(self, name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            state_dict = torch.load(load_path, map_location=self.device)
            if saved_module:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]
                    new_state_dict[name] = v

                net.load_state_dict(new_state_dict)

            else:
                net.load_state_dict(state_dict)


    def get_input(self, input):
        """
        :param input: get data from dataloader (B, C, H, W)
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)


    def set_requires_grad(self, nets, requires_grad=False):
        # Make as list
        if not isinstance(nets, list):
            nets = [nets]
        # set require_grad
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_current_losses(self):
        losses = OrderedDict()
        for name in self.loss_names:
            losses[name] = float(getattr(self, 'loss_'+name))
        return losses

    def save_networks(self, epoch, save_dir):
        torch.save(self.genA2B.module.cpu().state_dict(), save_dir + '/'+self.opt.log_name+'_temp(%d)_genA2B.pth'%(epoch))
        torch.save(self.disB.module.cpu().state_dict(), save_dir + '/'+self.opt.log_name+'_temp(%d)_disB.pth' % (epoch))

        self.genA2B.to(self.device)
        self.disB.to(self.device)


    def save_tmp_gen(self, epoch, save_dir):
        torch.save(self.genA2B.module.cpu().state_dict(), save_dir + '/'+self.opt.log_name+'_temp(%d)_genA2B.pth' % (epoch))
        self.genA2B.to(self.device)

    def save_bst_gen(self, save_dir):
        torch.save(self.genA2B.module.cpu().state_dict(), save_dir + '/'+self.opt.log_name+'_bst_genA2B.pth')
        self.genA2B.to(self.device)

    def print_temp_result_A2B(self, input):
        self.real_A = input['A'].unsqueeze(0).to(self.device)
        temp_result = self.genA2B(self.real_A).squeeze().detach().cpu().numpy()
        return temp_result

    ### training functions
    def forward(self):
        self.fake_B = self.genA2B(self.real_A)


    def backward_G(self, flag_repeat):
        self.loss_G_A2B = self.criterionGAN(self.disB(self.fake_B), True) * self.opt.lambda_GAN

        ## Idt
        self.idt_B = self.genA2B(self.real_B)
        if self.opt.lambda_idt>0:
            self.loss_idt_B = self.criterionIdt(self.real_B, self.idt_B) * self.opt.lambda_idt
        else:
            self.loss_idt_B = 0


        real_A_feat = self.genA2B(self.real_A, encode_only=True, layers=self.nce_layers)
        fake_B_feat = self.genA2B(self.fake_B, encode_only=True, layers=self.nce_layers)
        idt_B_feat = self.genA2B(self.idt_B, encode_only=True, layers=self.nce_layers)
        if self.opt.nce_idt:
            real_B_feat = self.genA2B(self.real_B, encode_only=True, layers=self.nce_layers)

        ## Deep metric learning
        if self.opt.lambda_Metric > 0.0:
            self.loss_Metric = self.calculate_Metric_loss(real_A_feat, fake_B_feat)
        else:
            self.loss_Metric, self.loss_Metric_bd = 0.0, 0.0

        self.loss_Metric_Y = 0
        if self.opt.nce_idt and self.opt.lambda_Metric > 0.0:
            self.loss_Metric_Y = self.calculate_Metric_loss(real_B_feat, idt_B_feat)
            loss_Metric_both = (self.loss_Metric + self.loss_Metric_Y) * 0.5
        else:
            loss_Metric_both = self.loss_Metric

        self.loss_G = self.loss_G_A2B + self.loss_idt_B + loss_Metric_both

        if flag_repeat == 1:
            self.loss_G.backward()
        else:
            self.loss_G.backward(retain_graph=True)


    def calculate_Metric_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_qs = tgt
        feat_ks = src

        vector_k_pool = []
        vector_q_pool = []
        for feat_q, feat_k, netF in zip(feat_qs, feat_ks, self.netFs):
            vector_k, sample_ids = netF(feat_k, self.opt.num_patches, None)
            vector_q, _ = netF(feat_q, self.opt.num_patches, sample_ids)
            vector_k_pool.append(vector_k)
            vector_q_pool.append(vector_q)

        total_Metric_loss = 0.0
        if self.opt.netF == 'pyramidF' or self.opt.netF == 'pyramidF_down':  # vector_k is list
            level = vector_q_pool[0].__len__()
            for f_q, f_k, crit, nce_layer in zip(vector_q_pool, vector_k_pool, self.criterionMetric, self.nce_layers):
                for i in range(level):
                    loss = crit(f_q[i], f_k[i]) * self.opt.lambda_Metric
                    total_Metric_loss += loss.mean() / level

        else:
            for f_q, f_k, crit, nce_layer in zip(vector_q_pool, vector_k_pool, self.criterionMetric, self.nce_layers):
                loss = crit(f_q, f_k) * self.opt.lambda_Metric
                total_Metric_loss += loss.mean()

        return total_Metric_loss / n_layers


    def backward_D_basic(self, netD, real, fake, flag_repeat):
        pred_real = netD(real)
        pred_fake = netD(fake.detach())

        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D = (loss_D_real+loss_D_fake)*0.5

        if flag_repeat == 1:
            loss_D.backward()
        else:
            loss_D.backwrad(retain_graph=True)
        return loss_D

    def backward_D_B(self, flag_repeat):
        self.loss_D_B = self.backward_D_basic(self.disB, self.real_B, self.fake_B, flag_repeat)


    def optimize_parameters(self):
        self.forward()

        #Train Generator
        self.set_requires_grad([self.disB], False)
        for i in range(self.opt.g_train_num):
            flag_repeat = self.opt.g_train_num - i
            self.optim_G.zero_grad()
            self.optim_F.zero_grad()
            self.backward_G(flag_repeat)
            self.optim_G.step()
            self.optim_F.step()

        #Train Discriminator
        self.set_requires_grad([self.disB], True)
        for j in range(self.opt.d_train_num):
            flag_repeat = self.opt.d_train_num - j
            self.optim_D_B.zero_grad()
            self.backward_D_B(flag_repeat)
            self.optim_D_B.step()
