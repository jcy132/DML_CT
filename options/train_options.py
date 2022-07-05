
import torch
import argparse
import os

from util.util import check_dir

import util.util as util
class TrainOptions():

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        ## save & path
        parser.add_argument('--prj_name', type=str, default='prac')
        parser.add_argument('--log_name', type=str, default='prac')
        parser.add_argument('--data_root', type=str, default='./data')

        ## wavelet
        parser.add_argument('--is_wt', type=str, default='True')
        parser.add_argument('--wt_type', type=str, default='db3')
        parser.add_argument('--wt_level', type=int, default=6)
        parser.add_argument('--wt_pad_mode', type=str, default='symmetric')


        ## Dataset
        parser.add_argument('--dataset_name', type=str, default='AAPM')
        parser.add_argument('--train_size', type=int, default=3112)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--image_size', type=int, default=512)
        parser.add_argument('--patch_size', type=int, default=128)
        parser.add_argument('--margin', type=int, default=30)
        parser.add_argument('--crop_mode', type=str, default='rect')      # radial | rect


        ### Epoch
        parser.add_argument('--epoch', type=int, default=200)
        parser.add_argument('--epoch_decay', type=int, default=100)
        parser.add_argument('--decay_policy', type=str, default='linear')
        parser.add_argument('--save_period', type=int, default=100)
        parser.add_argument('--im_period', type=int, default=10)
        parser.add_argument('--val_period', type=int, default=10)


        ### Training setting
        parser.add_argument('--gpu_ids', type=str, default='0,1')  # --gpu_ids 0 | 1 | 0,1
        parser.add_argument('--g_lr', type=float, default=2e-4)
        parser.add_argument('--d_lr', type=float, default=2e-4)
        parser.add_argument('--g_train_num', type=int, default=1)
        parser.add_argument('--d_train_num', type=int, default=1)
        parser.add_argument('--num_workers', type=int, default=4)


        ### Loss
        parser.add_argument('--lambda_GAN', type=float, default=1)
        parser.add_argument('--lambda_idt', type=float, default=5)
        parser.add_argument('--lambda_Metric', type=float, default=0.1)
        parser.add_argument('--metric_T', type=float, default=0.15, help='temperature for metric loss')


        ### Network
        parser.add_argument('--model', type=str, default='cutGAN')
        parser.add_argument('--gan_mode', type=str, default='lsgan')
        parser.add_argument('--gen_model', type=str, default ='rrdb')
        parser.add_argument('--gen_residual', type=util.str2bool, nargs='?',const=True, default=False)
        parser.add_argument('--dis_model', type=str, default='dis_sn')
        parser.add_argument('--netF', type=str, default='convF_down_pool',help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=128)
        parser.add_argument('--netF_down', type=int, default=0)
        parser.add_argument('--nce_layers', type=str, default='1,2', help='compute NCE loss on which layers')
        parser.add_argument('--num_patches', type=int, default=128, help='number of patches per layer')
        parser.add_argument('--mapper_model', type=str, default='mlp')

        parser.add_argument('--ngf', type=int, default=32)
        parser.add_argument('--ndf', type=int, default=32)
        parser.add_argument('--init_type', type=str, default='normal')
        parser.add_argument('--init_gain', type=float, default=0.01)
        parser.add_argument('--dis_depth', type=int, default=3)

        ##conti_train
        parser.add_argument('--conti_train', action='store_true')
        parser.add_argument('--gen_A2B_load_path', type=str, default = '')
        parser.add_argument('--dis_B_load_path', type=str, default = 'disB.pth')
        parser.add_argument('--start_epoch', type=int, default=101)



        ### patch-wise setting
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')

        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--alpha', type=float, default=0.2)

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt, _ = parser.parse_known_args() #get arg from command line

        self.parser = parser
        return parser.parse_args()  #get arg from command line, '' not work


    def parse(self):
        opt = self.gather_options()
        # boolean variables
        opt.is_wt = (opt.is_wt =='True')

        # make dirs
        ## change folder-name style here
        opt.folder_name = opt.prj_name
        if opt.is_wt:
            opt.folder_name = opt.prj_name + '_' + opt.wt_type + '_%d'%(opt.wt_level)

        opt.tb_path = './checkpoints/' + opt.folder_name + '/log'
        opt.img_save_path = './checkpoints/' + opt.folder_name + '/img'
        opt.model_save_path = './checkpoints/' + opt.folder_name + '/m_save'


        check_dir(opt.tb_path)
        check_dir(opt.img_save_path)
        check_dir(opt.model_save_path)

        # convert gpu_ids as integer
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            opt.gpu_ids.append(id)
        if len(opt.gpu_ids)>0:
            torch.cuda.set_device(opt.gpu_ids[0])


        self.opt = opt
        return opt

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        message_txt = message

        for k, v in vars(opt).items():
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message_txt += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        message_txt += '----------------- End -------------------'
        print(message_txt)

        # save to the disk
        expr_dir = './checkpoints/' + opt.folder_name
        check_dir(expr_dir)
        if not opt.conti_train:
            file_name = os.path.join(expr_dir, opt.log_name+'_train.txt')
        else:
            file_name = os.path.join(expr_dir, opt.log_name +'_conti%d'%(opt.start_epoch) +'_train.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message_txt)
            opt_file.write('\n')