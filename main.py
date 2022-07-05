import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import OrderedDict
from tqdm import trange

from data_manager._data_manager import create_TrainDataset, data_sampler, create_ValDataset
from models._model_manager import create_model
from util.visualizer import show_tsboard, show_r_tsboard
from util.util import psnr_np
from options.train_options import TrainOptions


def main():
    # -------------------------- argument-------------------------
    train_option = TrainOptions()
    args = train_option.parse()
    train_option.print_options(args)
    # -------------------------------------------------------------

    writer = SummaryWriter(args.tb_path)

    train_set = create_TrainDataset(args)
    val_set = create_ValDataset(args)

    train_sampler = data_sampler(args, args.train_size, len(train_set))
    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, sampler=train_sampler, pin_memory=True, num_workers=args.num_workers)
    print('>> train_size: %d / whole: %d'% (args.train_size, len(train_set)))
    print('>> test_size: %d' % len(val_set))
    print(val_set.__getitem__(39)['A_path'])

    model = create_model(args)
    model.set_network()
    print('>> Data ready')

    # ----------------------- training ----------------------
    ### loss_mean initiallize
    loss_mean_dict = OrderedDict()
    dataset_len = args.train_size

    psnr_best=0
    ssim_best=0
    for i in trange(args.epoch):
        ts_epoch = i+1
        for name in model.loss_names:
            loss_mean_dict['loss_'+name+'_mean'] = 0

        for j, data in enumerate(train_loader):
            model.get_input(data)
            model.optimize_parameters()

            losses = model.get_current_losses()
            for label, value in losses.items():
                loss_mean_dict['loss_'+label+'_mean'] += value
        for label, value in loss_mean_dict.items():
            loss_mean_dict[label] = value*(args.batch_size/dataset_len)


        if args.conti_train:
            ts_epoch = i+args.start_epoch

        ### show losses
        message = '\n(epoch: %d) ' % (ts_epoch)
        for k, v in loss_mean_dict.items():
            message += '%s: %.3f ' % (k, v)
        print(message)
        for k, v in loss_mean_dict.items():
            writer.add_scalar(args.log_name + '/%s' % k, v, ts_epoch)

        ### learning rate
        for scheduler in model.schedulers:
            scheduler.step()
        for param_group in model.optim_G.param_groups:
            lr = param_group['lr']
        writer.add_scalar(args.log_name+'/lr', lr, ts_epoch)

        ### show result images & save
        if ((ts_epoch) % args.im_period == 0):
            temp_input = val_set.__getitem__(39)
            full_infer_np = model.print_temp_result_A2B(temp_input)
            temp_input_np = model.real_A.squeeze().cpu().numpy()
            if ts_epoch > args.epoch_decay-1:
                if args.is_wt:
                    show_r_tsboard(writer, full_infer_np, ts_epoch, args.log_name + '/full_infer', 0.1)
                else:
                    show_tsboard(writer, full_infer_np, ts_epoch, args.log_name + '/full_infer')
                show_r_tsboard(writer, full_infer_np - temp_input_np, ts_epoch, args.log_name + '/diff', 0.05)

            ### image & generator save
            np.save(args.img_save_path + '/' + args.log_name + '_temp(%d)_full_infer.npy' % (ts_epoch), full_infer_np)
            if (ts_epoch>10):
                model.save_tmp_gen(ts_epoch, args.model_save_path)

        ### whole network save
        if ((ts_epoch) % args.save_period == 0):
            model.save_networks(ts_epoch, args.model_save_path)

        # ----------------------- evaluation ----------------------
        if ((ts_epoch) % args.val_period == 0)&(ts_epoch>args.epoch//2):
        # if ((ts_epoch) % args.val_period == 0):
            print('>> evaluation')
            psnr_avg=0
            ssim_avg=0
            val_size = val_set.__len__()
            for j in range(val_size):
                temp_input = val_set.__getitem__(j)
                input_np_l = temp_input['A_l'].squeeze().numpy()
                ref_np_h = temp_input['B'].squeeze().numpy()
                ref_np_l = temp_input['B_l'].squeeze().numpy()
                model_output = model.print_temp_result_A2B(temp_input)

                if args.is_wt:
                    full_infer_np = input_np_l + model_output
                    ref_img = ref_np_l + ref_np_h
                else:
                    full_infer_np = model_output
                    ref_img = ref_np_l
                ref_max = ref_img.max()
                psnr_f = psnr_np(full_infer_np, ref_img, ref_max)
                from skimage.metrics import structural_similarity as ssim
                ssim_f = ssim(full_infer_np, ref_img, data_range=np.amax(ref_max))
                psnr_avg+=psnr_f/val_size
                ssim_avg+=ssim_f/val_size

            if psnr_avg > psnr_best:
                print('>>> save best model')
                psnr_best = psnr_avg
                ssim_best = ssim_avg
                model.save_bst_gen(args.model_save_path)
            print('PSNR_%.4f  SSIM_%.4f |(best)PSNR_%.4f SSIM_%.4f'%(psnr_avg, ssim_avg, psnr_best, ssim_best))

if __name__ == '__main__':
    print('>> start')
    main()
