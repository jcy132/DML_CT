import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pywt


import math
import os

import random
from torch.autograd import Variable
import torch.nn.functional as F

from scipy.ndimage import uniform_filter
from skimage.util.arraycrop import crop

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_image(path):                   # path의 image를 torch float tensor로 반환, max, norm 진행 ( channel * W * H )
    image = np.load(path)
    image = np.maximum(image, 0)        # 음수 값을 제거
    norm_const = np.max(image)
    image = image/norm_const            # max=1 이도록 normalize
    image = torch.from_numpy(image)     # (W * H)
    image = image.unsqueeze(0)          # (channel * W * H)
    return image


def show_image(path):
    image = np.load(path)
    img = Image.fromarray(image)
    plt.figure(1)
    plt.imshow(img, cmap='gray')
    plt.show()


def show_np(array, num):
    img = Image.fromarray(array)
    plt.figure(num)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def show_np_r(array, min, max, num):
    plt.figure(num)
    plt.imshow(array, norm=None, cmap='gray', vmin= min, vmax=max)
    plt.axis('off')
    plt.show()

def show_np_r_scaleBar(array, min, max, num, bar_min, bar_max):
    plt.figure(num)
    plt.imshow(array, norm=None, cmap='gray', vmin= min, vmax=max)
    plt.axis('off')
    plt.colorbar()
    plt.show()

def save_tight(array, min, max, name):
    plt.imshow(array, norm=None, cmap='gray', vmin= min, vmax=max)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(fname = name+'.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def diff_show(np1, np2, range1, range2, num):
    plt.figure(num)
    diff = np1-np2
    mean = np.mean(diff)
    plt.imshow(diff, norm=None, cmap='gray', vmin= range1, vmax=range2)
    plt.show()


def save_image_png(array, path):
    img = Image.fromarray(array)
    img.save(path)


def check_dir(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def psnr(img1_tensor, img2_tensor):           #  tensor input (channel, W, H)
    img1 = img1_tensor.numpy()
    img2 = img2_tensor.numpy()

    mse = np.mean((img1-img2)**2)
    max = 1

    if mse == 0:
        result = 100
    else:
         result = 20 * math.log10(max/math.sqrt(mse))

    return result

def psnr_np(img1, img2, max):           #  tensor input (channel, W, H)
    mse = np.mean((img1-img2)**2)

    if mse == 0:
        result = 100
    else:
         result = 20 * math.log10(max/math.sqrt(mse))

    return result


def nmse(img1_np, img2_np):
    img1 = img1_np
    img2 = img2_np
    mse = np.mean((img1 - img2) ** 2)
    nmse = mse/np.sum(img1**2)
    return nmse

def ssim(img1_np, img2_np, c1, c2):
    '''
    c1 = (0.01*range)**2
    c2 = (0.03*range)**2

    m1 = np.mean(img1_np)
    m2 = np.mean(img2_np)

    var1 = np.var(img1_np)
    var2 = np.var(img2_np)
    cov12=np.sum((img1_np-m1)*(img2_np-m2))/(len(img1_np)-1)**2

    v = ((2*m1*m2+c1)*(2*cov12+c2))/((m1**2+m2**2+c1)*(var1+var2+c2))
    '''

    d_range = 2 #(-1, 1)
    img1_np = img1_np.astype(np.float64)
    img2_np = img2_np.astype(np.float64)
    k1 = 0.01
    k2 = 0.03
    sigma = 1.5

    c1=(k1*d_range)**2#6.5025
    c2=(k2*d_range)**2#58.5225
    c3 = c2/2


    m1 = np.mean(img1_np)
    m2 = np.mean(img2_np)

    var1 = np.var(img1_np)
    var2 = np.var(img2_np)

    sig1 = np.sqrt(np.var(img1_np))
    sig2 = np.sqrt(np.var(img2_np))
    #sig12 = np.sqrt(np.var(img1_np * img2_np) - m1 * m2)
    sig12 = np.sqrt(np.mean((img1_np-np.mean(img1_np))*(img2_np-np.mean(img2_np))))

    l = (2*m1*m2+c1)/(m1**2+m2**2+c1)
    c = (2*sig1*sig2+c2)/(var1+var2+c2)
    #s = (sig12+c3)/(sig1*sig2+c3)
    v=l*c
    return v, l*c, l, c, s

def ssim_web(img1_np, img2_np):

    d_range = 2 #(-1, 1)
    x = img1_np.astype(np.float64)
    y = img2_np.astype(np.float64)
    K1 = 0.01
    K2 = 0.03
    sigma = 1.5
    win_size = 7

    ndim = x.ndim
    NP = win_size ** ndim
    conv_norm = NP / (NP-1)

    ux = uniform_filter(x, size=7)
    uy = uniform_filter(y, size=7)

    uxx = uniform_filter(x * x, size=7)
    uyy = uniform_filter(y * y, size=7)
    uxy = uniform_filter(y * y, size=7)
    vx = conv_norm*(uxx - ux * ux)
    vy = conv_norm*(uyy - uy * uy)
    vxy = conv_norm*(uxy - ux * uy)

    R = d_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    c3 = C2 / 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    pad = (win_size - 1) // 2
    mssim = crop(S, pad).mean()

    return mssim


def crop_patch(img, lbl, img_size, patch_size, margin):
    #seed_u = random.randrange(0, img_size - patch_size)
    #seed_v = random.randrange(0, img_size - patch_size)

    if img_size - patch_size-margin == 0:
        seed_u = 0
        seed_v = 0
    else:
        seed_u = random.randrange(margin, img_size - patch_size-margin)
        seed_v = random.randrange(margin, img_size - patch_size-margin)

    cropped_img = img[:, :, seed_u:seed_u + patch_size, seed_v:seed_v + patch_size]
    cropped_lbl = lbl[:, :, seed_u:seed_u + patch_size, seed_v:seed_v + patch_size]

    return (cropped_img, cropped_lbl)

def crop_patch_radial(img, lbl, img_size, patch_size, margin):
    if img_size - patch_size-margin == 0:
        seed_u = 0
        seed_v = 0
    else:
        radius = img_size / 2 - patch_size / 2 - margin
        seed_ang = random.randrange(0, 2 * (math.pi))
        seed_u = img_size/2 - radius*math.sin(seed_ang)
        seed_v = img_size/2 + radius*math.cos(seed_ang)

    cropped_img = img[:, :, seed_u:seed_u + patch_size, seed_v:seed_v + patch_size]
    cropped_lbl = lbl[:, :, seed_u:seed_u + patch_size, seed_v:seed_v + patch_size]

    return (cropped_img, cropped_lbl)


def assemble_patch(model, device, img, img_size, patch_size):                            # device: model's device / img : torch.tensor(W,H)
    result_np = np.zeros((img_size, img_size))  # Result image initialize

    for k in range(int(img_size / patch_size + 1)):
        for m in range(int(img_size / patch_size) + 1):
            seed_u = k * patch_size
            seed_v = m * patch_size

            if (k != int(img_size / patch_size)) & (m != int(img_size / patch_size)):
                patch = img[seed_u:seed_u + patch_size, seed_v:seed_v + patch_size]
            elif (k != int(img_size / patch_size)) & (m == int(img_size / patch_size)):                 # m 성분이 끝부분 경우
                patch = img[seed_u:seed_u + patch_size, (img_size - patch_size):img_size]
            elif (k == int(img_size / patch_size)) & (m != int(img_size / patch_size)):                 # k 성분이 끝부분 경우
                patch = img[(img_size - patch_size):img_size, seed_v:seed_v + patch_size]
            else:                                                                                       # k, m 성분이 끝부분 경우
                patch = img[img_size - patch_size : img_size, (img_size - patch_size):img_size]

            patch = patch.unsqueeze(0).unsqueeze(0)                             # (batch, channel, W, H)

            input = Variable(patch).to(device)
            out = model.forward(input)
            out = out.cpu()
            result_patch = out.squeeze(0).squeeze(0)
            result_patch = result_patch.detach().numpy()  # grad 성분이 없어야 함. 그러므로 detach

            if (k != int(img_size / patch_size)) & (m != int(img_size / patch_size)):
                result_np[seed_u:seed_u + patch_size, seed_v:seed_v + patch_size] = result_patch
            elif (k != int(img_size / patch_size)) & (m == int(img_size / patch_size)):
                result_np[seed_u : seed_u + patch_size, (img_size - patch_size):img_size] = result_patch
            elif (k == int(img_size / patch_size)) & (m != int(img_size / patch_size)):
                result_np[(img_size - patch_size):img_size, seed_v:seed_v + patch_size] = result_patch
            else:
                result_np[(img_size - patch_size):img_size, (img_size - patch_size):img_size] = result_patch

    return result_np


def assemble_patch_overlap(model, device, img, patch_size, stride):                            # device: model's device / img : torch.tensor(W,H)
    result_np = np.zeros(img.size())  # Result image initialize
    denom_np = np.zeros(img.size())
    denom_patch = np.ones((patch_size, patch_size))

    num_x = int(np.ceil(img.size()[1]/stride))
    num_y = int(np.ceil(img.size()[0]/stride))

    n_x = num_x-1
    n_y = num_y-1

    for k in range(num_y):
        for m in range(num_x):
            seed_y = k * stride
            seed_x = m * stride

            if (k != n_y) & (m != n_x):
                patch = img[seed_y:seed_y + patch_size, seed_x:seed_x + patch_size]
            elif (k != n_y) & (m == n_x):                 # m 성분이 끝부분 경우
                patch = img[seed_y:seed_y + patch_size, (img.size()[1] - patch_size):img.size()[1]]
            elif (k == n_y) & (m != n_x):                 # k 성분이 끝부분 경우
                patch = img[(img.size()[0] - patch_size):img.size()[0], seed_x:seed_x + patch_size]
            else:                                                                                       # k, m 성분이 끝부분 경우
                patch = img[(img.size()[0] - patch_size):img.size()[0], (img.size()[1] - patch_size):img.size()[1]]

            patch = patch.unsqueeze(0).unsqueeze(0)                                                     # (batch, channel, W, H)

            input = Variable(patch).to(device)
            out = model.forward(input)
            out = out.cpu()
            result_patch = out.squeeze(0).squeeze(0)
            result_patch = result_patch.detach().numpy()  # grad 성분이 없어야 함. 그러므로 detach

            if (k != n_y) & (m != n_x):
                result_np[seed_y:seed_y + patch_size, seed_x:seed_x + patch_size] += result_patch
                denom_np[seed_y:seed_y + patch_size, seed_x:seed_x + patch_size] += denom_patch
            elif (k != n_y) & (m == n_x):
                result_np[seed_y:seed_y + patch_size, (img.size()[1] - patch_size):img.size()[1]] += result_patch
                denom_np[seed_y:seed_y + patch_size, (img.size()[1] - patch_size):img.size()[1]] += denom_patch
            elif (k == n_y) & (m != n_x):
                result_np[(img.size()[0] - patch_size):img.size()[0], seed_x:seed_x + patch_size] += result_patch
                denom_np[(img.size()[0] - patch_size):img.size()[0], seed_x:seed_x + patch_size] += denom_patch
            else:
                result_np[(img.size()[0] - patch_size):img.size()[0], (img.size()[1] - patch_size):img.size()[1]] += result_patch
                denom_np[(img.size()[0] - patch_size):img.size()[0], (img.size()[1] - patch_size):img.size()[1]] += denom_patch

    return result_np/denom_np

def ones(batch_size, patch_size, train_patch):
    if (train_patch != True):
        ones_label = torch.FloatTensor(batch_size, 1, 62, 62).fill_(1.0)
    elif (train_patch == True):
        a = int(patch_size / 8 - 2)                                             # dis network depth 따라 달라짐. 유의.
        ones_label = torch.FloatTensor(batch_size, 1, a, a).fill_(1.0)

    return ones_label


def zeros(batch_size, patch_size, train_patch):
    if (train_patch != True):
        zeros_label = torch.FloatTensor(batch_size, 1, 62, 62).fill_(0.0)
    elif (train_patch == True):
        a = int(patch_size / 8 - 2)
        zeros_label = torch.FloatTensor(batch_size, 1, a, a).fill_(0.0)

    return zeros_label


def ones_manual(batch_size, size):
    ones_label = torch.FloatTensor(batch_size, 1, size, size).fill_(1.0)
    return ones_label


def zeros_manual(batch_size, size):
    zeros_label = torch.FloatTensor(batch_size, 1, size, size).fill_(0.0)
    return zeros_label


def comp(image_torch):
    #return image_torch**(1/5)
    #return torch.log(image_torch+2)
    return image_torch

def decomp(image_torch):
    #return image_torch**(5)
    #return torch.exp(image_torch)-2
    return image_torch


def show_losses(writer, loss_mean_dict, ts_epoch, opt):
    loss_D_A_mean = loss_mean_dict['loss_D_A_mean']
    loss_D_B_mean = loss_mean_dict['loss_D_B_mean']
    loss_G_A2B_mean = loss_mean_dict['loss_G_A2B_mean']
    loss_G_B2A_mean = loss_mean_dict['loss_G_B2A_mean']
    loss_cyc_A_mean = loss_mean_dict['loss_cyc_A_mean']
    loss_cyc_B_mean = loss_mean_dict['loss_cyc_B_mean']

    # if opt.gan_mode == 'wgan':
    #     writer.add_scalars(opt.log_name+'/loss_D_A', {'D_A': loss_D_A_mean, 'gp_A': loss_mean_dict['loss_gp_A_mean']}, ts_epoch)
    #     writer.add_scalars(opt.log_name+'/loss_D_B', {'D_B': loss_D_B_mean, 'gp_B': loss_mean_dict['loss_gp_B_mean']}, ts_epoch)
    # else:
    writer.add_scalar(opt.log_name+'/loss_D_A', loss_D_A_mean, ts_epoch)
    writer.add_scalar(opt.log_name+'/loss_D_B', loss_D_B_mean, ts_epoch)

    writer.add_scalars(opt.log_name+'/loss_G_A2B', {'GAN_Loss': loss_G_A2B_mean, 'Cyclic_Loss': loss_cyc_A_mean}, ts_epoch)
    writer.add_scalars(opt.log_name+'/loss_G_B2A', {'GAN_Loss': loss_G_B2A_mean, 'Cyclic_Loss': loss_cyc_B_mean}, ts_epoch)

    if opt.lambda_idt > 0:
        writer.add_scalars(opt.log_name+'/loss_idt', {'idt_A': loss_mean_dict['loss_idt_A_mean'], 'idt_B': loss_mean_dict['loss_idt_B_mean']}, ts_epoch)


def infer_patch_model(model, img, patch_size, stride):
    result_np = np.zeros(img.size())  # Result image initialize
    denom_np = np.zeros(img.size())
    denom_patch = np.ones((patch_size, patch_size))

    num_x = int(np.ceil(img.size()[1] / stride))
    num_y = int(np.ceil(img.size()[0] / stride))

    n_x = num_x - 1
    n_y = num_y - 1

    for k in range(num_y):
        for m in range(num_x):
            seed_y = k * stride
            seed_x = m * stride

            if (k != n_y) & (m != n_x):
                patch = img[seed_y:seed_y + patch_size, seed_x:seed_x + patch_size]
            elif (k != n_y) & (m == n_x):                 # m 성분이 끝부분 경우
                patch = img[seed_y:seed_y + patch_size, (img.size()[1] - patch_size):img.size()[1]]
            elif (k == n_y) & (m != n_x):                 # k 성분이 끝부분 경우
                patch = img[(img.size()[0] - patch_size):img.size()[0], seed_x:seed_x + patch_size]
            else:                                                                                       # k, m 성분이 끝부분 경우
                patch = img[(img.size()[0] - patch_size):img.size()[0], (img.size()[1] - patch_size):img.size()[1]]

            patch = patch.unsqueeze(0).unsqueeze(0)                                                     # (batch, channel, W, H)
            input = patch.to(model.device)
            out = model.gen_A2B(input)
            result_patch = out.cpu().squeeze(0).squeeze(0).detach().numpy()  # grad 성분이 없어야 함. 그러므로 detach

            if (k != n_y) & (m != n_x):
                result_np[seed_y:seed_y + patch_size, seed_x:seed_x + patch_size] += result_patch
                denom_np[seed_y:seed_y + patch_size, seed_x:seed_x + patch_size] += denom_patch
            elif (k != n_y) & (m == n_x):
                result_np[seed_y:seed_y + patch_size, (img.size()[1] - patch_size):img.size()[1]] += result_patch
                denom_np[seed_y:seed_y + patch_size, (img.size()[1] - patch_size):img.size()[1]] += denom_patch
            elif (k == n_y) & (m != n_x):
                result_np[(img.size()[0] - patch_size):img.size()[0], seed_x:seed_x + patch_size] += result_patch
                denom_np[(img.size()[0] - patch_size):img.size()[0], seed_x:seed_x + patch_size] += denom_patch
            else:
                result_np[(img.size()[0] - patch_size):img.size()[0], (img.size()[1] - patch_size):img.size()[1]] += result_patch
                denom_np[(img.size()[0] - patch_size):img.size()[0], (img.size()[1] - patch_size):img.size()[1]] += denom_patch

    return result_np/denom_np



def infer_patch_net(net, net_device, img, patch_size, stride):
    result_np = np.zeros(img.size())  # Result image initialize
    denom_np = np.zeros(img.size())
    denom_patch = np.ones((patch_size, patch_size))

    num_x = int(np.ceil(img.size()[1] / stride))
    num_y = int(np.ceil(img.size()[0] / stride))

    n_x = num_x - 1
    n_y = num_y - 1

    for k in range(num_y):
        for m in range(num_x):
            seed_y = k * stride
            seed_x = m * stride

            if (k != n_y) & (m != n_x):
                patch = img[seed_y:seed_y + patch_size, seed_x:seed_x + patch_size]
            elif (k != n_y) & (m == n_x):                 # m 성분이 끝부분 경우
                patch = img[seed_y:seed_y + patch_size, (img.size()[1] - patch_size):img.size()[1]]
            elif (k == n_y) & (m != n_x):                 # k 성분이 끝부분 경우
                patch = img[(img.size()[0] - patch_size):img.size()[0], seed_x:seed_x + patch_size]
            else:                                                                                       # k, m 성분이 끝부분 경우
                patch = img[(img.size()[0] - patch_size):img.size()[0], (img.size()[1] - patch_size):img.size()[1]]

            patch = patch.unsqueeze(0).unsqueeze(0)                                                     # (batch, channel, W, H)
            input = patch.to(net_device)
            out = net(input)
            result_patch = out.cpu().squeeze(0).squeeze(0).detach().numpy()  # grad 성분이 없어야 함. 그러므로 detach

            if (k != n_y) & (m != n_x):
                result_np[seed_y:seed_y + patch_size, seed_x:seed_x + patch_size] += result_patch
                denom_np[seed_y:seed_y + patch_size, seed_x:seed_x + patch_size] += denom_patch
            elif (k != n_y) & (m == n_x):
                result_np[seed_y:seed_y + patch_size, (img.size()[1] - patch_size):img.size()[1]] += result_patch
                denom_np[seed_y:seed_y + patch_size, (img.size()[1] - patch_size):img.size()[1]] += denom_patch
            elif (k == n_y) & (m != n_x):
                result_np[(img.size()[0] - patch_size):img.size()[0], seed_x:seed_x + patch_size] += result_patch
                denom_np[(img.size()[0] - patch_size):img.size()[0], seed_x:seed_x + patch_size] += denom_patch
            else:
                result_np[(img.size()[0] - patch_size):img.size()[0], (img.size()[1] - patch_size):img.size()[1]] += result_patch
                denom_np[(img.size()[0] - patch_size):img.size()[0], (img.size()[1] - patch_size):img.size()[1]] += denom_patch

    return result_np/denom_np



def infer_patch_model_edgeRemove(model, img, patch_size, stride, edgeRemove_pix):
    #pad input img
    h, w = img.shape
    result_np = np.zeros(img.size())
    denom_np = np.zeros(img.size())
    pad_vector = (edgeRemove_pix, edgeRemove_pix, edgeRemove_pix, edgeRemove_pix)
    img = F.pad(img.view(1, 1, h, w), pad_vector, mode='replicate').squeeze()
    #img = np.pad(img, pad_width=edgeRemove_pix)

    #compensate edgeRemoved_pix
    patch_size_removed = patch_size-2*edgeRemove_pix
    stride_removed = stride-2*edgeRemove_pix
    denom_patch = np.ones((patch_size_removed, patch_size_removed))
    num_x = int(np.ceil(w / stride_removed))
    num_y = int(np.ceil(h / stride_removed))

    n_x = num_x - 1
    n_y = num_y - 1

    for k in range(num_y):
        for m in range(num_x):
            seed_y = k * stride_removed
            seed_x = m * stride_removed

            if (k != n_y) & (m != n_x):
                patch = img[seed_y:seed_y + patch_size, seed_x:seed_x + patch_size]
            elif (k != n_y) & (m == n_x):                 # m 성분이 끝부분 경우
                patch = img[seed_y:seed_y + patch_size, (img.size()[1] - patch_size):img.size()[1]]
            elif (k == n_y) & (m != n_x):                 # k 성분이 끝부분 경우
                patch = img[(img.size()[0] - patch_size):img.size()[0], seed_x:seed_x + patch_size]
            else:                                                                                       # k, m 성분이 끝부분 경우
                patch = img[(img.size()[0] - patch_size):img.size()[0], (img.size()[1] - patch_size):img.size()[1]]

            patch = patch.unsqueeze(0).unsqueeze(0)                                                     # (batch, channel, W, H)
            input = patch.to(model.device)
            out = model.genA2B(input)
            result_patch = out.cpu().squeeze(0).squeeze(0).detach().numpy()  # grad 성분이 없어야 함. 그러므로 detach

            if (k != n_y) & (m != n_x):
                result_patch_removed = result_patch[edgeRemove_pix:-edgeRemove_pix, edgeRemove_pix:-edgeRemove_pix]
                result_np[seed_y:seed_y + patch_size_removed, seed_x:seed_x + patch_size_removed] += result_patch_removed
                denom_np[seed_y:seed_y + patch_size_removed, seed_x:seed_x + patch_size_removed] += denom_patch
            elif (k != n_y) & (m == n_x):
                result_patch_removed = result_patch[edgeRemove_pix:-edgeRemove_pix, edgeRemove_pix:-edgeRemove_pix]
                result_np[seed_y:seed_y + patch_size_removed, (w - patch_size_removed):w] += result_patch_removed
                denom_np[seed_y:seed_y + patch_size_removed, (w - patch_size_removed):w] += denom_patch
            elif (k == n_y) & (m != n_x):
                result_patch_removed = result_patch[edgeRemove_pix:-edgeRemove_pix, edgeRemove_pix:-edgeRemove_pix]
                result_np[(h - patch_size_removed):h, seed_x:seed_x + patch_size_removed] += result_patch_removed
                denom_np[(h - patch_size_removed):h, seed_x:seed_x + patch_size_removed] += denom_patch
            else:
                result_patch_removed = result_patch[edgeRemove_pix:-edgeRemove_pix, edgeRemove_pix:-edgeRemove_pix]
                result_np[(h - patch_size_removed):h, (w - patch_size_removed):w] += result_patch_removed
                denom_np[(h - patch_size_removed):h, (w - patch_size_removed):w] += denom_patch

    return result_np/denom_np

def wt_decomp(input_np, wt_type, level, pad_mode):
    arr = pywt.wavedec2(input_np, wavelet=wt_type, mode=pad_mode, level=level)
    arr[0] = np.zeros_like(arr[0], dtype=np.float32)
    input_h = pywt.waverec2(arr, wavelet=wt_type, mode=pad_mode).astype('float32')
    input_l = input_np-input_h
    return input_l, input_h