import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import glob
import os
import pywt
import random
import math



def get_path(data_root, file_list, phase):
    domainA_dir = os.path.join(data_root, phase, file_list[0])
    domainB_dir = os.path.join(data_root, phase, file_list[1])

    npy_A_path = glob.glob(domainA_dir + '/*npy')
    npy_B_path = glob.glob(domainB_dir + '/*npy')

    return npy_A_path, npy_B_path


class AAPM_data(Dataset):

    def __init__(self, opt, phase):
        self.margin = opt.margin
        self.crop_size = opt.patch_size
        self.crop_mode = opt.crop_mode

        ## Dataset
        self.data_root = opt.data_root
        self.transform = transforms.ToTensor()

        ## file path
        self.phase = phase
        file_list = ['quarter_dose', 'full_dose']
        self.files_A, self.files_B = get_path(self.data_root, file_list, phase)

        ## wavelet
        self.is_wt = opt.is_wt
        if self.is_wt:
            self.wt_type = opt.wt_type
            self.wt_level = opt.wt_level
            self.wt_pad = opt.wt_pad_mode
            self.padding = tuple()


    def __getitem__(self, index):

        index_A = index % len(self.files_A)
        index_B = index % len(self.files_B)
        A_path = self.files_A[index_A]
        B_path = self.files_B[index_B]

        A_img = torch.from_numpy(np.load(A_path))
        B_img = torch.from_numpy(np.load(B_path))


        A_hu = (1000/0.0192)*(A_img-0.0192)
        A_n = (A_hu + 1000) / 2000
        B_hu = (1000 / 0.0192) * (B_img - 0.0192)
        B_n = (B_hu + 1000) / 2000

        ## wavelet
        if self.is_wt:
            A_l, A_out = self.wt_decomp(A_n, self.wt_type, self.wt_level, self.wt_pad)
            B_l, B_out = self.wt_decomp(B_n, self.wt_type, self.wt_level, self.wt_pad)
        else:
            A_out = A_n.numpy()
            B_out = B_n.numpy()
            A_l = torch.from_numpy(A_out).unsqueeze(0).type(torch.float32)
            B_l = torch.from_numpy(B_out).unsqueeze(0).type(torch.float32)

        if self.phase=='train':
            A_out = self.crop(A_out, self.crop_size, margin=self.margin, mode=self.crop_mode)
            B_out = self.crop(B_out, self.crop_size, margin=self.margin, mode=self.crop_mode)

        A_out = torch.from_numpy(A_out).unsqueeze(0).type(torch.float32)
        B_out = torch.from_numpy(B_out).unsqueeze(0).type(torch.float32)

        return {'A': A_out, 'B': B_out, 'A_l': A_l, 'B_l': B_l, 'A_path': A_path, 'B_path': B_path}


    def wt_decomp(self, array, wt_type, wt_level, wt_pad):
        ## padding
        array_pad = self.apply_wave_padding(array)

        arr = pywt.wavedec2(array_pad, wavelet=wt_type, mode=wt_pad, level=wt_level)
        arr[0] = np.zeros(arr[0].shape, dtype=np.float32)
        arr_h = pywt.waverec2(arr, wavelet=wt_type, mode=wt_pad).astype(np.float32)

        ## unpadding
        (t, d), (l, r) = self.padding
        arr_h = arr_h[t:-d, l:-r]

        arr_l = array - arr_h
        return arr_l, arr_h

    def apply_wave_padding(self, image: np.ndarray):
        wavelet = pywt.Wavelet(name='db3')  # (name=self.opt.wavelet)
        if wavelet.dec_len != wavelet.rec_len:
            raise NotImplementedError('Padding assumes decomposition and reconstruction to have the same filter length')
        assert image.ndim == 2, 'Image must be 2D.'
        filter_len = wavelet.dec_len
        level = self.wt_level  # self.opt.level
        h, w = image.shape

        # Extra length necessary to prevent artifacts due to separation of low and high frequencies.
        # Size must be divisible by (2^level) for no shifting artifacts to occur.
        # The final modulo ensures that divisible lengths add 0 instead of 2^level.
        hh = ((2 ** level) - h % (2 ** level)) % (2 ** level)
        ww = ((2 ** level) - w % (2 ** level)) % (2 ** level)

        # Extra length necessary to prevent artifacts from kernel going over the edge into padding region.
        # Padding size much be such that even the innermost decomposition is perfectly within the kernel.
        # I have found that the necessary padding is filter_len, not (filter_len-1). The former is also usually even.
        hh += filter_len * (2 ** level)
        ww += filter_len * (2 ** level)

        self.padding = ((hh // 2, hh - hh // 2), (ww // 2, ww - ww // 2))

        return np.pad(image, pad_width=self.padding, mode='symmetric')


    def crop(self, img, crop_size, margin=0, mode='rect'):  #mode : radial | rect
        img_h, img_w = (img.shape[0], img.shape[1])

        if mode == 'rect':
            seed_w = random.randrange(margin, img_w - crop_size - margin)
            seed_h = random.randrange(margin, img_h - crop_size - margin)

        elif mode == 'radial':
            radius = random.uniform(0, img_w / 2 - crop_size / 2 - margin)
            seed_ang = random.uniform(0, 2 * (math.pi))
            seed_w = int(img_w / 2 - radius * math.sin(seed_ang) - crop_size/2)
            seed_h = int(img_h / 2 + radius * math.cos(seed_ang) - crop_size/2)

        cropped_img = img[seed_h:seed_h + crop_size, seed_w:seed_w + crop_size]
        return cropped_img

    def __len__(self):
        return len(self.files_A)

    def name(self):
        return 'AAPM_data'