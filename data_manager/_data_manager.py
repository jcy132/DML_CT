import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from data_manager.AAPM import AAPM_data


def create_TrainDataset(opt):
    if opt.dataset_name == 'AAPM':
        return AAPM_data(opt, phase='train')
    else:
        raise NotImplementedError('Dataset not implemented')

def create_ValDataset(opt):
    if opt.dataset_name == 'AAPM':
        return AAPM_data(opt, phase='val')
    else:
        raise NotImplementedError('Dataset not implemented')

def create_TestDataset(opt):
    if opt.dataset_name == 'AAPM':
        return AAPM_data(opt, phase='test')
    else:
        raise NotImplementedError('Dataset not implemented')


def data_sampler(opt, subset_size, len_dataset):
    assert subset_size < len_dataset
    indices = list(range(subset_size))
    np.random.shuffle(indices)
    sampler = SubsetRandomSampler(indices)
    return sampler