import torch
import numpy as np


def show_tsboard(writer, np_array, epoch, title):
    arr = np_array.copy()
    ts_array = torch.from_numpy(arr).unsqueeze(0)
    ts_array[ts_array < 0] = 0
    writer.add_image(title, ts_array/np.max(np_array), epoch)

def show_r_tsboard(writer, np_array, epoch, title, range):
    arr = np_array.copy()
    ts_array = torch.from_numpy(arr).unsqueeze(0)
    ts_array[ts_array < -range] = -range
    ts_array[ts_array > range] = range
    ts_array = ts_array*(1/(2*range))
    ts_array = ts_array + 0.5
    writer.add_image(title, ts_array, epoch)


