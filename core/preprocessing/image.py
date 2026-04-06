import SimpleITK as sitk
import numpy as np
import torch.nn.functional as F
import torchio as tio
import torch
from .normalization import z_score


def crop_to_nonzero(img, seg=None):
    img_arr = sitk.GetArrayFromImage(img)

    nonzero_voxels = np.argwhere(img_arr > 0)

    assert nonzero_voxels.size > 0, 'Empty image'

    min_ch, min_h, min_w = np.maximum(nonzero_voxels.min(axis=0) - 2, 0)
    max_ch, max_h, max_w = nonzero_voxels.max(axis=0) + 3

    if seg is not None:
        return (img[min_w : max_w, min_h : max_h, min_ch : max_ch],
            seg[min_w : max_w, min_h : max_h, min_ch : max_ch])
    
    return img[min_w : max_w, min_h : max_h, min_ch : max_ch]

def process_data_for_segmentation(img, seg, z_score_params):
    img = sitk.DICOMOrient(img)
    seg = sitk.DICOMOrient(seg)

    img, seg = crop_to_nonzero(img, seg)
    
    img = resample_img(img)
    seg = resample_img(seg, True)

    img_arr = sitk.GetArrayFromImage(img)
    seg_arr = sitk.GetArrayFromImage(seg)

    z_score_params['img'] = img_arr
    img_arr = z_score(params=z_score_params)

    return img_arr, seg_arr

def preprocess_img(img, target_height=256, target_width=256):
    img = np.transpose(img, (1, 2, 0))
    img = crop_pad(img, 2, 0, 1, target_shape=(target_height, target_width, img.shape[2]))
    return img

def crop_pad(img, ch_id, h_id, w_id, target_shape=(10, 224, 224)):
    if img.shape == target_shape:
        return img
    
    y_pad = max(0, target_shape[h_id] - img.shape[h_id])
    x_pad = max(0, target_shape[w_id] - img.shape[w_id])

    if not (x_pad and y_pad):
        x_left = max(0, img.shape[w_id] - target_shape[w_id]) // 2
        x_right = x_left + target_shape[w_id]
        y_top = max(0, img.shape[h_id] - target_shape[h_id]) // 2
        y_bottom = y_top + target_shape[h_id]

        img = (img[y_top : y_bottom, x_left : x_right , :] if ch_id
                else img[: , y_top : y_bottom, x_left : x_right])
    
    y_before = y_pad // 2
    y_after = y_pad - y_before 
    x_before = x_pad // 2
    x_after = x_pad - x_before
    
    img = pad(img, x_before, x_after, y_before, y_after, ch_id)
    
    return img

def pad(img, x_before, x_after, y_before, y_after, ch_id):
    if isinstance(img, np.ndarray):
        img = (np.pad(img, ((y_before, y_after), (x_before, x_after), (0, 0)),
                      mode='constant', constant_values=0.0)
                if ch_id else
                np.pad(img, ((0, 0), (y_before, y_after), (x_before, x_after)),
                       mode='constant', constant_values=0.0))
        
    elif isinstance(img, torch.Tensor):
        img = (F.pad(img, (0, 0, x_before, x_after, y_before, y_after), value=0.0)
               if ch_id else F.pad(img, (x_before, x_after, y_before, y_after), value=0.0))
    
    else:
        raise TypeError('img argument must be torch.Tensor or numpy.ndarray')
    
    return img

def resample_img(img, is_mask=False):
    if is_mask:
        resample = tio.Resample((1, 1, 1), label_interpolation='nearest', image_interpolation='nearest')
    else:
        resample = tio.Resample((1, 1, 1))
    return resample(img)

def crop_or_pad_img(img, h=256, w=256, ch=5):
    crop_pad = tio.CropOrPad((h, w, ch))
    return crop_pad(img)
