import numpy as np
import SimpleITK as sitk
import torch
import json
import os
from ..preprocessing import crop_to_nonzero, resample_img


def get_physical_point_and_location(df_row):
    xy = df_row['coordinates']
    loc = df_row['location']

    x = json.loads(xy.replace("\'", "\""))['x']
    y = json.loads(xy.replace("\'", "\""))['y']

    return x, y, loc

def get_idx():
    i = 0
    while True:
        yield i
        i += 1

def repeat_samples(x, y, repeats):
    aneurysm_mask = np.where(y > 0)[0]
    healthy_mask = np.where(y == 0)[0]

    x_aneurysm = x[aneurysm_mask]
    y_aneurysm = y[aneurysm_mask]
    x_healthy = x[healthy_mask]
    y_healthy = y[healthy_mask]

    x_aneurysm = np.repeat(x_aneurysm, repeats, axis=0)
    y_aneurysm = np.repeat(y_aneurysm, repeats, axis=0)

    new_x = np.concatenate([x_aneurysm, x_healthy], axis=0)
    new_y = np.concatenate([y_aneurysm, y_healthy], axis=0)

    return new_x, new_y

def to_tensor(x_train, y_train, x_test, y_test, permute_order=(0, -1, 1, 2)):
    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()

    x_test_tensor = torch.from_numpy(x_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()

    x_train_tensor = x_train_tensor.permute(permute_order)
    x_test_tensor = x_test_tensor.permute(permute_order)

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor

def compute_data_stats(train_series_lst, folder, train_df):
    cta_means, cta_stds, mri_means, mri_stds = [], [], [], []

    for ser in train_series_lst:
        if train_df.loc[train_df['SeriesInstanceUID'] == ser]['Modality'].iloc[0] == 'CTA':
            means = cta_means
            stds = cta_stds
        else:
            means = mri_means
            stds = mri_stds

        path = os.path.join(folder, f'{ser}.nii')
        img = sitk.ReadImage(path)
        img = crop_to_nonzero(img)
        img = resample_img(img)
        img = sitk.GetArrayFromImage(img)

        means.append(img.mean())
        stds.append(img.std())

    return np.mean(cta_means), np.means(cta_stds), np.mean(mri_means), np.means(mri_stds)