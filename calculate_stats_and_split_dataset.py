import os
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from core.preprocessing import crop_to_nonzero, resample_img
from core.constants import seg_with_loc


def compute_data_stats(train_series, folder, train_df):
    cta_means, cta_stds, mri_means, mri_stds = [], [], [], []

    for ser in train_series:
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

    return (float(np.mean(cta_means)),
            float(np.mean(cta_stds)),
            float(np.mean(mri_means)),
            float(np.mean(mri_stds)))

def split_series_for_datasets(seg_with_loc, train_size=0.8):
    train_ser, test_ser = train_test_split(seg_with_loc, train_size=train_size,random_state=42)
    return train_ser, test_ser

def main():
    base_data_dir = 'C:\\RSNA_data'
    segmentations_path = os.path.join(base_data_dir, 'segmentations')
    train_df = pd.read_csv(os.path.join(base_data_dir, 'train.csv'))

    seg_configs_path = 'configs/segmentation'
    os.makedirs(seg_configs_path, exist_ok=True)

    split_json_path = os.path.join(seg_configs_path, 'splits.json')
    stats_json_path = os.path.join(seg_configs_path, 'dataset_stats.json')

    train_ser, test_ser = split_series_for_datasets(seg_with_loc)
        
    data_split = {'train_series' : list(train_ser),
                'test_series' : list(test_ser)}

    with open(split_json_path, 'w') as file:
        json.dump(data_split, file, indent=4)

    cta_mean, cta_std, mri_mean, mri_std = compute_data_stats(train_ser, segmentations_path, train_df)

    stats = {'cta_mean' : cta_mean,
            'cta_std' : cta_std,
            'mri_mean' : mri_mean,
            'mri_std' : mri_std}

    with open(stats_json_path, 'w') as file:
        json.dump(stats, file, indent=4)

if __name__ == '__main__':
    main()