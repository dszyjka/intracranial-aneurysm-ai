import os
import json
import numpy as np
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from core.preprocessing import process_image_for_segmentation


def process_and_save(read_path, save_dir, series, z_score_params, data_df):
    for ser in tqdm(series):
        row = data_df.loc[ser]

        is_cta = row['Modality'] == 'CTA'
        img_params = {
            'modality': row['Modality'],
            'mean_val': z_score_params['cta_mean'] if is_cta else z_score_params['mri_mean'],
            'std_val': z_score_params['cta_std'] if is_cta else z_score_params['mri_std']
        }

        img_path = os.path.join(read_path, f'{ser}.nii')
        seg_path = os.path.join(read_path, f'{ser}_cowseg.nii')

        img = sitk.ReadImage(img_path)
        seg = sitk.ReadImage(seg_path)

        img, seg = process_image_for_segmentation(img, seg, img_params)

        save(save_dir, img, seg, ser)

def create_catalogs(base_save_dir):
    segmentation_dir = os.path.join(base_save_dir, 'segmentation data')
    os.makedirs(segmentation_dir, exist_ok=True)

    train_dir = os.path.join(segmentation_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    
    test_dir = os.path.join(segmentation_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)

    for dir in (train_dir, test_dir):
        imgs_dir = os.path.join(dir, 'images')
        os.makedirs(imgs_dir, exist_ok=True)
        seg_dir = os.path.join(dir, 'segmentations')
        os.makedirs(seg_dir, exist_ok=True)

    return train_dir, test_dir

def save(save_dir, img, seg, ser):
    np.save(os.path.join(save_dir, 'images', f'{ser}.npy'), img)
    np.save(os.path.join(save_dir, 'segmentations', f'{ser}.npy'), seg)

def main():
    stats_path = Path('configs/segmentation/dataset_stats.json')
    splits_path = Path('configs/segmentation/splits.json')

    with open(stats_path) as file:
        stats = json.load(file)

    with open(splits_path, 'r') as file:
        splits = json.load(file)
        train_series = splits.get("train_series", [])
        test_series = splits.get("test_series", [])

    base_save_dir = 'RSNA_my_folder'
    data_dir = 'RSNA_data'
    train_dir, test_dir = create_catalogs(base_save_dir)

    data_df = pd.read_csv(os.path.join(data_dir, 'train.csv')).set_index('SeriesInstanceUID')

    process_and_save(data_dir, train_dir, train_series, stats, data_df)
    process_and_save(data_dir, test_dir, test_series, stats, data_df)

if __name__ == '__main__':
    main()