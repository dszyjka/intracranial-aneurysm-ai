import torch
import os
import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset
from core.preprocessing import process_image_for_segmentation


class SegmentationDataset(Dataset):

    def __init__(self, path, series_lst, train_df, z_score_params, patch_size, transform=None):
        self.path = path
        self.data = train_df.loc[train_df['SeriesInstanceUID'].isin(series_lst)].reset_index(drop=True)
        self.z_score_params = z_score_params
        self.patch_size = patch_size
        self.transform = transform

    def _get_patch(self, img_arr, seg_arr, vessel_in_centre_prob=0.7):
        if np.random.rand() >= vessel_in_centre_prob:
            center = np.array(np.random.randint(0, seg_arr.shape, size=(3,)))
        else:
            vessels_indices = np.argwhere(seg_arr > 0)
        
            if len(vessels_indices) > 0:
                center = vessels_indices[np.random.randint(0, len(vessels_indices))]
            else:
                center = np.array([d // 2 for d in seg_arr.shape])

        slices = []
        for d in range(3):
            start = max(0, center[d] - (self.patch_size[d] // 2))
            end = max(0, start + self.patch_size[d])

            if end > seg_arr.shape[d]:
                end = seg_arr.shape[d]
                start = max(0, seg_arr.shape[d] - self.patch_size[d])

            slices.append(slice(start, end))

        img_arr = img_arr[tuple(slices)]
        seg_arr = seg_arr[tuple(slices)]

        if img_arr.shape != self.patch_size:
            pad_width = [(0, p - i) for p, i in zip(self.patch_size, img_arr.shape)]
            img_arr = np.pad(img_arr, pad_width, mode='constant', constant_values=0.0)
            seg_arr = np.pad(seg_arr, pad_width, mode='constant', constant_values=0.0)

        return img_arr, seg_arr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        modality_params = {}

        if row['Modality'] == 'CTA':
            modality_params['modality'] = 'CTA'
            modality_params['mean_val'] = self.z_score_params['cta_mean']
            modality_params['std_val'] = self.z_score_params['cta_std']
        else:
            modality_params['modality'] = 'MRI'
            modality_params['mean_val'] = self.z_score_params['mri_mean']
            modality_params['std_val'] = self.z_score_params['mri_std']

        img_path = os.path.join(self.path, f'{row['SeriesInstanceUID']}.nii')
        seg_path = os.path.join(self.path, f'{row['SeriesInstanceUID']}_cowseg.nii')

        img = sitk.ReadImage(img_path)
        seg = sitk.ReadImage(seg_path)

        img_arr, seg_arr = process_image_for_segmentation(img, seg, modality_params)

        patch_img, patch_seg = self._get_patch(img_arr, seg_arr)

        if self.transform is not None:
            patch_img = self.transform(patch_img)
            patch_seg = self.transform(patch_seg)

        patch_img = torch.from_numpy(patch_img).float().unsqueeze(0)
        patch_seg = torch.from_numpy(patch_seg).long()

        return patch_img, patch_seg