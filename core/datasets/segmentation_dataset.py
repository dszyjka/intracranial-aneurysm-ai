import torch
import os
import numpy as np
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):

    def __init__(self, files_dir, series, patch_size, transform=None):
        self.files_dir = files_dir
        self.series = series
        self.patch_size = patch_size
        self.transform = transform
        self.vessel_indices = self._precompute_vessel_indices()

    def _precompute_vessel_indices(self):
        vessel_indices = {}

        for ser in self.series:
            seg_path = os.path.join(self.files_dir, 'segmentations', f'{ser}.npy')
            seg = np.load(seg_path)
            vessel_indices[ser] = np.argwhere(seg > 0)

        return vessel_indices

    def _get_patch(self, img_arr, seg_arr, ser, vessel_in_centre_prob=0.7):
        if torch.rand(1).item() >= vessel_in_centre_prob:
            center = np.array([torch.randint(0, dim, (1,)).item() for dim in seg_arr.shape])
        else:
            vessels = self.vessel_indices[ser]
        
            if len(vessels) > 0:
                center = vessels[torch.randint(0, len(vessels), (1,)).item()]
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
            pad_width = [((p - i) // 2, (p - i) - ((p - i) // 2))
                         for p, i in zip(self.patch_size, img_arr.shape)]
            
            img_arr = np.pad(img_arr, pad_width, mode='constant', constant_values=0.0)
            seg_arr = np.pad(seg_arr, pad_width, mode='constant', constant_values=0.0)

        return img_arr, seg_arr

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        ser = self.series[idx]

        img_path = os.path.join(self.files_dir, 'images', f'{ser}.npy')
        seg_path = os.path.join(self.files_dir, 'segmentations', f'{ser}.npy')

        img = np.load(img_path, mmap_mode='r')
        seg = np.load(seg_path, mmap_mode='r')

        img_patch, seg_patch = self._get_patch(img, seg, ser)

        img_tensor = torch.from_numpy(img_patch.copy()).float().unsqueeze(0)
        seg_tensor = torch.from_numpy(seg_patch.copy()).long()

        if self.transform is not None:
            img_tensor, seg_tensor = self.transform(img_tensor, seg_tensor)

        return img_tensor, seg_tensor