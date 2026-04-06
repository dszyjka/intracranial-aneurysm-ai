import numpy as np


def normalize_data(train_data, test_data):
    min_val = np.min(train_data)
    max_val = np.max(train_data)
    return (train_data - min_val) / (max_val - min_val), (test_data - min_val) / (max_val - min_val)

def z_score(train_data, test_data, params=None):
    if params is None:
        mean_val = np.mean(train_data)
        std_val = np.std(train_data)
        return (train_data - mean_val) / std_val, (test_data - mean_val) / std_val
    
    return (params['img'] - params['mean_val']) / params['std_val']

def image_clipping(img_arr, modality):
    if modality == 'CTA':
        vmin = -100
        vmax = 600
        return np.clip(img_arr, vmin, vmax)

    p5 = np.percentile(img_arr, 5)
    p95 = np.percentile(img_arr, 95)

    vmin = p5 * 2
    vmax = min(p95 * 2, img_arr.max())

    return np.clip(img_arr, vmin, vmax)