from torch.utils.data import Dataset
from core.preprocessing import crop_pad


class MedicalDataset(Dataset):

    def __init__(
            self, images, labels, transform=None, target_shape=(10, 224, 224), channel_id=0, h_id=1, w_id=2):
        
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_shape = target_shape
        self.ch_id = channel_id
        self.h_id = h_id
        self.w_id = w_id

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img = crop_pad(img, self.ch_id, self.h_id, self.w_id, self.target_shape)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label