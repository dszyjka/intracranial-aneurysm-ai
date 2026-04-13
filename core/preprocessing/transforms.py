from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np


classification_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
])

class SegmentationTransform:
    def __call__(self, img, seg):
        if np.random.random() < 0.5:
            img = F.hflip(img)
            seg = F.hflip(seg)

        angle = np.random.uniform(-20, 20)
        img = F.rotate(img, angle, interpolation=F.InterpolationMode.BILINEAR)
        seg = F.rotate(seg, angle, interpolation=F.InterpolationMode.NEAREST)

        return img, seg