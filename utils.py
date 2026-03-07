from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import torch
import numpy as np


class MedicalDataset(Dataset):

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
def evaluate_model(model, device, loader):
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).squeeze()

            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)

            y_pred.extend(pred.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.Blues)

    plt.show()

def draw_img(img):
    fix, ax = plt.subplots(2, 5, figsize=(30, 5))

    ax[0, 0].imshow(img[:, :, 0], cmap='gray')
    ax[0, 1].imshow(img[:, :, 1], cmap='gray')
    ax[0, 2].imshow(img[:, :, 2], cmap='gray')
    ax[0, 3].imshow(img[:, :, 3], cmap='gray')
    ax[0, 4].imshow(img[:, :, 4], cmap='gray')

    ax[1, 0].imshow(img[:, :, 5], cmap='gray')
    ax[1, 1].imshow(img[:, :, 6], cmap='gray')
    ax[1, 2].imshow(img[:, :, 7], cmap='gray')
    ax[1, 3].imshow(img[:, :, 8], cmap='gray')
    ax[1, 4].imshow(img[:, :, 9], cmap='gray')

    plt.show()

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

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
])

label2organ = {
  1: "Other Posterior Circulation",
  2: "Basilar Tip",
  3: "Right Posterior Communicating Artery",
  4: "Left Posterior Communicating Artery",
  5: "Right Infraclinoid Internal Carotid Artery",
  6: "Left Infraclinoid Internal Carotid Artery",
  7: "Right Supraclinoid Internal Carotid Artery",
  8: "Left Supraclinoid Internal Carotid Artery",
  9: "Right Middle Cerebral Artery",
  10: "Left Middle Cerebral Artery",
  11: "Right Anterior Cerebral Artery",
  12: "Left Anterior Cerebral Artery",
  13: "Anterior Communicating Artery"
}