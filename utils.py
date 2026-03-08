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
    
def train_model(model, optimizer, criterion, train_loader, test_loader, num_epochs, device):
    acc_train_hist, loss_train_hist, acc_val_hist, loss_val_hist = [], [], [], []

    lowest_loss = np.inf
    best_model_weights = {k : v.cpu() for k, v in model.state_dict().items()}

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch+1} / {num_epochs}')

        for phase in ('train', 'val'):
            if phase == 'train':
                dataloader = train_loader
                acc_hist = acc_train_hist
                loss_hist = loss_train_hist
                model.train()
            else:
                dataloader = test_loader
                acc_hist = acc_val_hist
                loss_hist = loss_val_hist
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0
            total_samples = 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.view(-1))

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * labels.size(0)
                running_corrects += (pred == labels.squeeze()).sum().item()
                total_samples += labels.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples

            loss_hist.append(epoch_loss)
            acc_hist.append(epoch_acc)

            print(f'Loss: {epoch_loss}, accuracy: {epoch_acc}')

            if phase == 'val' and epoch_loss < lowest_loss:
                print(f'New lowest loss. Epoch {epoch+1}')
                lowest_loss = epoch_loss
                best_model_weights = {k : v.cpu() for k, v in model.state_dict().items()}

    history = {'train_acc' : acc_train_hist,
               'train_loss' : loss_train_hist,
               'val_acc' : acc_val_hist,
               'val_loss' : loss_val_hist}

    return best_model_weights, history

def draw_loss_and_acc_history(hist):
    train_acc = hist['train_acc']
    train_loss = hist['train_loss']
    val_acc = hist['val_acc']
    val_loss = hist['val_loss']
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ax[0].plot(train_acc, color='red', label='train_acc')
    ax[0].plot(val_acc, color='blue', label='val_acc')

    ax[1].plot(train_loss, color='red', label='train_loss')
    ax[1].plot(val_loss, color='blue', label='val_loss')

    ax[0].set_title('ACCURACY HISTORY')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    ax[1].set_title('LOSS HISTORY')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

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

def to_tensor(x_train, y_train, x_test, y_test, permute_order=(0, -1, 1, 2)):
    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()

    x_test_tensor = torch.from_numpy(x_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()

    x_train_tensor = x_train_tensor.permute(permute_order)
    x_test_tensor = x_test_tensor.permute(permute_order)

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor

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