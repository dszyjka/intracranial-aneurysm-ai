import torch
import numpy as np


def train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs, device):
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
                dataloader = val_loader
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