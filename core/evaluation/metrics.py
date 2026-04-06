from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import torch


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