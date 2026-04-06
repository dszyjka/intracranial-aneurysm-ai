import matplotlib.pyplot as plt


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

def draw_img(img):
    fig, ax = plt.subplots(2, 5, figsize=(30, 5))

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