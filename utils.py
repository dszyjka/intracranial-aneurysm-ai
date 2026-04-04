from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torchio as tio
import SimpleITK as sitk
import torch
import json
import os


def crop_pad(img, ch_id, h_id, w_id, target_shape=(10, 224, 224)):
    if img.shape == target_shape:
        return img
    
    y_pad = max(0, target_shape[h_id] - img.shape[h_id])
    x_pad = max(0, target_shape[w_id] - img.shape[w_id])

    if not (x_pad and y_pad):
        x_left = max(0, img.shape[w_id] - target_shape[w_id]) // 2
        x_right = x_left + target_shape[w_id]
        y_top = max(0, img.shape[h_id] - target_shape[h_id]) // 2
        y_bottom = y_top + target_shape[h_id]

        img = (img[y_top : y_bottom, x_left : x_right , :] if ch_id
                else img[: , y_top : y_bottom, x_left : x_right])
    
    y_before = y_pad // 2
    y_after = y_pad - y_before 
    x_before = x_pad // 2
    x_after = x_pad - x_before
    
    img = pad(img, x_before, x_after, y_before, y_after, ch_id)
    
    return img

def pad(img, x_before, x_after, y_before, y_after, ch_id):
    if isinstance(img, np.ndarray):
        img = (np.pad(img, ((y_before, y_after), (x_before, x_after), (0, 0)),
                      mode='constant', constant_values=0.0)
                if ch_id else
                np.pad(img, ((0, 0), (y_before, y_after), (x_before, x_after)),
                       mode='constant', constant_values=0.0))
        
    elif isinstance(img, torch.Tensor):
        img = (F.pad(img, (0, 0, x_before, x_after, y_before, y_after), value=0.0)
               if ch_id else F.pad(img, (x_before, x_after, y_before, y_after), value=0.0))
    
    else:
        raise TypeError('img argument must be torch.Tensor or numpy.ndarray')
    
    return img

def resample_img(img, is_mask=False):
    if is_mask:
        resample = tio.Resample((1, 1, 1), label_interpolation='nearest', image_interpolation='nearest')
    else:
        resample = tio.Resample((1, 1, 1))
    return resample(img)

def crop_or_pad_img(img, h=256, w=256, ch=5):
    crop_pad = tio.CropOrPad((h, w, ch))
    return crop_pad(img)

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

def get_physical_point_and_location(df_row):
    xy = df_row['coordinates']
    loc = df_row['location']

    x = json.loads(xy.replace("\'", "\""))['x']
    y = json.loads(xy.replace("\'", "\""))['y']

    return x, y, loc

def preprocess_img(img, target_height=256, target_width=256):
    img = np.transpose(img, (1, 2, 0))
    img = crop_pad(img, 2, 0, 1, target_shape=(target_height, target_width, img.shape[2]))
    return img

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

def get_idx():
    i = 0
    while True:
        yield i
        i += 1

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

def crop_to_nonzero(img, seg=None):
    img_arr = sitk.GetArrayFromImage(img)

    nonzero_voxels = np.argwhere(img_arr > 0)

    assert nonzero_voxels.size > 0, 'Empty image'

    min_ch, min_h, min_w = np.maximum(nonzero_voxels.min(axis=0) - 2, 0)
    max_ch, max_h, max_w = nonzero_voxels.max(axis=0) + 3

    if seg is not None:
        return (img[min_w : max_w, min_h : max_h, min_ch : max_ch],
            seg[min_w : max_w, min_h : max_h, min_ch : max_ch])
    
    return img[min_w : max_w, min_h : max_h, min_ch : max_ch]

def process_data_for_segmentation(img, seg, z_score_params):
    img = sitk.DICOMOrient(img)
    seg = sitk.DICOMOrient(seg)

    img, seg = crop_to_nonzero(img, seg)
    
    img = resample_img(img)
    seg = resample_img(seg, True)

    img_arr = sitk.GetArrayFromImage(img)
    seg_arr = sitk.GetArrayFromImage(seg)

    z_score_params['img'] = img_arr
    img_arr = z_score(params=z_score_params)

    return img_arr, seg_arr

def compute_data_stats(train_series_lst, folder, train_df):
    cta_means, cta_stds, mri_means, mri_stds = [], [], [], []

    for ser in train_series_lst:
        if train_df.loc[train_df['SeriesInstanceUID'] == ser]['Modality'].iloc[0] == 'CTA':
            means = cta_means
            stds = cta_stds
        else:
            means = mri_means
            stds = mri_stds

        path = os.path.join(folder, f'{ser}.nii')
        img = sitk.ReadImage(path)
        img = crop_to_nonzero(img)
        img = resample_img(img)
        img = sitk.GetArrayFromImage(img)

        means.append(img.mean())
        stds.append(img.std())

    return np.mean(cta_means), np.means(cta_stds), np.mean(mri_means), np.means(mri_stds)

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
    
class SegmentationDataset(Dataset):

    def __init__(self, path, series_lst, train_df, z_score_params):
        self.path = path
        self.data = train_df.loc[train_df['SeriesInstanceUID'].isin(series_lst)].reset_index(drop=True)
        self.z_score_params = z_score_params

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        z_score_params = {}

        if row['Modality'] == 'CTA':
            z_score_params['mean_val'] = self.z_score_params['cta_mean']
            z_score_params['std__val'] = self.z_score_params['cta_std']
        else:
            z_score_params['mean_val'] = self.z_score_params['mri_mean']
            z_score_params['std__val'] = self.z_score_params['mri_std']

        img_path = os.path.join(self.path, f'{row['SeriesInstanceUID']}.nii')
        seg_path = os.path.join(self.path, f'{row['SeriesInstanceUID']}_cowseg.nii')

        img = sitk.ReadImage(img_path)
        #img = sitk.DICOMOrient(img)
        #img_arr = sitk.GetArrayFromImage(img)

        seg = sitk.ReadImage(seg_path)
        #seg = sitk.DICOMOrient(seg)
        #seg_arr = sitk.GetArrayFromImage(seg)

        img, seg = process_data_for_segmentation(img, seg, z_score_params)

        # place for rest of the __getitem__ method


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

organ2label = {
  "Other Posterior Circulation" : 1,
  "Basilar Tip" : 2,
  "Right Posterior Communicating Artery" : 3,
  "Left Posterior Communicating Artery" : 4,
  "Right Infraclinoid Internal Carotid Artery" : 5,
  "Left Infraclinoid Internal Carotid Artery" : 6,
  "Right Supraclinoid Internal Carotid Artery" : 7,
  "Left Supraclinoid Internal Carotid Artery" : 8,
  "Right Middle Cerebral Artery" : 9,
  "Left Middle Cerebral Artery" : 10,
  "Right Anterior Cerebral Artery" : 11,
  "Left Anterior Cerebral Artery" : 12,
  "Anterior Communicating Artery" : 13
}

seg_with_loc = ['1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381',
 '1.2.826.0.1.3680043.8.498.10076056930521523789588901704956188485',
 '1.2.826.0.1.3680043.8.498.10188636688783982623025997809119805350',
 '1.2.826.0.1.3680043.8.498.10410600166004340343973545138447283460',
 '1.2.826.0.1.3680043.8.498.10540586847553109495238524904638776495',
 '1.2.826.0.1.3680043.8.498.10929608782694347957516071062422315982',
 '1.2.826.0.1.3680043.8.498.10935907012185032169927418164924236382',
 '1.2.826.0.1.3680043.8.498.11140496970152788589837488009637704168',
 '1.2.826.0.1.3680043.8.498.11163718560814217911019576488539324434',
 '1.2.826.0.1.3680043.8.498.11422928060228360802778018026859204182',
 '1.2.826.0.1.3680043.8.498.11447542941959800581541313722844637822',
 '1.2.826.0.1.3680043.8.498.11504459395565711149380261095223705023',
 '1.2.826.0.1.3680043.8.498.11557464859397815362951522785245632020',
 '1.2.826.0.1.3680043.8.498.11624217734793256238140178687655335066',
 '1.2.826.0.1.3680043.8.498.11639720015527164474926997755882681707',
 '1.2.826.0.1.3680043.8.498.11641438607169452758239778414614826230',
 '1.2.826.0.1.3680043.8.498.11821446229980500432989393232863242415',
 '1.2.826.0.1.3680043.8.498.11924949819899884502738782576851659426',
 '1.2.826.0.1.3680043.8.498.11938739392606296532297884225608408867',
 '1.2.826.0.1.3680043.8.498.11999987145696510072091906561590137848',
 '1.2.826.0.1.3680043.8.498.12132622846836853200891705613461466627',
 '1.2.826.0.1.3680043.8.498.12180351938456969219537687190067731477',
 '1.2.826.0.1.3680043.8.498.12226380705607315060235896835122737788',
 '1.2.826.0.1.3680043.8.498.12271269630687930751200307891697907423',
 '1.2.826.0.1.3680043.8.498.12427930128533148989436011949311706348',
 '1.2.826.0.1.3680043.8.498.12709802490782896897031103447163443069',
 '1.2.826.0.1.3680043.8.498.12773309706735630359315214846273921394',
 '1.2.826.0.1.3680043.8.498.12792960392435514526913217158720555996',
 '1.2.826.0.1.3680043.8.498.12812390336793304037901571645929430100',
 '1.2.826.0.1.3680043.8.498.12873050136415197430227722045995986358',
 '1.2.826.0.1.3680043.8.498.12888459003897616398890411591973176636',
 '1.2.826.0.1.3680043.8.498.12896910506681881306246412668919668702',
 '1.2.826.0.1.3680043.8.498.12898332622076283462996059479076432725',
 '1.2.826.0.1.3680043.8.498.12904246053955178641505906243733756576',
 '1.2.826.0.1.3680043.8.498.12914952223659958493995413641114579279',
 '1.2.826.0.1.3680043.8.498.13128656559176299272467358793386537400',
 '1.2.826.0.1.3680043.8.498.13359737970612926494907045108541390310',
 '1.2.826.0.1.3680043.8.498.13789305723712362238118274295587312089',
 '1.2.826.0.1.3680043.8.498.14375161350968928494386548917647435597',
 '1.2.826.0.1.3680043.8.498.15111820005882064793593034423469604305',
 '1.2.826.0.1.3680043.8.498.15412988336827906186857260013885503248',
 '1.2.826.0.1.3680043.8.498.15777485274723969278718374949878560903',
 '1.2.826.0.1.3680043.8.498.16386250344855221757144432829845114733',
 '1.2.826.0.1.3680043.8.498.16984390277144742906667579449023180512',
 '1.2.826.0.1.3680043.8.498.17415277997649872560329721717694101082',
 '1.2.826.0.1.3680043.8.498.19915189891686122627071348069843885714',
 '1.2.826.0.1.3680043.8.498.21004106426734635526381567602936015568',
 '1.2.826.0.1.3680043.8.498.23047023542526806696555440426928375679',
 '1.2.826.0.1.3680043.8.498.23055827202917388669053993133576833763',
 '1.2.826.0.1.3680043.8.498.24023896361071846724104915533800547445',
 '1.2.826.0.1.3680043.8.498.24587963869128721940158079207224095554',
 '1.2.826.0.1.3680043.8.498.24941924992372724575490063788348447936',
 '1.2.826.0.1.3680043.8.498.27693546360513068451517048347207987807',
 '1.2.826.0.1.3680043.8.498.27857528510177554953207997404329765760',
 '1.2.826.0.1.3680043.8.498.28722601444191262075880952461419085326',
 '1.2.826.0.1.3680043.8.498.31897325247898403027455884342546675049',
 '1.2.826.0.1.3680043.8.498.32250259987224176174516959348681094310',
 '1.2.826.0.1.3680043.8.498.34439485184360273751379923196589017042',
 '1.2.826.0.1.3680043.8.498.35327124657045713676192746001247576881',
 '1.2.826.0.1.3680043.8.498.35378146560080702211693278243609271022',
 '1.2.826.0.1.3680043.8.498.36205761227502095958293403225062705137',
 '1.2.826.0.1.3680043.8.498.36516744229109249667702200145077143886',
 '1.2.826.0.1.3680043.8.498.36563348911961346172279351080943665664',
 '1.2.826.0.1.3680043.8.498.36928611823925733133253145871406408988',
 '1.2.826.0.1.3680043.8.498.37086262716517957668471635372810376638',
 '1.2.826.0.1.3680043.8.498.38245669369430321272819874468980907728',
 '1.2.826.0.1.3680043.8.498.38904475631578710113273863766282479811',
 '1.2.826.0.1.3680043.8.498.39640919070091958876744231048011388614',
 '1.2.826.0.1.3680043.8.498.40402571428459178954472078378902050472',
 '1.2.826.0.1.3680043.8.498.42092450058597943280470345107435382425',
 '1.2.826.0.1.3680043.8.498.42672154202952548010999212369080894652',
 '1.2.826.0.1.3680043.8.498.42933230680553480084056393591634621848',
 '1.2.826.0.1.3680043.8.498.43495968397556043698567120038117641587',
 '1.2.826.0.1.3680043.8.498.43536331102142701793144520859521601945',
 '1.2.826.0.1.3680043.8.498.47622062519393262272120105951011625928',
 '1.2.826.0.1.3680043.8.498.47802313478131783077762931281303667601',
 '1.2.826.0.1.3680043.8.498.47887093599897399482447594752785316358',
 '1.2.826.0.1.3680043.8.498.49640345168968922611291772802640560828',
 '1.2.826.0.1.3680043.8.498.49718418682238683779854914910561017368',
 '1.2.826.0.1.3680043.8.498.50241233088534910114736887318508484246',
 '1.2.826.0.1.3680043.8.498.50268462808449401128173812870329002342',
 '1.2.826.0.1.3680043.8.498.50275403170194436966991630938339966596',
 '1.2.826.0.1.3680043.8.498.50369188120242587742908379292729868174',
 '1.2.826.0.1.3680043.8.498.52363954882447190271251269039176558430',
 '1.2.826.0.1.3680043.8.498.53947155422591684879953627516013605305',
 '1.2.826.0.1.3680043.8.498.54865110953409154322874363435644372368',
 '1.2.826.0.1.3680043.8.498.55051557363776453883164282380323354147',
 '1.2.826.0.1.3680043.8.498.55520651046049733868642268089599441721',
 '1.2.826.0.1.3680043.8.498.56109731607412273442907651635753012241',
 '1.2.826.0.1.3680043.8.498.56479623144539472445940519727300319231',
 '1.2.826.0.1.3680043.8.498.56867346585094457716984380929416039466',
 '1.2.826.0.1.3680043.8.498.58839417089022860359638460482101293080',
 '1.2.826.0.1.3680043.8.498.61152918475243358118286003299125054478',
 '1.2.826.0.1.3680043.8.498.65011208113835286935212080363533579671',
 '1.2.826.0.1.3680043.8.498.65654303333996310125136982540737772052',
 '1.2.826.0.1.3680043.8.498.66341469849558089736451534296312923277',
 '1.2.826.0.1.3680043.8.498.67256382079119118825371537284628604044',
 '1.2.826.0.1.3680043.8.498.68161752706586485995657009830735928975',
 '1.2.826.0.1.3680043.8.498.68276712082656957005274595949315894066',
 '1.2.826.0.1.3680043.8.498.68356160898101066850726244725552676010',
 '1.2.826.0.1.3680043.8.498.68654901185438820364160878605611510817',
 '1.2.826.0.1.3680043.8.498.68709340002397343932718258443293606585',
 '1.2.826.0.1.3680043.8.498.69401690945645968072368812538918487252',
 '1.2.826.0.1.3680043.8.498.69568746915553014138135720681936366640',
 '1.2.826.0.1.3680043.8.498.70243202242722756546202582478829903758',
 '1.2.826.0.1.3680043.8.498.71796601538792777580416179841706319140',
 '1.2.826.0.1.3680043.8.498.72679260079421518845786364620483278827',
 '1.2.826.0.1.3680043.8.498.73820261697830420042473892884688067574',
 '1.2.826.0.1.3680043.8.498.75016896260047968433534297207591136672',
 '1.2.826.0.1.3680043.8.498.75294325392457179365040684378207706807',
 '1.2.826.0.1.3680043.8.498.75798029534455454939797323020706657426',
 '1.2.826.0.1.3680043.8.498.77257791208759842602760935296318202703',
 '1.2.826.0.1.3680043.8.498.79099213587801933936080747802403048718',
 '1.2.826.0.1.3680043.8.498.79942836660118710928733936389534291771',
 '1.2.826.0.1.3680043.8.498.80048101091444895066772572129871971243',
 '1.2.826.0.1.3680043.8.498.80114244849666367523293067199486077713',
 '1.2.826.0.1.3680043.8.498.80190289468142266421549927426167714158',
 '1.2.826.0.1.3680043.8.498.80461517820710375402982229582943598734',
 '1.2.826.0.1.3680043.8.498.81098958708250149437576237811675033160',
 '1.2.826.0.1.3680043.8.498.82247540847692847800462620079965863384',
 '1.2.826.0.1.3680043.8.498.82641698422464356104108563099150990855',
 '1.2.826.0.1.3680043.8.498.84908441442551598157537604822760711232',
 '1.2.826.0.1.3680043.8.498.84955070686251417902923705821409495324',
 '1.2.826.0.1.3680043.8.498.85709849872024108265120796348331660195',
 '1.2.826.0.1.3680043.8.498.86822530556046989269633487715061058236',
 '1.2.826.0.1.3680043.8.498.87794163393266428648659243169230666286',
 '1.2.826.0.1.3680043.8.498.88044882887797890422716086408658477347',
 '1.2.826.0.1.3680043.8.498.88512241250207324783783101806489145581',
 '1.2.826.0.1.3680043.8.498.88662334466087798807484415780594176763',
 '1.2.826.0.1.3680043.8.498.88739296218460643753583291722714541935',
 '1.2.826.0.1.3680043.8.498.88905360377095450551559885185901908404',
 '1.2.826.0.1.3680043.8.498.89421386426320866039573378582181968701',
 '1.2.826.0.1.3680043.8.498.89990837914171555676446644356114244393',
 '1.2.826.0.1.3680043.8.498.90015157820692758596783999454928886688',
 '1.2.826.0.1.3680043.8.498.90168683694094931217787644438845074017',
 '1.2.826.0.1.3680043.8.498.92418959634964175917370213963992652610',
 '1.2.826.0.1.3680043.8.498.92543328866053664733167983708344898988',
 '1.2.826.0.1.3680043.8.498.92773748942952645243074808740855383414',
 '1.2.826.0.1.3680043.8.498.96218477847514569819859044953648183121',
 '1.2.826.0.1.3680043.8.498.97057911327885502714270510313728134927',
 '1.2.826.0.1.3680043.8.498.97256479550884529885940791074752719030',
 '1.2.826.0.1.3680043.8.498.98123758735027035609698227781754927939',
 '1.2.826.0.1.3680043.8.498.98133633346919790888527055899070500258']

if __name__ == '__main__':
    a = np.ones((1, 10, 10))
    print(a)
    b = crop_pad(a, 0, 1, 2, (1, 16, 16))
    c = crop_pad(a, 0, 1, 2, (1, 5, 15))
    print(b)
    print(c.shape)