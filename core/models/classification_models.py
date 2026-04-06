import torch
from torch import nn
from transformers import AutoModel
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights, shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights


class STN(nn.Module):
    def __init__(self, input_channels=10, output_size=(224, 224)):
        super(STN, self).__init__()

        self.output_size = output_size
        self.slice_start = max(0, -(input_channels - 10))
        self.read_theta = False
        
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(True),
            nn.Linear(128, 6),
        )
        
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = x[:, self.slice_start:, :, :]
        xs = self.localization(xs)
        xs = xs.view(-1, 64 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        if self.read_theta:
            print(f'Theta: {theta}')

        target_shape = torch.Size([x.shape[0], x.shape[1], self.output_size[0], self.output_size[1]])
        
        grid = F.affine_grid(theta, target_shape, align_corners=False)
        
        x_transformed = F.grid_sample(x, grid, align_corners=False)
        
        return x_transformed

class AneurysmClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, output_size=(224, 224), stn_only_on_seg=False):
        super(AneurysmClassifier, self).__init__()

        self.stn = STN(5, output_size) if stn_only_on_seg else STN(in_channels, output_size)
        weights = ResNet18_Weights.DEFAULT
        self.resnet = resnet18(weights=weights)

        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        ''' # for resnet50
        for name, layer in self.resnet.named_children():
            if name in ('layer4', 'fc'):
                layer.trainable = True
            else:
                layer.trainable = False'''
        
        orig_weights = self.resnet.conv1.weight.data
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        avg_weights = orig_weights.mean(dim=1, keepdim=True)
        self.resnet.conv1.weight.data = avg_weights.repeat(1, 10, 1, 1)

        self.resnet.conv1.weight.requires_grad = True

        self.resnet.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.stn(x)
        x = self.resnet(x)
        return x
    
class AneurysmClassifierDINO(nn.Module):

    def __init__(self, in_channels, num_classes, output_size=(224, 224), stn_only_on_seg=False):
        super().__init__()

        self.stn = STN(5, output_size) if stn_only_on_seg else STN(in_channels, output_size)
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base")

        for param in self.dino.parameters():
            param.requires_grad = False

        self.prepare_to_dino = nn.Conv2d(in_channels, 3, kernel_size=1)

        self.clf = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stn(x)
        x = self.prepare_to_dino(x)
        outputs = self.dino(pixel_values=x)
        x = outputs.last_hidden_state[:, 0, :]
        x = self.clf(x)
        return x

class AneurysmClassifierResNet(nn.Module):

    def __init__(self, in_channels, num_classes, all_trainable=False):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        if not all_trainable:
            for param in self.resnet.parameters():
                param.requires_grad = False

            for param in self.resnet.layer4.parameters():
                param.requires_grad = True

        orig_conv1_weights = self.resnet.conv1.weight.clone()
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            avg_orig_weights = orig_conv1_weights.mean(dim=1, keepdim=True)
            self.resnet.conv1.weight.copy_(avg_orig_weights.repeat(1, in_channels, 1, 1))

        self.resnet.conv1.weight.requires_grad = True

        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

class AneurysmClassifierMobileNet(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        orig_conv1 = self.mobilenet.features[0][0]

        self.mobilenet.features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=orig_conv1.out_channels,
            kernel_size=orig_conv1.kernel_size,
            stride=orig_conv1.stride,
            padding=orig_conv1.padding,
            bias=False)
        
        with torch.no_grad():
            mean_conv1_weights = orig_conv1.weight.mean(dim=1, keepdim=True)
            self.mobilenet.features[0][0].weight.copy_(mean_conv1_weights.repeat(1, in_channels, 1, 1))

        self.mobilenet.features[0][0].weight.requires_grad = True
        
        in_features = self.mobilenet.classifier[-1].in_features
        self.mobilenet.classifier[-1] = nn.Linear(in_features, num_classes)


    def forward(self, x):
        return self.mobilenet(x)
    
class AneurysmClassifierShuffleNet(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.shufflenet = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
        orig_conv1 = self.shufflenet.conv1[0]

        self.shufflenet.conv1[0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=orig_conv1.out_channels,
            kernel_size=orig_conv1.kernel_size,
            stride=orig_conv1.stride,
            padding=orig_conv1.padding,
            bias=False
        )

        with torch.no_grad():
            mean_conv1_weights = orig_conv1.weight.mean(dim=1, keepdim=True)
            self.shufflenet.conv1[0].weight.copy_(mean_conv1_weights.repeat(1, in_channels, 1, 1))
            
            self.shufflenet.conv1[0].weight.requires_grad = True

        in_features = self.shufflenet.fc.in_features
        self.shufflenet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.shufflenet(x)
