import torch
from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, channels, conv_layers):
        super().__init__()

        layers = []
        
        for _ in range(conv_layers):
            layers.extend(
                (
                    nn.Conv3d(channels, channels, kernel_size=5, stride=1, padding='same'), # here 'same' = 2
                    nn.InstanceNorm3d(channels),
                    nn.PReLU(channels)
                )
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        res = x
        x = self.block(x)
        return x + res

class VNet(nn.Module):

    def __init__(self, start_channels, class_num):
        super().__init__()

        self.stem_conv = nn.Sequential(
            nn.Conv3d(1, start_channels, kernel_size=5, stride=1, padding='same'),
            nn.InstanceNorm3d(start_channels),
            nn.PReLU(start_channels)
        )

        self.encoder_res_blocks = nn.ModuleList([
            ResidualBlock(start_channels, 1),
            ResidualBlock(start_channels * 2, 2),
            ResidualBlock(start_channels * 4, 3),
            ResidualBlock(start_channels * 8, 3)
        ])

        self.down_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(start_channels, start_channels * 2, kernel_size=2, stride=2),
                nn.PReLU(start_channels * 2)
            ),
            nn.Sequential(
                nn.Conv3d(start_channels * 2, start_channels * 4, kernel_size=2, stride=2),
                nn.PReLU(start_channels * 4)
            ),
            nn.Sequential(
                nn.Conv3d(start_channels * 4, start_channels * 8, kernel_size=2, stride=2),
                nn.PReLU(start_channels * 8)
            ),
            nn.Sequential(
                nn.Conv3d(start_channels * 8, start_channels * 16, kernel_size=2, stride=2),
                nn.PReLU(start_channels * 16)
            ),
        ])

        self.bottom_block = ResidualBlock(start_channels * 16, 3)

        self.decoder_res_blocks = nn.ModuleList([
            ResidualBlock(start_channels * 16, 3),
            ResidualBlock(start_channels * 8, 3),
            ResidualBlock(start_channels * 4, 2),
            ResidualBlock(start_channels * 2, 1)
        ])

        self.up_convs = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(start_channels * 16, start_channels * 8, kernel_size=2, stride=2),
                nn.PReLU(start_channels * 8)
            ),
            nn.Sequential(
                nn.ConvTranspose3d(start_channels * 16, start_channels * 4, kernel_size=2, stride=2),
                nn.PReLU(start_channels * 4)
            ),
            nn.Sequential(
                nn.ConvTranspose3d(start_channels * 8, start_channels * 2, kernel_size=2, stride=2),
                nn.PReLU(start_channels * 2)
            ),
            nn.Sequential(
                nn.ConvTranspose3d(start_channels * 4, start_channels, kernel_size=2, stride=2),
                nn.PReLU(start_channels)
            )
        ])

        self.last_conv = nn.Conv3d(start_channels * 2, class_num, kernel_size=1, stride=1)

    def forward(self, x):
        horizontal_connections = []
        
        x = self.stem_conv(x)

        for res_block, down_conv in zip(self.encoder_res_blocks, self.down_convs):
            x = res_block(x)
            horizontal_connections.append(x)
            x = down_conv(x)

        x = self.bottom_block(x)

        for res_block, up_conv in zip(self.decoder_res_blocks, self.up_convs):
            x = up_conv(x)
            encoder_x = horizontal_connections.pop()
            x = torch.cat((x, encoder_x), dim=1)
            x = res_block(x)

        x = self.last_conv(x)

        return x
