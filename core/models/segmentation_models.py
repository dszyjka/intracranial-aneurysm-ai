from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, channels, conv_layers):
        super().__init__()

        layers = []
        
        for _ in range(conv_layers):
            layers.extend(
                (
                    nn.Conv3d(channels, channels, kernel_size=5, stride=1, padding='same'), # here 'same' = 2
                    nn.BatchNorm3d(channels),
                    nn.PReLU()
                )
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        res = x
        x = self.block(x)
        return x + res

if __name__ == '__main__':
    res = ResidualBlock(32, 4)