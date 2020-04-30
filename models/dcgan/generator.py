import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, out_channels: int = 3, latent: int = 100, grow_factor: int = 64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent, grow_factor * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(grow_factor * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(grow_factor * 8, grow_factor * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(grow_factor * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(grow_factor * 4, grow_factor * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(grow_factor * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(grow_factor * 2, grow_factor, 4, 2, 1, bias=False),
            nn.BatchNorm2d(grow_factor),
            nn.ReLU(True),
            nn.ConvTranspose2d(grow_factor, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


