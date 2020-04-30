import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_channels: int = 3, shrink_factor: int = 64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(input_channels, shrink_factor, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(shrink_factor, shrink_factor * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(shrink_factor * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(shrink_factor * 2, shrink_factor * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(shrink_factor * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(shrink_factor * 4, shrink_factor * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(shrink_factor * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(shrink_factor * 8, 1, 2, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


