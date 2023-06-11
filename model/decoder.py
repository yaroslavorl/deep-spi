import torch.nn as nn
import torch.nn.functional as F


class Coder(nn.Module):
    def __init__(self, patterns, size_img):
        super().__init__()
        self.coder = nn.Conv2d(1, patterns, kernel_size=size_img, bias=False)

    def forward(self, x):
        return self.coder(x)


class Decoder(nn.Module):

    def __init__(self, patterns, size_img):
        super().__init__()

        self.norm_1 = nn.BatchNorm2d(patterns)
        self.decoder = nn.ConvTranspose2d(patterns, 1, kernel_size=size_img, bias=False)
        self.norm_2 = nn.BatchNorm2d(1)
        self.conv_1 = nn.Conv2d(1, 64, kernel_size=(9, 9), padding=4)
        self.conv_2 = nn.Conv2d(64, 32, kernel_size=(1, 1), padding=0)
        self.conv_3 = nn.Conv2d(32, 1, kernel_size=(5, 5), padding=2)

    def forward(self, x):
        x = self.norm_1(x)
        x = self.decoder(x)
        x = self.norm_2(x)
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        return x
