from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 16)
        self.down1 = down(16, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 64)
        # self.down4 = down(256, 256)
        # self.up1 = up(512, 128)
        self.up2 = up(128, 32)
        self.up3 = up(64, 16)
        self.up4 = up(32, 16)
        self.outc = outconv(16, 2)
        self.linear1 = nn.Conv2d(8, 32, 1)
        self.linear2 = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        skip = x.clone()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.linear1(torch.cat((x, skip), dim=1))
        x = self.linear2(x)
        return x
