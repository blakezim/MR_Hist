from .unet_parts import *
from .dense_parts import *


class UNetDense(nn.Module):
    def __init__(self, n_channels, n_transitions, n_classes):
        super(UNetDense, self).__init__()
        growth_rate = 4
        dense_blocks = 4
        self.inc = inconv(n_channels, 16)
        self.down1 = down(16, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 64)
        self.up2 = up(128, 32)
        self.up3 = up(64, 16)
        self.up4 = up(32, n_transitions - 1)
        self.dense = self._make_dense(n_transitions, growth_rate, dense_blocks)
        self.outc = outconv(n_transitions + (growth_rate * dense_blocks), n_classes)

    @staticmethod
    def _make_dense(nChannels, growthRate, nDenseBlocks):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x, t1):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = torch.cat([x, t1], 1)
        x = self.dense(x)
        x = self.outc(x)
        return x
