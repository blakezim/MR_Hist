from .vnet_parts import *


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=False, nll=False):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(4, elu)
        self.down_tr32 = DownTransition(4, 1, elu)
        self.down_tr64 = DownTransition(8, 2, elu)
        self.down_tr128 = DownTransition(16, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(32, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(64, 64, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(64, 32, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(32, 16, 1, elu)
        self.up_tr32 = UpTransition(16, 8, 1, elu)
        self.out_tr = OutputTransition(8, elu, nll)
        self.linear1 = nn.Conv3d(22, 32, 3, padding=1)
        self.linear2 = nn.Conv3d(32, 64, 1)
        self.linear3 = nn.Conv3d(64, 1, 1)

    def forward(self, x, eval=False):
        skip = x.clone()
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128, eval)
        out = self.up_tr128(out, out64, eval)
        out = self.up_tr64(out, out32, eval)
        out = self.up_tr32(out, out16, eval)
        out = self.out_tr(out)
        out = self.linear1(torch.cat((out, skip), dim=1))
        out = self.linear2(out)
        out = self.linear3(out)

        return out
