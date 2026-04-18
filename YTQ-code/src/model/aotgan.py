import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .common import BaseNetwork


class InpaintGenerator(BaseNetwork):
    def __init__(self, args):  # 1046
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, 7),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
        )

        self.middle = nn.Sequential(*[AOTBlock(256, args.rates) for _ in range(args.block_num)])

        self.decoder = nn.Sequential(
            UpConv(256, 128), nn.ReLU(True), UpConv(128, 64), nn.ReLU(True), nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

        self.edge_head = EdgeHead(in_ch=256, mid_ch=128)

        self.init_weights()

    def forward(self, x, mask, return_edge: bool = False):
        # x: (B,3,H,W)  mask: (B,1,H,W)
        x_in = torch.cat([x, mask], dim=1)

        feat = self.encoder(x_in)  # (B,256,H/4,W/4)
        x_mid = self.middle(feat)
        pred = self.decoder(x_mid)
        pred = torch.tanh(pred)

        if not return_edge:
            return pred

        # edge logits: (B,1,H,W)  (logits, no sigmoid)
        edge_logits = self.edge_head(feat, target_size=mask.shape[-2:])
        return pred, edge_logits


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))

class EdgeHead(nn.Module):
    """
    输入：encoder 特征 (B, 256, H/4, W/4)
    输出：edge logits (B, 1, H, W)  —— 注意：logits，不要 sigmoid
    """
    def __init__(self, in_ch=256, mid_ch=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch // 2, 3, padding=1)
        self.conv3 = nn.Conv2d(mid_ch // 2, 1, 3, padding=1)
        self.act = nn.ReLU(True)

    def forward(self, feat, target_size):
        x = self.act(self.conv1(feat))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.act(self.conv2(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv3(x)  # logits: (B,1, H/?, W/?)

        # 强制对齐到目标尺寸（最稳，避免 255/256 这种奇怪对不上）
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x

class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                "block{}".format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate), nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate), nn.ReLU(True)
                ),
            )
        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f"block{str(i).zfill(2)}")(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat


# ----- discriminator -----
class Discriminator(BaseNetwork):
    def __init__(
        self,
    ):
        super(Discriminator, self).__init__()
        inc = 3
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(inc, 64, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

        self.init_weights()

    def forward(self, x):
        feat = self.conv(x)
        return feat
