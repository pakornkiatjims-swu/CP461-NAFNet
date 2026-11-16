"""NAFNet architecture (Chen et al., ECCV 2022), identical to the notebook used for training.

Parameter names must stay unchanged so that nafnet_ft.pth loads with strict=True.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return (self.weight.view(1, -1, 1, 1) * (x - mu) / (var + self.eps).sqrt()
                + self.bias.view(1, -1, 1, 1))


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c, DW=2, FFN=2):
        super().__init__()
        d = c * DW
        self.conv1 = nn.Conv2d(c, d, 1)
        self.conv2 = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.conv3 = nn.Conv2d(d // 2, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(d // 2, d // 2, 1))
        self.sg = SimpleGate()
        f = FFN * c
        self.conv4 = nn.Conv2d(c, f, 1)
        self.conv5 = nn.Conv2d(f // 2, c, 1)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.conv2(self.conv1(self.norm1(inp)))
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=(), dec_blk_nums=()):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        c = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, 2))
            c *= 2
        self.middle_blks = nn.Sequential(*[NAFBlock(c) for _ in range(middle_blk_num)])
        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        p = self.padder_size
        pad = (0, (p - W % p) % p, 0, (p - H % p) % p)
        x = F.pad(inp, pad)
        x = self.intro(x)
        encs = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for dec, up, skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x) + skip
            x = dec(x)
        x = self.ending(x)
        x = x + F.pad(inp, pad)
        return x[:, :, :H, :W]


def build_model():
    """NAFNet-width64 as used for REDS deblurring (67.9M parameters)."""
    return NAFNet(img_channel=3, width=64, middle_blk_num=1,
                  enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
