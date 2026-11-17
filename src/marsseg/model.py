"""U-Net with a pretrained ResNet-18 encoder.

Encoder ("what is in the picture"): ResNet-18 trained on ImageNet photos. Each stage halves
the resolution and adds channels: 256 -> 128 -> 64 -> 32 -> 16 -> 8 pixels across.
Decoder ("where exactly"): upsample back to full size, at each step concatenating the
encoder's feature map of the same size (the "skip connections" that give U-Net its U shape
and keep sharp edges). A final 1x1 conv gives 4 scores per pixel, one per terrain class.

Navcam images are grayscale, ResNet expects RGB. Instead of copying the gray channel three
times we sum the first conv layer's RGB weights into one channel: same result, 3x cheaper.
"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class Up(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in + c_skip, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetResNet18(nn.Module):
    def __init__(self, n_classes: int = 4, pretrained: bool = True):
        super().__init__()
        r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        conv1.weight.data = r.conv1.weight.data.sum(dim=1, keepdim=True)
        self.stem = nn.Sequential(conv1, r.bn1, r.relu)        # 1/2 resolution, 64 ch
        self.pool = r.maxpool
        self.enc1, self.enc2, self.enc3, self.enc4 = r.layer1, r.layer2, r.layer3, r.layer4
        # enc1: 1/4, 64ch | enc2: 1/8, 128ch | enc3: 1/16, 256ch | enc4: 1/32, 512ch
        self.up3 = Up(512, 256, 256)
        self.up2 = Up(256, 128, 128)
        self.up1 = Up(128, 64, 64)
        self.up0 = Up(64, 64, 32)
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        e1 = self.enc1(self.pool(s0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d = self.up0(self.up1(self.up2(self.up3(e4, e3), e2), e1), s0)
        d = nn.functional.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(d)  # (batch, n_classes, H, W) raw scores ("logits")
