from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# ------------------------------
# Model Definition
# ------------------------------
dropout_value = 0.05

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # CONVOLUTION BLOCK 1 (C1) — RF: 9
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=0, bias=False),  # 32x32 → 30x30, RF=3
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(32, 64, kernel_size=3, padding=0, bias=False), # 30x30 → 28x28, RF=5
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(64, 64, kernel_size=3, padding=0, dilation=2, bias=False), # 28x28 → 24x24, RF=9
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        )

        # CONVOLUTION BLOCK 2 (C2) — RF: 13
        self.c2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=False), # 24x24 → 24x24, RF=9
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),  # Depthwise, RF=11
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=1, padding=0, bias=False),  # Pointwise, RF=11
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(64, 64, kernel_size=3, padding=0, bias=False), # 24x24 → 22x22, RF=13
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        )

        # CONVOLUTION BLOCK 3 (C3) — RF: 21
        self.c3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=2, bias=False), # 22x22 → 10x10, RF=17
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=False), # 10x10 → 10x10, RF=17
            nn.AvgPool2d(2, 2)  # 10x10 → 5x5, RF=21
        )

        # CONVOLUTION BLOCK 4 (C4) — RF: 44
        self.c4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),  # 5x5 → 5x5, RF=25
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),  # 5x5 → 5x5, RF=29
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.AvgPool2d(kernel_size=5)  # 5x5 → 1x1, RF=44
        )

        # OUTPUT BLOCK (O) — 1x1x10
        self.output = nn.Conv2d(64, 10, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x = self.c1(x)       # → 24x24x64
        x = self.c2(x)       # → 22x22x64
        x = self.c3(x)       # → 5x5x32
        x = self.c4(x)       # → 1x1x64
        x = self.output(x)   # → 1x1x10
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

