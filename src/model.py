"""Models (Step 5).

cnn       : small CNN, no BatchNorm (Tier A, decision D9). About 0.55 million parameters.
resnet18  : CIFAR-style ResNet-18 (3x3 first layer, no max-pool; decision D12),
            about 11.17 million parameters. norm="batch" (default) or "group" (decision D9).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GN_GROUPS = 8  # number of groups when norm="group" (write the final choice in notes/decisions.md)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 32x32 -> 16x16
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 16x16 -> 8x8
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def _norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        return nn.GroupNorm(GN_GROUPS, channels)
    raise ValueError(f"unknown norm '{kind}' (use 'batch' or 'group')")


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int, norm: str):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.n1 = _norm(norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.n2 = _norm(norm, planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), _norm(norm, planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, norm: str = "batch"):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.n1 = _norm(norm, 64)
        blocks, in_planes = [], 64
        for planes, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            for s in (stride, 1):
                blocks.append(BasicBlock(in_planes, planes, s, norm))
                in_planes = planes
        self.layers = nn.Sequential(*blocks)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.n1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def build_model(name: str, num_classes: int = 10, norm: str = "batch") -> nn.Module:
    if name == "cnn":
        return SmallCNN(num_classes)
    if name == "resnet18":
        return ResNet18(num_classes, norm)
    raise ValueError(f"unknown model '{name}' (use 'cnn' or 'resnet18')")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
