# Placeholder stub for Bradley-Terry-Luce models
# This is imported in environments.py but not actually used

import torch.nn as nn


class RewardNet(nn.Module):
    """Placeholder reward network class."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, x):
        pass
