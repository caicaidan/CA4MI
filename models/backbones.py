import torch
import torch.nn as nn
from typing import List
import numpy as np
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):

    def __init__(self,
                 args,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 latent_dim: int = 128,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.n_times = args.n_times
        self.n_chans = args.n_chans
        self.latent_dim = args.latent_dim
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.n_chans, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))

        self.linear = nn.Linear(self.feature_dim(), latent_dim, bias=False)

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.n_chans, self.n_times)
            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return self.F2 * mock_eeg.shape[3]



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x

class DeepConvNet(nn.Module):
    def __init__(self, args, latent_dim: int = 64, kernel_1: int = 64, kernel_2: int = 16,
                 dropout: float = 0., block_out_channels: List[int] = [25, 25, 50, 100, 200], pool_size: int = 4):
        """DeepConvNet implementation with customizable parameters."""
        super(DeepConvNet, self).__init__()
        self.n_chans, self.n_times = args.n_chans, args.n_times

        self.first_conv_block = nn.Sequential(
            Conv2dWithConstraint(1, block_out_channels[0], kernel_size=(1, kernel_1), max_norm=2, padding=(0, kernel_1 // 2)),
            Conv2dWithConstraint(block_out_channels[0], block_out_channels[1], kernel_size=(self.n_chans, 1), max_norm=2, bias=False),
            nn.BatchNorm2d(block_out_channels[1]), nn.ELU(), nn.MaxPool2d((1, pool_size))
        )

        self.deep_block = nn.ModuleList([
            self._default_block(block_out_channels[i - 1], block_out_channels[i], kernel_2, pool_size) for i in range(2, 5)
        ])

        self.linear = nn.Linear(self._feature_dim(), latent_dim, bias=False)

    def _default_block(self, in_channels, out_channels, kernel_size, pool_size):
        return nn.Sequential(
            nn.Dropout(0.25),
            Conv2dWithConstraint(in_channels, out_channels, (1, kernel_size), max_norm=2, padding=(0, kernel_size // 2), bias=False),
            nn.BatchNorm2d(out_channels), nn.ELU(), nn.AvgPool2d((1, pool_size))
        )

    def _feature_dim(self):
        """Calculate the flattened dimension after convolutions."""
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.n_chans, self.n_times)
            mock_eeg = self.first_conv_block(mock_eeg)
            for block in self.deep_block:
                mock_eeg = block(mock_eeg)
        return mock_eeg.shape[1] * mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.first_conv_block(x)
        for block in self.deep_block:
            x = block(x)
        x = x.flatten(start_dim=1)
        return self.linear(x)

class ShallowConvNet(nn.Module):
    def __init__(self, args, F1=8, F2=16, D=2, latent_dim=128, depth=24, dropout=0.25):
        """ShallowConvNet implementation with customizable parameters."""
        super(ShallowConvNet, self).__init__()
        self.n_chans, self.n_times = int(args.n_chans), int(args.n_times)

        self.block = nn.Sequential(
            nn.Conv2d(1, F1, (1, depth), padding=(0, depth // 2), bias=False),
            nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(F1, F1 * D, (self.n_chans, 1), max_norm=1, groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(p=dropout)
        )

        self.linear = nn.Linear(self._feature_dim(), latent_dim, bias=False)

    def _feature_dim(self):
        """Calculate the flattened dimension after convolutions."""
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.n_chans, self.n_times)
            mock_eeg = self.block(mock_eeg)
        return int(np.prod(mock_eeg.size()[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.block(x)
        x = x.flatten(start_dim=1)
        return self.linear(x)




