
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.shape
        avg = self.mlp(self.avg_pool(x).view(b, c))
        mx  = self.mlp(self.max_pool(x).view(b, c))
        w = self.sigmoid(avg + mx).view(b, c, 1)
        return x * w

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        w = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w

class CBAM(nn.Module):
    def __init__(self, channels, level):
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention(level)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x

class EstimationSubnet(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv1d(64, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_samples, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x)) # (b, 1, n_samples)
        x = x.view(x.size(0), -1) # (b, n_samples)
        eta = self.fc(x) # (b, 1)
        return eta
class DenoisingSubnet(nn.Module): 
    def __init__(self):
        super().__init__()
        self.in_conv = nn.Conv1d(2, 64, 3, padding=1)

        body = []
        for _ in range(7):               # layers 2-8
            body += [nn.Conv1d(64, 64, 3, padding=1),
                     nn.BatchNorm1d(64),
                     nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*body)

        self.cbam = CBAM(64)
        self.out_conv = nn.Conv1d(64, 1, 3, padding=1) # predicts noise r
        self.relu = nn.ReLU(inplace=True)

    def forward(self, noisy, eta):
        b, _, n = noisy.shape
        eta_map = eta.view(b, 1, 1).expand(b, 1, n)
        x = torch.cat([noisy, eta_map], dim=1) # (b, 2, n)

        x = self.relu(self.in_conv(x))
        x = self.body(x)
        x = self.cbam(x)
        r = self.out_conv(x) # (b, 1, n) predicted noise
        return r

class TSCNN(nn.Module):
    def __init__(self, n_samples=1001):
        super().__init__()
        self.es = EstimationSubnet(n_samples)
        self.ds = DenoisingSubnet()

    def forward(self, noisy):
        eta = self.es(noisy)             # (b, 1) noise level estimate
        r = self.ds(noisy, eta)          # (b, 1, n) predicted noise
        x_hat = noisy - r                # residual: clean = noisy - noise
        return x_hat, eta