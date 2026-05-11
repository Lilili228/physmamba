import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, max(1, in_planes // ratio), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_planes // ratio), in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(res))


class ResidualCBAM(nn.Module):
    def __init__(self, channels):
        super(ResidualCBAM, self).__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        att = x * self.ca(x)
        att = att * self.sa(att)
        return x + self.beta * att

class CSAF(nn.Module):
    def __init__(self, curr_ch, shallow_ch, deep_ch, groups=2):
        super(CSAF, self).__init__()
        self.groups = groups
        self.curr_ch = curr_ch
        self.shallow_align = None
        self.deep_align = None



        if shallow_ch is not None:
            self.shallow_align = nn.Sequential(
                nn.Conv2d(shallow_ch, curr_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_ch, affine=True),
                nn.LeakyReLU(0.2, True)
            )
        if deep_ch is not None:
            self.deep_align = nn.Sequential(
                nn.Conv2d(deep_ch, curr_ch, kernel_size=1, bias=False),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.InstanceNorm2d(curr_ch, affine=True),
                nn.LeakyReLU(0.2, True)
            )


        self.weight_gen = nn.Sequential(
            nn.Conv2d(curr_ch, curr_ch // 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(curr_ch // 4, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(curr_ch // 4, groups * 2, kernel_size=1)
        )


        self.fusion_conv = nn.Sequential(
            nn.Conv2d(curr_ch, curr_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(curr_ch, affine=True),
            nn.ReLU(True)
        )


        self.final_proj = nn.Sequential(
            nn.Conv2d(curr_ch * 2, curr_ch, kernel_size=1, bias=False),
            nn.InstanceNorm2d(curr_ch, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(curr_ch, curr_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(curr_ch, affine=True)
        )

        self.cbam = ResidualCBAM(curr_ch)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(True)

    def forward(self, x_curr, x_shallow=None, x_deep=None):
        B, C, H, W = x_curr.size()
        identity = x_curr


        l = self.shallow_align(x_shallow) if (self.shallow_align and x_shallow is not None) else torch.zeros_like(
            x_curr)
        h = self.deep_align(x_deep) if (self.deep_align and x_deep is not None) else torch.zeros_like(x_curr)


        raw_weights = self.weight_gen(x_curr)
        weights = F.softmax(raw_weights.view(B, self.groups, 2, H, W), dim=2)


        l_groups = torch.chunk(l, self.groups, dim=1)
        h_groups = torch.chunk(h, self.groups, dim=1)

        fused_list = []
        for i in range(self.groups):
            w_l, w_h = weights[:, i, 0:1, :, :], weights[:, i, 1:2, :, :]
            fused_list.append(w_l * l_groups[i] + w_h * h_groups[i])

        delta = self.fusion_conv(torch.cat(fused_list, dim=1))
        combined = torch.cat([identity, self.gamma * delta], dim=1)
        out = self.relu(identity + self.final_proj(combined))

        return self.cbam(out)
