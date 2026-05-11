import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from einops import rearrange, repeat
from timm.models.layers import DropPath
import matplotlib.pyplot as plt
import numpy as np
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn



class ThermalFeatureExtractor(nn.Module):


    def __init__(self, bins=32, embed_dim=128):
        super().__init__()
        self.bins = bins
        self.projection = nn.Sequential(
            nn.Linear(bins + 2, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        bin_centers = torch.linspace(-1, 1, bins)
        self.register_buffer('bin_centers', bin_centers)
        self.sigma = 5.0/ bins

    def forward(self, ir_img):
        b = ir_img.shape[0]


        mean = ir_img.mean(dim=[1, 2, 3], keepdim=True).view(b, 1)

        std = ir_img.std(dim=[1, 2, 3], keepdim=True).clamp(min=1e-6).view(b, 1)

        x = ir_img.view(b, -1, 1)
        dist = torch.abs(x - self.bin_centers)
        soft_assign = torch.exp(-dist ** 2 / (2 * self.sigma ** 2))
        soft_hist = soft_assign.mean(dim=1)


        soft_hist = soft_hist / (soft_hist.sum(dim=1, keepdim=True) + 1e-8)


        stats_vec = torch.cat([mean, std, soft_hist], dim=1)
        thermal_token = self.projection(stats_vec)


        return thermal_token.view(b, 1, 1, -1)




class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        self.x_proj_weight = nn.Parameter(torch.empty(4, (self.dt_rank + self.d_state * 2), self.d_inner))
        nn.init.orthogonal_(self.x_proj_weight)

        self.dt_projs_weight = nn.Parameter(torch.empty(4, self.d_inner, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.empty(4, self.d_inner))
        nn.init.orthogonal_(self.dt_projs_weight)

        self.A_logs = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float().repeat(self.d_inner * 4, 1))
        )
        self.Ds = nn.Parameter(torch.ones(self.d_inner * 4))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack(
            [x.view(B, -1, L),
             torch.transpose(x, 2, 3).contiguous().view(B, -1, L)],
            dim=1
        )
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (B, K, d, L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs, dts, Bs, Cs = xs.float(), dts.float(), Bs.float(), Cs.float()
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)

        out_y = selective_scan_fn(
            xs.reshape(B, -1, L),
            dts.reshape(B, -1, L),
            As,
            Bs.reshape(B, K, self.d_state, L),
            Cs.reshape(B, K, self.d_state, L),
            self.Ds.float(),
            z=None,
            delta_bias=self.dt_projs_bias.reshape(-1),
            delta_softplus=True,
        ).view(B, K, -1, L)

        y1 = out_y[:, 0]
        y2 = torch.transpose(out_y[:, 1].reshape(B, -1, W, H), 2, 3).contiguous().view(B, -1, L)
        y3 = torch.flip(out_y[:, 2], dims=[-1])
        y4 = torch.transpose(
            torch.flip(out_y[:, 3], dims=[-1]).reshape(B, -1, W, H), 2, 3
        ).contiguous().view(B, -1, L)

        return y1 + y2 + y3 + y4

    def forward(self, x: torch.Tensor):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        y = torch.transpose(y, 1, 2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y) * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out




class VSSBlock_Thermal(nn.Module):
    def __init__(self, dim, d_state=16, drop_path=0.):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.ss2d = SS2D(d_model=dim, d_state=d_state)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.thermal_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x, thermal_token):

        shortcut = x
        x = self.ln(x)
        x = self.ss2d(x)


        gate = self.thermal_gate(thermal_token)
        x = x * gate

        return shortcut + self.drop_path(x)


class MambaThermalDiscriminator(nn.Module):


    def __init__(self, input_nc=2, embed_dim=128, patch_size=16, depth=4):
        super().__init__()

        self.patch_embed = nn.Conv2d(input_nc, embed_dim, kernel_size=patch_size, stride=patch_size)


        self.thermal_extractor = ThermalFeatureExtractor(bins=64, embed_dim=embed_dim)


        self.blocks = nn.ModuleList([
            VSSBlock_Thermal(dim=embed_dim, d_state=16) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        ir_img = x[:, -1:, :, :]
        thermal_token = self.thermal_extractor(ir_img)

        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1).contiguous()

        for blk in self.blocks:
            x = blk(x, thermal_token)

        x = self.norm(x)


        logits = self.classifier(x)
        logits = logits.permute(0, 3, 1, 2).contiguous()

        return logits

