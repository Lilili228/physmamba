import functools
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from models.unetgan import layers



class Unet_DiscriminatorGenerator(nn.Module):
    def __init__(self, input_nc, output_nc=None, ngf=64, use_dropout=False, resolution=512, gpu_ids=[]):
        super(Unet_DiscriminatorGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # ============================================================
        # 銆愬弻娴佹灦鏋勬牳蹇冪粍瑁呫€?        # 娴?1: 鍘熸湰鐨?UNet锛屾仮澶嶇函鍑€ CNN锛岃緭鍑哄叏灏哄绌洪棿鐭╅樀 (鎶撳井瑙傦紝淇?LPIPS)
        self.model_unet = Unet_Discriminator(input_nc, resolution=resolution)

    def forward(self, inp):
        if self.gpu_ids and isinstance(inp.data, torch.cuda.FloatTensor):
            # U-Net 浼氳繑鍥?(鍍忕礌绾ц緭鍑? 鐡堕灞傝緭鍑?
            unet_out, bottleneck_out = torch.nn.parallel.data_parallel(self.model_unet, inp, self.gpu_ids)
        else:
            unet_out, bottleneck_out = self.model_unet(inp)

        # Return both local and bottleneck predictions for the PhysMamba GAN losses.
        return unet_out, bottleneck_out


def D_unet_arch(input_c=3, ch=64, attention='64', ksize='333333', dilation='111111', out_channel_multiplier=1):
    # (杩欓儴鍒嗕唬鐮佷繚鎸佷綘鐨勫師鏍蜂笉鍔?
    arch = {}
    n = 2
    ocm = out_channel_multiplier

    arch[128] = {'in_channels': [input_c] + [ch * item for item in [1, 2, 4, 8, 16, 8 * n, 4 * 2, 2 * 2, 1 * 2, 1]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 8, 4, 2, 1, 1]],
                 'downsample': [True] * 5 + [False] * 5,
                 'upsample': [False] * 5 + [True] * 5,
                 'resolution': [64, 32, 16, 8, 4, 8, 16, 32, 64, 128],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 11)}}

    arch[256] = {
        'in_channels': [input_c] + [ch * item for item in [1, 2, 4, 8, 8, 16, 8 * 2, 8 * 2, 4 * 2, 2 * 2, 1 * 2, 1]],
        'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 8, 8, 4, 2, 1, 1]],
        'downsample': [True] * 6 + [False] * 6,
        'upsample': [False] * 6 + [True] * 6,
        'resolution': [128, 64, 32, 16, 8, 4, 8, 16, 32, 64, 128, 256],
        'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                      for i in range(2, 13)}}

    arch[512] = {'in_channels': [input_c] + [ch * item for item in
                                             [1, 2, 4, 8, 8, 16, 32, 16 * 2, 8 * 2, 8 * 2, 4 * 2, 2 * 2, 1 * 2, 1]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 32, 16, 8, 8, 4, 2, 1, 1]],
                 'downsample': [True] * 7 + [False] * 7,
                 'upsample': [False] * 7 + [True] * 7,
                 'resolution': [256, 128, 64, 32, 16, 8, 4, 8, 16, 32, 64, 128, 256, 512],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 15)}}
    return arch


class Unet_Discriminator(nn.Module):
    def __init__(self, input_c, D_ch=64, D_wide=True, resolution=512,
                 D_kernel_size=3, D_attn='64', n_classes=1000,
                 num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                 D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
                 SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
                 D_init='ortho', skip_init=False, D_param='SN', decoder_skip_connection=True, **kwargs):
        super(Unet_Discriminator, self).__init__()

        self.ch = D_ch
        self.D_wide = D_wide
        self.resolution = resolution
        self.kernel_size = D_kernel_size
        self.attention = D_attn
        self.n_classes = n_classes
        self.activation = D_activation
        self.init = D_init
        self.D_param = D_param
        self.SN_eps = SN_eps
        self.fp16 = D_fp16

        if self.resolution == 128:
            self.save_features = [0, 1, 2, 3, 4]
        elif self.resolution == 256:
            self.save_features = [0, 1, 2, 3, 4, 5]
        elif self.resolution == 512:
            self.save_features = [0, 1, 2, 3, 4, 5, 6]
        self.out_channel_multiplier = 1

        self.arch = D_unet_arch(input_c, self.ch, self.attention, out_channel_multiplier=self.out_channel_multiplier)[
            resolution]
        self.unconditional = True

        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d, kernel_size=3, padding=1, num_svs=num_D_SVs,
                                                num_itrs=num_D_SV_itrs, eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear, num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                  eps=self.SN_eps)
            self.which_embedding = functools.partial(layers.SNEmbedding, num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                     eps=self.SN_eps)

        self.blocks = []

        for index in range(len(self.arch['out_channels'])):
            if self.arch["downsample"][index]:
                self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                               out_channels=self.arch['out_channels'][index],
                                               which_conv=self.which_conv,
                                               wide=self.D_wide,
                                               activation=self.activation,
                                               preactivation=(index > 0),
                                               downsample=(
                                                   nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
            elif self.arch["upsample"][index]:
                upsample_function = (
                    functools.partial(F.interpolate, scale_factor=2, mode="nearest") if self.arch['upsample'][
                        index] else None)
                self.blocks += [[layers.GBlock2(in_channels=self.arch['in_channels'][index],
                                                out_channels=self.arch['out_channels'][index],
                                                which_conv=self.which_conv,
                                                activation=self.activation,
                                                upsample=upsample_function, skip_connection=True)]]

            attention_condition = index < 5
            if self.arch['attention'][self.arch['resolution'][index]] and attention_condition:
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        last_layer = nn.Conv2d(self.ch * self.out_channel_multiplier, 1, kernel_size=1)
        self.blocks.append(last_layer)

        self.linear_middle = self.which_linear(32 * self.ch if resolution == 512 else 16 * self.ch, output_dim)
        # self.sigmoid = nn.Sigmoid()

        if not self.unconditional:
            self.embed_middle = self.which_embedding(self.n_classes, 16 * self.ch)
            self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])



        if not skip_init:
            self.init_weights()

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                self.param_count += sum([p.data.nelement() for p in module.parameters()])

    def forward(self, x, y=None):
        h = x
        residual_features = []
        residual_features.append(x)

        for index, blocklist in enumerate(self.blocks[:-1]):
            if self.resolution == 128:
                if index == 6:
                    h = torch.cat((h, residual_features[4]), dim=1)
                elif index == 7:
                    h = torch.cat((h, residual_features[3]), dim=1)
                elif index == 8:
                    h = torch.cat((h, residual_features[2]), dim=1)
                elif index == 9:
                    h = torch.cat((h, residual_features[1]), dim=1)
            elif self.resolution == 256:
                if index == 7:
                    h = torch.cat((h, residual_features[5]), dim=1)
                elif index == 8:
                    h = torch.cat((h, residual_features[4]), dim=1)
                elif index == 9:
                    h = torch.cat((h, residual_features[3]), dim=1)
                elif index == 10:
                    h = torch.cat((h, residual_features[2]), dim=1)
                elif index == 11:
                    h = torch.cat((h, residual_features[1]), dim=1)
            elif self.resolution == 512:
                if index == 8:
                    h = torch.cat((h, residual_features[6]), dim=1)
                elif index == 9:
                    h = torch.cat((h, residual_features[5]), dim=1)
                elif index == 10:
                    h = torch.cat((h, residual_features[4]), dim=1)
                elif index == 11:
                    h = torch.cat((h, residual_features[3]), dim=1)
                elif index == 12:
                    h = torch.cat((h, residual_features[2]), dim=1)
                elif index == 13:
                    h = torch.cat((h, residual_features[1]), dim=1)

            for block in blocklist:
                h = block(h)

            if index in self.save_features[:-1]:
                residual_features.append(h)

            if index == self.save_features[-1]:


                h_ = torch.sum(self.activation(h), [2, 3])
                bottleneck_out = self.linear_middle(h_)
                # bottleneck_out = self.sigmoid(bottleneck_out)
                if not self.unconditional:
                    emb_mid = self.embed_middle(y)
                    projection = torch.sum(emb_mid * h_, 1, keepdim=True)
                    bottleneck_out = bottleneck_out + projection

        out = self.blocks[-1](h)

        if not self.unconditional:
            emb = self.embed(y)
            emb = emb.view(emb.size(0), emb.size(1), 1, 1).expand_as(h)
            proj = torch.sum(emb * h, 1, keepdim=True)
            out = out + proj

        out = out.view(out.size(0), 1, self.resolution, self.resolution)

        return out, bottleneck_out
