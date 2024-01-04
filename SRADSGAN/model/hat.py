import argparse
import os
import numpy as np
import time
import math
import itertools
import sys
from collections import OrderedDict
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from skimage.measure import compare_ssim
from skimage.measure import compare_mse
from skimage.measure import compare_psnr
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.models import vgg19

from utils import utils as srutils
from utils.logger import Logger, PrintLogger
from data import *
# from loss import GANLoss
# from architecture import RRDBNet, Discriminator_VGG_256, VGGFeatureExtractor
# from model.base_networks import *
from torch.nn import Parameter
from data.data import get_training_datasets, get_test_datasets, get_RGB_trainDataset, get_RGB_testDataset
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from einops import rearrange

# from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def weights_init_normal(m, mean=0.0, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        vgg19_model = vgg19(pretrained=True)

        # Extracts features at the 11th layer
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:12])

    def forward(self, img):
        out = self.feature_extractor(img)
        return out



def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class HAB(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.conv_scale = conv_scale
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask):
        h, w = x_size
        b, _, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # Conv_X
        conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = attn_mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x

class OCAB(nn.Module):
    # overlapping cross-attention block
    def __init__(self, dim,
                input_resolution,
                window_size,
                overlap_ratio,
                num_heads,
                qkv_bias=True,
                qk_scale=None,
                mlp_ratio=2,
                norm_layer=nn.LayerNorm
                ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3,  bias=qkv_bias)
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim,dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

    def forward(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2) # 3, b, c, h, w
        q = qkv[0].permute(0, 2, 3, 1) # b, h, w, c
        kv = torch.cat((qkv[1], qkv[2]), dim=1) # b, 2*c, h, w

        # partition windows
        q_windows = window_partition(q, self.window_size)  # nw*b, window_size, window_size, c
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        kv_windows = self.unfold(kv) # b, c*w*w, nw
        kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous() # 2, nw*b, ow*ow, c
        k_windows, v_windows = kv_windows[0], kv_windows[1] # nw*b, ow*ow, c

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, nq, d
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1)  # ws*ws, wse*wse, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
        x = x.view(b, h * w, self.dim)

        x = self.proj(x) + shortcut

        x = x + self.mlp(self.norm2(x))
        return x

class AttenBlocks(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            HAB(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # OCAB
        self.overlap_attn = OCAB(
                            dim=dim,
                            input_resolution=input_resolution,
                            window_size=window_size,
                            overlap_ratio=overlap_ratio,
                            num_heads=num_heads,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            mlp_ratio=mlp_ratio,
                            norm_layer=norm_layer
                            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, params):
        for blk in self.blocks:
            x = blk(x, x_size, params['rpi_sa'], params['attn_mask'])

        x = self.overlap_attn(x, x_size, params['rpi_oca'])

        if self.downsample is not None:
            x = self.downsample(x)
        return x

class RHAG(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv'):
        super(RHAG, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = AttenBlocks(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv = nn.Identity()

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size, params):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

class Upsample(nn.Sequential):
    def __init__(self, upscale_factor):
        super(Upsample, self).__init__()

        # Upsampling layers
        upsampling = []
        upsampling_two = [nn.Conv2d(64, 64 * 4, 3, 1, 1),
                          nn.PixelShuffle(upscale_factor=2)]
        upsampling_three = [nn.Conv2d(64, 64 * 9, 3, 1, 1),
                            nn.PixelShuffle(upscale_factor=3)]
        if (upscale_factor & (upscale_factor - 1)) == 0:  # Is scale = 2^n
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling += upsampling_two
        elif upscale_factor % 3 == 0:
            for _ in range(int(math.log(upscale_factor, 3))):
                upsampling += upsampling_three

        self.upsampling = nn.Sequential(*upsampling)

    def forward(self, x):
        out = self.upsampling(x)
        return out

class GeneratorResNet(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6, 6, 6),
                 # window_size=8,    #when scale =3,9
                 window_size=9,      #when scale =2,4,8
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 overlap_ratio=0.5,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='pixelshuffle',
                 resi_connection='1conv',
                 **kwargs):
        super(GeneratorResNet, self).__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        relative_position_index_OCA = self.calculate_rpi_oca()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)
        self.register_buffer('relative_position_index_OCA', relative_position_index_OCA)

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Hybrid Attention Groups (RHAG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHAG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv_after_body = nn.Identity()

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_sa(self):
        # calculate relative position index for SA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def calculate_rpi_oca(self):
        # calculate relative position index for OCA
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]   # 2, ws*ws, wse*wse

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1

        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])

        # Calculate attention mask and relative position index in advance to speed up inference.
        # The original code is very time-cosuming for large window size.
        attn_mask = self.calculate_mask(x_size).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA, 'rpi_oca': self.relative_position_index_OCA}

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size, params)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean

        return x

class HAT(object):
    def __init__(self, args):
        # parameters
        self.model_name = args.model_name
        self.train_dataset = args.train_dataset
        self.test_dataset = args.test_dataset
        self.crop_size = args.crop_size
        self.test_crop_size = args.test_crop_size
        self.hr_height = args.hr_height
        self.hr_width = args.hr_width
        self.num_threads = args.num_threads
        self.num_channels = args.num_channels
        self.scale_factor = args.scale_factor
        self.epoch = args.epoch
        self.num_epochs = args.num_epochs
        self.save_epochs = args.save_epochs
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.lr = args.lr
        self.b1 = args.b1
        self.b2 = args.b2
        self.data_dir = args.data_dir
        self.root_dir = args.root_dir
        self.save_dir = args.save_dir
        self.gpu_mode = args.gpu_mode
        self.n_cpu = args.n_cpu
        self.sample_interval = args.sample_interval

        self.clip_value = args.clip_value
        self.lambda_gp = args.lambda_gp
        self.gp = args.gp
        self.penalty_type = args.penalty_type
        self.grad_penalty_Lp_norm = args.grad_penalty_Lp_norm
        self.relative = args.relativeGan
        self.loss_Lp_norm = args.loss_Lp_norm
        self.weight_gan = args.weight_gan
        self.weight_content = args.weight_content
        self.max_train_samples = args.max_train_samples

        self.cuda = True if torch.cuda.is_available() else False
        # import PerceptualSimilarity
        from utils import PerceptualSimilarity
        self.PerceptualModel = PerceptualSimilarity.PerceptualLoss(model='net-lin', net='alex', use_gpu=self.cuda)

        self.log_dict = OrderedDict()

        # set the logger
        log_dir = os.path.join(self.root_dir, self.save_dir, 'logs')
        log_freq = 200
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.logger = Logger(self.model_name, log_dir, log_freq)

    def load_dataset(self, dataset='train', max_samples=20000):
        if self.num_channels == 1:
            is_gray = True
        else:
            is_gray = False

        if dataset == 'train':
            print('Loading train dct_datasets...')
            train_set = get_RGB_trainDataset(self.data_dir, self.train_dataset, self.crop_size, self.scale_factor, is_gray=is_gray)
            return DataLoader(dataset=train_set, num_workers=self.num_threads, batch_size=self.batch_size, shuffle=True, drop_last=True)
        elif dataset == 'test':
            print('Loading test dct_datasets...')
            test_set = get_RGB_testDataset(self.data_dir, self.test_dataset, self.crop_size, self.scale_factor, is_gray=is_gray)
            return DataLoader(dataset=test_set, num_workers=self.num_threads, batch_size=self.test_batch_size, shuffle=False, drop_last=True)

    def train(self):
        device_ids = [0]
        cuda = True if torch.cuda.is_available() else False
        self.device = torch.device('cuda' if cuda else 'cpu')

        # Initialize generator and discriminator

        self.generator = GeneratorResNet(upscale=self.scale_factor, in_chans=3, img_size=int(self.crop_size / self.scale_factor))
        self.feature_extractor = FeatureExtractor()
        # self.feature_extractor = VGGFeatureExtractor(feature_layer=34, use_bn=False, use_input_norm=True, device=self.device)

        # srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='Generator', input_size=(1,3,int(256/self.scale_factor),int(256/self.scale_factor)))
        srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='Generator')
        srutils.print_network_to_file(self.feature_extractor, save_dir=self.save_dir, tag='FeatureExtractor')


        # Losses
        # self.criterion_GAN = torch.nn.MSELoss().to(self.device)
        if self.loss_Lp_norm=="L1":
            self.criterion_content = torch.nn.L1Loss().to(self.device)
        else:
            self.criterion_content = torch.nn.MSELoss().to(self.device)

        if len(device_ids) > 1:
            self.generator = nn.DataParallel(self.generator).to(self.device)
            self.feature_extractor = nn.DataParallel(self.feature_extractor).to(self.device)

        if cuda:
            self.generator = self.generator.cuda(device_ids[0])
            self.feature_extractor = self.feature_extractor.cuda(device_ids[0])
            # self.criterion_GAN = self.criterion_GAN.cuda(device_ids[0])
            self.criterion_content = self.criterion_content.cuda(device_ids[0])

        model_dir = os.path.join(self.save_dir, 'model')
        if self.epoch != 0:
            # Load pretrained models
            G_model_path = model_dir+'/generator_param_epoch_%d.pkl' % self.epoch
            self.load_epoch_network(load_path=G_model_path, network=self.generator, strict=True)
        else:
            # Initialize weights
            self.generator.apply(srutils.weights_init_normal)

        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.batch_size, self.num_channels, self.crop_size // self.scale_factor, self.crop_size // self.scale_factor)
        input_hr = Tensor(self.batch_size, self.num_channels, self.crop_size, self.crop_size)
        input_bc = Tensor(self.batch_size, self.num_channels, self.crop_size, self.crop_size)
        #load train dataset
        dataloader = self.load_dataset(dataset='train', max_samples=self.max_train_samples)

        sys.stdout = PrintLogger(self.save_dir)
        # ----------
        #  Training
        # ----------
        print('Training is started.')
        avg_loss_G = []
        avg_loss_D = []
        # avg_mse = []
        avg_psnr = []
        avg_ssim = []
        avg_ergas = []
        avg_lpips = []
        step = 0
        start_time = time.time()

        tensor_to_img = ToPILImage()
        global val_psnr
        global val_ssim
        global val_ergas
        global val_lpips
        global val_psnr_max
        global val_ssim_max
        global val_ergas_max
        global val_lpips_max
        global val_loss_no_improve_count
        global val_loss_noimprove_max_count
        global val_loss_best_step
        val_psnr = 0
        val_ssim = 0
        val_ergas = 0
        val_lpips = 0
        val_psnr_max = 0
        val_ssim_max = 0
        val_ergas_max = 10000
        val_lpips_max = 10000
        val_loss_no_improve_count = 0
        val_loss_noimprove_max_count = 5
        val_loss_best_step = 0
        global epoch
        epoch = self.epoch
        while epoch < self.num_epochs and self.lr >= 0.00001:
        # for epoch in range(self.epoch, self.num_epochs):
            # learning rate is decayed by a factor of 10 every 20 epochs
            # if (epoch + 1) % 10 == 0:
            #     for param_group in optimizer_G.param_groups:
            #         param_group["lr"] /= 2.0
            #     print("Learning rate decay: lr={}".format(optimizer_G.param_groups[0]["lr"]))

            epoch_loss_G = 0
            for i, (input, target, bc_input, imgpath) in enumerate(dataloader):

                # Configure model input
                imgs_lr = Variable(input_lr.copy_(input))
                imgs_hr = Variable(input_hr.copy_(target))
                imgs_bc = Variable(input_bc.copy_(bc_input))

                # ------------------
                #  Train Generators
                # ------------------

                optimizer_G.zero_grad()

                # Generate a high resolution image from low resolution input
                gen_hr = self.generator(imgs_lr)
                # print(gen_hr.shape, imgs_hr.shape)   # 256, 216
                # pixel loss
                pixel_loss_G = self.criterion_content(gen_hr, imgs_hr)
                # Content loss
                gen_features = self.feature_extractor(gen_hr)
                real_features = Variable(self.feature_extractor(imgs_hr).data, requires_grad=False)
                loss_content = self.criterion_content(gen_features, real_features)

                # Total loss
                loss_G = pixel_loss_G

                loss_G.backward()
                optimizer_G.step()
                # optimizer_G.module.step()

                print("[Epoch %d/%d] [Batch %d/%d] [G loss: %f]" %
                      (epoch, self.num_epochs, i, len(dataloader), loss_G.item()))

                epoch_loss_G += loss_G.data.item()
                # tensorboard logging
                self.logger.scalar_summary('loss_G', loss_G.item(), step + 1)

                step += 1

                batches_done = epoch * len(dataloader) + i
                if batches_done % self.sample_interval == 0:
                    # Save image sample
                    # bc_imgs = srutils.img_interp(imgs_lr.cpu(), 4)
                    bc_img = tensor_to_img(imgs_bc[0].cpu())
                    bc_img = np.array(bc_img)

                    hr_img = tensor_to_img(imgs_hr[0].cpu())
                    hr_img = np.array(hr_img)

                    gen_img = tensor_to_img(gen_hr[0].cpu())
                    gen_img = np.array(gen_img)

                    # mse = compare_mse(gen_img, hr_img)
                    recon_psnr = compare_psnr(gen_img, hr_img)
                    recon_ssim = compare_ssim(gen_img, hr_img, multichannel=True)
                    recon_ergas = srutils.compare_ergas2(hr_img, gen_img, scale=self.scale_factor)
                    recon_lpips = self.PerceptualModel.forward(gen_hr[0], imgs_hr[0], normalize=True).detach().item()

                    # mse1 = compare_mse(bc_img, hr_img)
                    bc_psnr = compare_psnr(bc_img, hr_img)
                    bc_ssim = compare_ssim(bc_img, hr_img, multichannel=True)
                    bc_ergas = srutils.compare_ergas2(hr_img, bc_img, scale=self.scale_factor)
                    bc_lpips = self.PerceptualModel.forward(imgs_bc[0], imgs_hr[0], normalize=True).detach().item()

                    result_imgs = [imgs_hr[0].cpu(), imgs_lr[0].cpu(), imgs_bc[0].cpu(), gen_hr[0].cpu()]
                    psnrs = [None, None, bc_psnr, recon_psnr]
                    ssims = [None, None, bc_ssim, recon_ssim]
                    ergas = [None, None, bc_ergas, recon_ergas]
                    lpips = [None, None, bc_lpips, recon_lpips]
                    indicators = {"PSNR":psnrs, "SSIM":ssims, "ERGAS":ergas, "LPIPS":lpips}
                    srutils.plot_result_by_name(result_imgs, indicators, batches_done, imgpath[0],
                                                     save_dir=self.save_dir, is_training=True, show=False)
                    # srutils.plot_test_result_by_name(result_imgs, psnrs, ssims, batches_done, imgpath[0], save_dir=self.save_dir, is_training=True, show=False)
                    # save_image(torch.cat((gen_hr.data, imgs_hr.data), -2),
                    #          'images/%d.png' % batches_done, normalize=True)
                    # log
                    time_elapsed = time.time() - start_time
                    print_rlt = OrderedDict()
                    print_rlt['model'] = self.model_name
                    print_rlt['epoch'] = epoch
                    print_rlt['iters'] = step
                    # print_rlt['G_lr'] = optimizer_G.module.param_groups[0]['lr']
                    # print_rlt['D_lr'] = optimizer_D.module.param_groups[0]['lr']
                    print_rlt['G_lr'] = optimizer_G.param_groups[0]['lr']
                    print_rlt['time'] = time_elapsed
                    print_rlt['G_loss'] = loss_G.item()
                    print_rlt['bicubic_psnr'] = bc_psnr
                    print_rlt['bicubic_ssim'] = bc_ssim
                    print_rlt['bicubic_ergas'] = bc_ergas
                    print_rlt['bicubic_lpips'] = bc_lpips
                    print_rlt['srwgan_psnr'] = recon_psnr
                    print_rlt['srwgan_ssim'] = recon_ssim
                    print_rlt['srwgan_ergas'] = recon_ergas
                    print_rlt['srwgan_lpips'] = recon_lpips
                    for k, v in self.log_dict.items():
                        print_rlt[k] = v
                    self.logger.print_format_results('train', print_rlt)

                    del bc_img, hr_img, gen_img, recon_psnr, recon_ssim, recon_ergas, recon_lpips, bc_ssim, bc_psnr, bc_ergas, bc_lpips, result_imgs, psnrs, ssims
                del imgs_hr, imgs_bc, imgs_lr, gen_hr, gen_features, real_features
                torch.cuda.empty_cache()

            # avg. loss per epoch
            avg_loss_G.append(epoch_loss_G / len(dataloader))

            val_psnr, val_ssim, val_ergas, val_lpips = self.validate(epoch=epoch, save_img=((epoch + 1) % self.save_epochs == 0))

            avg_psnr.append(val_psnr)
            avg_ssim.append(val_ssim)
            avg_ergas.append(val_ergas)
            avg_lpips.append(val_lpips)

            if val_psnr - val_psnr_max > 0:
                val_psnr_max = val_psnr
                val_loss_no_improve_count = 0
                val_loss_best_step = epoch
            elif val_ssim - val_ssim_max > 0:
                val_ssim_max = val_ssim
                val_loss_no_improve_count = 0
                val_loss_best_step = epoch
            elif val_ergas - val_ergas_max < 0:
                val_ergas_max = val_ergas
                val_loss_no_improve_count = 0
                val_loss_best_step = epoch
            elif val_lpips - val_lpips_max < 0:
                val_lpips_max = val_lpips
                val_loss_no_improve_count = 0
                val_loss_best_step = epoch
            else:
                val_loss_no_improve_count = val_loss_no_improve_count + 1

            self.save_epoch_network(save_dir=model_dir, network=self.generator, network_label='generator',
                                    iter_label=epoch + 1)
            # global epoch
            epoch += 1
            if val_loss_no_improve_count >= val_loss_noimprove_max_count:
                G_model_path = model_dir + '/generator_param_epoch_%d.pkl' % (val_loss_best_step + 1)
                self.load_epoch_network(load_path=G_model_path, network=self.generator)
                # for param_group in optimizer_G.module.param_groups:
                #     param_group["lr"] /= 2.0
                # print("Learning rate decay: lr={}".format(optimizer_G.module.param_groups[0]["lr"]))
                for param_group in optimizer_G.param_groups:
                    param_group["lr"] /= 2.0
                print("optimizer_G_Learning rate decay: lr={}".format(optimizer_G.param_groups[0]["lr"]))
                # global epoch
                self.lr /= 2.0
                epoch = val_loss_best_step + 1
                val_loss_no_improve_count = 0
                for _ in range(0, val_loss_noimprove_max_count):
                    avg_psnr.pop()
                    avg_ssim.pop()
                    avg_ergas.pop()
                    avg_lpips.pop()
            # if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            #     # Save model checkpoints
            #     torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % epoch)
            # Save trained parameters of model
            # if (epoch + 1) % self.save_epochs == 0:
            #     # self.save_model(epoch + 1)
            #     self.save_epoch_network(save_dir=model_dir, network=self.generator, network_label='generator', iter_label=epoch+1)
            torch.cuda.empty_cache()
        # Plot avg. loss
        srutils.plot_loss([avg_loss_G], self.num_epochs, save_dir=self.save_dir)
        srutils.plot_loss([avg_psnr], self.num_epochs, save_dir=self.save_dir, label='PSNR')
        srutils.plot_loss([avg_ssim], self.num_epochs, save_dir=self.save_dir, label='SSIM')
        srutils.plot_loss([avg_ergas], self.num_epochs, save_dir=self.save_dir, label='ERGAS')
        srutils.plot_loss([avg_lpips], self.num_epochs, save_dir=self.save_dir, label='LPIPS')
        print("Training is finished.")

        # Save final trained parameters of model
        self.save_model(epoch=None)

    def validate(self, epoch=0, save_img=False):
        # networks
        cuda = True if torch.cuda.is_available() else False

        # load model
        if self.gpu_mode:
            self.generator.cuda()

        # load dataset
        test_data_loader = self.load_dataset(dataset='test')

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.test_batch_size, self.num_channels, self.hr_height // self.scale_factor, self.hr_width // self.scale_factor)
        input_hr = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        input_bc = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        tensor_to_img = ToPILImage()
        # Test
        print('Test is started.')
        img_num = 0
        self.generator.eval()
        start_time = time.time()
        avg_bicubic_mse = 0.0
        avg_bicubic_psnr = 0.0
        avg_bicubic_ssim = 0.0
        avg_bicubic_ergas = 0.0
        avg_bicubic_lpips = 0.0
        avg_srcnn_mse = 0.0
        avg_srcnn_psnr = 0.0
        avg_srcnn_ssim = 0.0
        avg_srcnn_ergas = 0.0
        avg_srcnn_lpips = 0.0
        for iter, (input, target, bc_input, imgpath) in enumerate(test_data_loader):
            # Configure model input
            imgs_lr = Variable(input_lr.copy_(input))
            imgs_hr = Variable(input_hr.copy_(target))
            imgs_bc = Variable(input_bc.copy_(bc_input))

            # prediction
            recon_imgs = self.generator(imgs_lr)
            for i in range(self.test_batch_size):
                img_num += 1
                recon_img = recon_imgs[i].cpu().data
                gt_img = imgs_hr[i]  # utils.shave(target[i], border_size=8)
                # lr_img = imgs_lr[i]
                bc_img = imgs_bc[i]  # srutils.img_interp(lr_img.cpu(), 4)

                # calculate psnrs
                bc_img = tensor_to_img(bc_img.cpu())
                bc_img = np.array(bc_img)

                gt_img = tensor_to_img(gt_img.cpu())
                gt_img = np.array(gt_img)

                recon_img = tensor_to_img(recon_img.cpu())
                recon_img = np.array(recon_img)
                recon_mse = compare_mse(recon_img, gt_img)
                recon_psnr = compare_psnr(recon_img, gt_img)
                recon_ssim = compare_ssim(recon_img, gt_img, multichannel=True)
                recon_ergas = srutils.compare_ergas2(gt_img, recon_img, scale=self.scale_factor)
                # recon_lpips = srutils.compare_lpips(gt_img, recon_img, use_gpu=cuda)
                recon_lpips = self.PerceptualModel.forward(recon_imgs[i], imgs_hr[i], normalize=True).detach().item()

                bc_mse = compare_mse(bc_img, gt_img)
                bc_psnr = compare_psnr(bc_img, gt_img)
                bc_ssim = compare_ssim(bc_img, gt_img, multichannel=True)
                bc_ergas = srutils.compare_ergas2(gt_img, bc_img, scale=self.scale_factor)
                # bc_lpips = srutils.compare_lpips(gt_img, bc_img, use_gpu=cuda)
                bc_lpips = self.PerceptualModel.forward(imgs_bc[i], imgs_hr[i], normalize=True).detach().item()

                if save_img and iter % 50 == 0:
                    # save result images
                    result_imgs = [imgs_hr[i].cpu(), imgs_lr[i].cpu(), imgs_bc[i].cpu(), recon_imgs[i].cpu()]
                    psnrs = [None, None, bc_psnr, recon_psnr]
                    ssims = [None, None, bc_ssim, recon_ssim]
                    ergas = [None, None, bc_ergas, recon_ergas]
                    lpips = [None, None, bc_lpips, recon_lpips]
                    indicators = {"PSNR": psnrs, "SSIM": ssims, "ERGAS": ergas, "LPIPS": lpips}
                    # srutils.plot_test_result_by_name(result_imgs, psnrs, ssims, epoch, imgpath[i],
                                                     # save_dir=self.save_dir)
                    srutils.plot_result_by_name(result_imgs, indicators, epoch, imgpath[i],
                                                save_dir=self.save_dir)

                avg_bicubic_mse += bc_mse
                avg_bicubic_psnr += bc_psnr
                avg_bicubic_ssim += bc_ssim
                avg_bicubic_ergas += bc_ergas
                avg_bicubic_lpips += bc_lpips
                avg_srcnn_mse += recon_mse
                avg_srcnn_psnr += recon_psnr
                avg_srcnn_ssim += recon_ssim
                avg_srcnn_ergas += recon_ergas
                avg_srcnn_lpips += recon_lpips
                print("Saving %d test result images..." % img_num)

                del bc_img, gt_img, recon_img, recon_psnr, recon_ssim, recon_mse, recon_ergas, recon_lpips, bc_mse, bc_ssim, bc_psnr, bc_ergas, bc_lpips
            del imgs_hr, imgs_bc, imgs_lr, recon_imgs
            torch.cuda.empty_cache()

        avg_bicubic_mse = avg_bicubic_mse / img_num
        avg_bicubic_psnr = avg_bicubic_psnr / img_num
        avg_bicubic_ssim = avg_bicubic_ssim / img_num
        avg_bicubic_ergas = avg_bicubic_ergas / img_num
        avg_bicubic_lpips = avg_bicubic_lpips / img_num
        avg_srcnn_mse = avg_srcnn_mse / img_num
        avg_srcnn_psnr = avg_srcnn_psnr / img_num
        avg_srcnn_ssim = avg_srcnn_ssim / img_num
        avg_srcnn_ergas = avg_srcnn_ergas / img_num
        avg_srcnn_lpips = avg_srcnn_lpips / img_num
        time_elapsed = time.time() - start_time
        # Save to log
        print_rlt = OrderedDict()
        print_rlt['model'] = self.model_name
        print_rlt['epoch'] = epoch
        print_rlt['iters'] = epoch
        print_rlt['time'] = time_elapsed
        print_rlt['bicubic_mse'] = avg_bicubic_mse
        print_rlt['bicubic_psnr'] = avg_bicubic_psnr
        print_rlt['bicubic_ssim'] = avg_bicubic_ssim
        print_rlt['bicubic_ergas'] = avg_bicubic_ergas
        print_rlt['bicubic_lpips'] = avg_bicubic_lpips
        print_rlt['srcnn_mse'] = avg_srcnn_mse
        print_rlt['srcnn_psnr'] = avg_srcnn_psnr
        print_rlt['srcnn_ssim'] = avg_srcnn_ssim
        print_rlt['srcnn_ergas'] = avg_srcnn_ergas
        print_rlt['srcnn_lpips'] = avg_srcnn_lpips
        self.logger.print_format_results('val', print_rlt)
        print('-----------------------------------')
        del test_data_loader
        torch.cuda.empty_cache()
        return avg_srcnn_psnr, avg_srcnn_ssim, avg_srcnn_ergas, avg_srcnn_lpips

    # helper saving function that can be used by subclasses
    def save_epoch_network(self, save_dir, network, network_label, iter_label):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_filename = '{}_param_epoch_{}.pkl'.format(network_label, iter_label)
        save_path = os.path.join(save_dir, save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # helper loading function that can be used by subclasses
    def load_epoch_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path), strict=strict)
        print('Trained model is loaded.')

    def save_model(self, epoch=None):
        model_dir = os.path.join(self.save_dir, 'model')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if epoch is not None:
            torch.save(self.generator.state_dict(), model_dir + '/generator_param_epoch_%d.pkl' % epoch)
        else:
            torch.save(self.generator.state_dict(), model_dir + '/generator_param.pkl')
        print('Trained model is saved.')

    def load_model(self):
        model_dir = os.path.join(self.save_dir, 'model')
        model_name = model_dir + '/generator_param.pkl'
        if os.path.exists(model_name):
            self.generator.load_state_dict(torch.load(model_name), strict=False)
            print('Trained model is loaded.')
            return True
        else:
            print('No model exists to load.')
            return False

    def load_epoch_model(self, epoch):
        model_dir = os.path.join(self.save_dir, 'model')
        model_name = model_dir + '/generator_param_epoch_%d.pkl' % epoch
        if os.path.exists(model_name):
            self.generator.load_state_dict(torch.load(model_name))
            print('Trained model is loaded.')
            return True
        else:
            print('No model exists to load.')
            return False

    def mfeNew_validate(self, epoch=100, modelpath=None):
        # networks
        cuda = True if torch.cuda.is_available() else False

        # load model
        self.generator = GeneratorResNet(upscale=self.scale_factor, in_chans=3, img_size=int(self.crop_size / self.scale_factor))
        if self.gpu_mode:
            self.generator.cuda()

        if modelpath is not None:
            self.generator.load_state_dict(torch.load(modelpath), strict=False)

        # self.generator.eval()

        # load dataset
        test_data_loader = self.load_dataset(dataset='test')

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.test_batch_size, self.num_channels, self.hr_height // self.scale_factor, self.hr_width // self.scale_factor)
        input_hr = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        input_bc = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        tensor_to_img = ToPILImage()
        # Test
        print('Test is started.')
        img_num = 0
        self.generator.eval()
        start_time = time.time()
        avg_bicubic_mse = 0.0
        avg_bicubic_psnr = 0.0
        avg_bicubic_ssim = 0.0
        avg_bicubic_ergas = 0.0
        avg_bicubic_lpips = 0.0
        avg_srcnn_mse = 0.0
        avg_srcnn_psnr = 0.0
        avg_srcnn_ssim = 0.0
        avg_srcnn_ergas = 0.0
        avg_srcnn_lpips = 0.0
        for iter, (input, target, bc_input, imgpath) in enumerate(test_data_loader):
            # Configure model input
            imgs_lr = Variable(input_lr.copy_(input))
            imgs_hr = Variable(input_hr.copy_(target))
            imgs_bc = Variable(input_bc.copy_(bc_input))

            # prediction
            recon_imgs = self.generator(imgs_lr)
            for i in range(self.test_batch_size):
                img_num += 1
                recon_img = recon_imgs[i].cpu().data
                gt_img = imgs_hr[i]  # utils.shave(target[i], border_size=8)
                # lr_img = imgs_lr[i]
                bc_img = imgs_bc[i]  # srutils.img_interp(lr_img.cpu(), 4)

                # calculate psnrs
                bc_img = tensor_to_img(bc_img.cpu())
                bc_img = np.array(bc_img)

                gt_img = tensor_to_img(gt_img.cpu())
                gt_img = np.array(gt_img)

                recon_img = tensor_to_img(recon_img.cpu())
                recon_img = np.array(recon_img)
                recon_mse = compare_mse(recon_img, gt_img)
                recon_psnr = compare_psnr(recon_img, gt_img)
                recon_ssim = compare_ssim(recon_img, gt_img, multichannel=True)
                recon_ergas = srutils.compare_ergas2(gt_img, recon_img, scale=self.scale_factor)
                recon_lpips = self.PerceptualModel.forward(recon_imgs[i], imgs_hr[i], normalize=True).detach().item()

                bc_mse = compare_mse(bc_img, gt_img)
                bc_psnr = compare_psnr(bc_img, gt_img)
                bc_ssim = compare_ssim(bc_img, gt_img, multichannel=True)
                bc_ergas = srutils.compare_ergas2(gt_img, bc_img, scale=self.scale_factor)
                bc_lpips = self.PerceptualModel.forward(imgs_bc[i], imgs_hr[i], normalize=True).detach().item()



                result_imgs = [imgs_hr[i].cpu(), imgs_lr[i].cpu(), imgs_bc[i].cpu(), recon_imgs[i].cpu()]
                mses = [None, None, bc_mse, recon_mse]
                psnrs = [None, None, bc_psnr, recon_psnr]
                ssims = [None, None, bc_ssim, recon_ssim]
                ergas = [None, None, bc_ergas, recon_ergas]
                lpips = [None, None, bc_lpips, recon_lpips]
                srutils.mfe_plot_test_result2(result_imgs, mses, psnrs, ssims, ergas, lpips, img_num, save_dir=self.save_dir)

                avg_bicubic_mse += bc_mse
                avg_bicubic_psnr += bc_psnr
                avg_bicubic_ssim += bc_ssim
                avg_bicubic_ergas += bc_ergas
                avg_bicubic_lpips += bc_lpips
                avg_srcnn_mse += recon_mse
                avg_srcnn_psnr += recon_psnr
                avg_srcnn_ssim += recon_ssim
                avg_srcnn_ergas += recon_ergas
                avg_srcnn_lpips += recon_lpips
                print("Saving the %d test dataset images" % img_num)

                del bc_img, gt_img, recon_img, recon_psnr, recon_ssim, recon_mse, recon_ergas, recon_lpips, bc_mse, bc_ssim, bc_psnr, bc_ergas, bc_lpips
            del imgs_hr, imgs_bc, imgs_lr, recon_imgs
            torch.cuda.empty_cache()

        avg_bicubic_mse = avg_bicubic_mse / img_num
        avg_bicubic_psnr = avg_bicubic_psnr / img_num
        avg_bicubic_ssim = avg_bicubic_ssim / img_num
        avg_bicubic_ergas = avg_bicubic_ergas / img_num
        avg_bicubic_lpips = avg_bicubic_lpips / img_num
        avg_srcnn_mse = avg_srcnn_mse / img_num
        avg_srcnn_psnr = avg_srcnn_psnr / img_num
        avg_srcnn_ssim = avg_srcnn_ssim / img_num
        avg_srcnn_ergas = avg_srcnn_ergas / img_num
        avg_srcnn_lpips = avg_srcnn_lpips / img_num
        time_elapsed = time.time() - start_time
        # Save to log
        print_rlt = OrderedDict()
        print_rlt['model'] = self.model_name
        print_rlt['epoch'] = epoch
        print_rlt['iters'] = epoch
        print_rlt['time'] = time_elapsed
        print_rlt['bicubic_mse'] = avg_bicubic_mse
        print_rlt['bicubic_psnr'] = avg_bicubic_psnr
        print_rlt['bicubic_ssim'] = avg_bicubic_ssim
        print_rlt['bicubic_ergas'] = avg_bicubic_ergas
        print_rlt['bicubic_lpips'] = avg_bicubic_lpips
        print_rlt['hat_mse'] = avg_srcnn_mse
        print_rlt['hat_psnr'] = avg_srcnn_psnr
        print_rlt['hat_ssim'] = avg_srcnn_ssim
        print_rlt['hat_ergas'] = avg_srcnn_ergas
        print_rlt['hat_lpips'] = avg_srcnn_lpips
        self.logger.print_format_results('val', print_rlt)
        print('-----------------------------------')
        del test_data_loader
        torch.cuda.empty_cache()
        return avg_srcnn_psnr, avg_srcnn_ssim, avg_srcnn_ergas, avg_srcnn_lpips

    def mfeNew_validateByClass(self, epoch, save_img=False, modelpath=None):
        # networks
        cuda = True if torch.cuda.is_available() else False
        # load model
        self.generator = GeneratorResNet(upscale=self.scale_factor, in_chans=3, img_size=int(self.crop_size / self.scale_factor))

        if modelpath is not None:
            self.generator.load_state_dict(torch.load(modelpath), strict=False)
        srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='HAT')

        if self.gpu_mode:
            self.generator.cuda()

        self.generator.eval()
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.test_batch_size, self.num_channels, self.hr_height // self.scale_factor, self.hr_width // self.scale_factor)
        input_hr = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        input_bc = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        # Test
        print('Test is started.')
        tensor_to_img = ToPILImage()
        start_time = time.time()
        average_bicubic_psnr = 0.0
        average_bicubic_ssim = 0.0
        average_bicubic_mse = 0.0
        average_bicubic_ergas = 0.0
        average_bicubic_lpips = 0.0
        average_srrgan_mse = 0.0
        average_srrgan_psnr = 0.0
        average_srrgan_ssim = 0.0
        average_srrgan_ergas = 0.0
        average_srrgan_lpips = 0.0
        num = 0
        # create train and val dataloader
        testDataset_dirs = []
        dataset = self.test_dataset[0]
        if dataset == 'UCMerced_LandUse':
            file_path = os.path.join(self.data_dir, dataset)
            dirs = sorted(os.listdir(file_path))
            for dir in dirs:
                dir_path = os.path.join(file_path, dir)
                if os.path.isdir(dir_path):
                    testDataset_dirs.append(dir_path)  # 最后一层类别文件夹
        for datasetDir in testDataset_dirs:
            datasetDir_list = []
            datasetDir_list.append(datasetDir)
            test_set = get_RGB_testDataset(self.data_dir, datasetDir_list, self.crop_size, self.scale_factor,
                                    is_gray=False)
            test_data_loader = DataLoader(dataset=test_set, num_workers=self.num_threads, batch_size=self.test_batch_size, shuffle=False, drop_last=True)
            print('Number of val images in [{:s}]: {:d}'.format(datasetDir, len(test_data_loader)))
            datasetName = os.path.split(datasetDir)[-1]

            avg_bicubic_psnr = 0.0
            avg_bicubic_ssim = 0.0
            avg_bicubic_mse = 0.0
            avg_bicubic_ergas = 0.0
            avg_bicubic_lpips = 0.0
            avg_srcnn_mse = 0.0
            avg_srcnn_psnr = 0.0
            avg_srcnn_ssim = 0.0
            avg_srcnn_ergas = 0.0
            avg_srcnn_lpips = 0.0
            img_num = 0
            for i, (input, target, bc_input, imgpath) in enumerate(test_data_loader):
                # Configure model input
                imgs_lr = Variable(input_lr.copy_(input))
                imgs_hr = Variable(input_hr.copy_(target))
                imgs_bc = Variable(input_bc.copy_(bc_input))

                # prediction
                recon_imgs = self.generator(imgs_lr)
                for i in range(self.test_batch_size):
                    img_num += 1
                    num += 1
                    recon_img = recon_imgs[i].cpu().data
                    gt_img = imgs_hr[i]  # utils.shave(target[i], border_size=8)
                    lr_img = imgs_lr[i]
                    bc_img = imgs_bc[i]  # srutils.img_interp(lr_img.cpu(), 4)

                    # calculate psnrs
                    bc_img = tensor_to_img(bc_img.cpu())
                    bc_img = np.array(bc_img)

                    gt_img = tensor_to_img(gt_img.cpu())
                    gt_img = np.array(gt_img)

                    recon_img = tensor_to_img(recon_img.cpu())
                    recon_img = np.array(recon_img)
                    recon_mse = compare_mse(recon_img, gt_img)
                    recon_psnr = compare_psnr(recon_img, gt_img)
                    recon_ssim = compare_ssim(recon_img, gt_img, multichannel=True)
                    recon_ergas = srutils.compare_ergas2(gt_img, recon_img, scale=self.scale_factor)
                    # recon_lpips = srutils.compare_lpips(gt_img, recon_img, use_gpu=cuda)
                    recon_lpips = self.PerceptualModel.forward(recon_imgs[i], imgs_hr[i], normalize=True).detach().item()

                    bc_mse = compare_mse(bc_img, gt_img)
                    bc_psnr = compare_psnr(bc_img, gt_img)
                    bc_ssim = compare_ssim(bc_img, gt_img, multichannel=True)
                    bc_ergas = srutils.compare_ergas2(gt_img, bc_img, scale=self.scale_factor)
                    # bc_lpips = srutils.compare_lpips(gt_img, bc_img, use_gpu=cuda)
                    bc_lpips = self.PerceptualModel.forward(imgs_bc[i], imgs_hr[i], normalize=True).detach().item()

                    if save_img:
                        # save result images
                        result_imgs = [imgs_hr[i].cpu(), imgs_lr[i].cpu(), imgs_bc[i].cpu(), recon_imgs[i].cpu()]
                        mses = [None, None, bc_mse, recon_mse]
                        psnrs = [None, None, bc_psnr, recon_psnr]
                        ssims = [None, None, bc_ssim, recon_ssim]
                        ergas = [None, None, bc_ergas, recon_ergas]
                        lpips = [None, None, bc_lpips, recon_lpips]
                        indicators = {"PSNR": psnrs, "SSIM": ssims, "ERGAS": ergas, "LPIPS": lpips}
                        srutils.plot_result_by_name(result_imgs, indicators, epoch, imgpath[i], is_training=False,
                                                    is_validation=True, classname=datasetName, save_dir=self.save_dir)
                        img_name = os.path.splitext(os.path.basename(imgpath[i]))[0]
                        img_dir = os.path.join(self.save_dir, 'validate', datasetName, '{:s}_x{:d}_{:d}.png'.format(img_name, self.scale_factor, epoch))
                        srutils.save_img1(recon_imgs[i].cpu(), os.path.join(self.save_dir, 'validate'), img_dir, cuda=cuda)

                    avg_bicubic_mse += bc_mse
                    avg_bicubic_psnr += bc_psnr
                    avg_bicubic_ssim += bc_ssim
                    avg_bicubic_ergas += bc_ergas
                    avg_bicubic_lpips += bc_lpips
                    avg_srcnn_mse += recon_mse
                    avg_srcnn_psnr += recon_psnr
                    avg_srcnn_ssim += recon_ssim
                    avg_srcnn_ergas += recon_ergas
                    avg_srcnn_lpips += recon_lpips

                    average_bicubic_mse += bc_mse
                    average_bicubic_psnr += bc_psnr
                    average_bicubic_ssim += bc_ssim
                    average_bicubic_ergas += bc_ergas
                    average_bicubic_lpips += bc_lpips
                    average_srrgan_mse += recon_mse
                    average_srrgan_psnr += recon_psnr
                    average_srrgan_ssim += recon_ssim
                    average_srrgan_ergas += recon_ergas
                    average_srrgan_lpips += recon_lpips
                    # loss_mse += mse
                    print("Saving %d test result images..." % img_num)

                    del bc_img, gt_img, recon_img, recon_mse, recon_psnr, recon_ssim, recon_ergas, recon_lpips, bc_mse, bc_ssim, bc_psnr, bc_ergas, bc_lpips
                del imgs_hr, imgs_bc, imgs_lr, recon_imgs

            avg_bicubic_mse = avg_bicubic_mse / img_num
            avg_bicubic_psnr = avg_bicubic_psnr / img_num
            avg_bicubic_ssim = avg_bicubic_ssim / img_num
            avg_bicubic_ergas = avg_bicubic_ergas / img_num
            avg_bicubic_lpips = avg_bicubic_lpips / img_num
            avg_srcnn_mse = avg_srcnn_mse / img_num
            avg_srcnn_psnr = avg_srcnn_psnr / img_num
            avg_srcnn_ssim = avg_srcnn_ssim / img_num
            avg_srcnn_ergas = avg_srcnn_ergas / img_num
            avg_srcnn_lpips = avg_srcnn_lpips / img_num
            time_elapsed = time.time() - start_time
            # Save to log
            print_rlt = OrderedDict()
            print_rlt['model'] = datasetName
            print_rlt['epoch'] = epoch
            print_rlt['iters'] = epoch
            print_rlt['time'] = time_elapsed
            print_rlt['bicubic_mse'] = avg_bicubic_mse
            print_rlt['bicubic_psnr'] = avg_bicubic_psnr
            print_rlt['bicubic_ssim'] = avg_bicubic_ssim
            print_rlt['bicubic_ergas'] = avg_bicubic_ergas
            print_rlt['bicubic_lpips'] = avg_bicubic_lpips
            print_rlt['hat_mse'] = avg_srcnn_mse
            print_rlt['hat_psnr'] = avg_srcnn_psnr
            print_rlt['hat_ssim'] = avg_srcnn_ssim
            print_rlt['hat_ergas'] = avg_srcnn_ergas
            print_rlt['hat_lpips'] = avg_srcnn_lpips
            self.logger.print_format_results('val', print_rlt)

            del test_data_loader
            del test_set

        average_bicubic_mse = average_bicubic_mse / num
        average_bicubic_psnr = average_bicubic_psnr / num
        average_bicubic_ssim = average_bicubic_ssim / num
        average_bicubic_ergas = average_bicubic_ergas / num
        average_bicubic_lpips = average_bicubic_lpips / num
        average_srrgan_mse = average_srrgan_mse / num
        average_srrgan_psnr = average_srrgan_psnr / num
        average_srrgan_ssim = average_srrgan_ssim / num
        average_srrgan_ergas = average_srrgan_ergas / num
        average_srrgan_lpips = average_srrgan_lpips / num
        time_elapsed = time.time() - start_time
        # Save to log
        print_rlt = OrderedDict()
        print_rlt['model'] = "Total"  # opt['model']
        print_rlt['epoch'] = epoch
        print_rlt['iters'] = epoch
        print_rlt['time'] = time_elapsed
        print_rlt['bicubic_mse'] = average_bicubic_mse
        print_rlt['bicubic_psnr'] = average_bicubic_psnr
        print_rlt['bicubic_ssim'] = average_bicubic_ssim
        print_rlt['bicubic_ergas'] = average_bicubic_ergas
        print_rlt['bicubic_lpips'] = average_bicubic_lpips
        print_rlt['hat_mse'] = average_srrgan_mse
        print_rlt['hat_psnr'] = average_srrgan_psnr
        print_rlt['hat_ssim'] = average_srrgan_ssim
        print_rlt['hat_ergas'] = average_srrgan_ergas
        print_rlt['hat_lpips'] = average_srrgan_lpips
        self.logger.print_format_results('val', print_rlt)
        print('-----------------------------------')
        torch.cuda.empty_cache()

    def mfe_test_single(self, img_fn, modelpath=None):
        # networks
        cuda = True if torch.cuda.is_available() else False
        # Initialize generator
        self.generator = GeneratorResNet(upscale=self.scale_factor, in_chans=3, img_size=int(self.crop_size / self.scale_factor))

        # load model
        if modelpath is not None:
            self.generator.load_state_dict(torch.load(modelpath), strict=False)

        if cuda:
            self.generator = self.generator.cuda()

        self.generator.eval()

        # load data
        img = Image.open(img_fn)
        toTensor = transforms.Compose([
            transforms.CenterCrop(self.test_crop_size),
            transforms.ToTensor()
        ])
        input = toTensor(img)
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.batch_size, self.num_channels, self.test_crop_size, self.test_crop_size)
        input_img = Variable(input_lr.copy_(input))
        bc_img = srutils.img_interp(input_img.cpu(), self.scale_factor)

        # prediction
        recon_imgs = self.generator(input_img)

        # save result images
        img_name = img_fn.split("/")[-1]
        srutils.save_img1(recon_imgs[0].cpu().data, self.save_dir, self.save_dir+'/SR_HAT_%s' % img_name)
        srutils.save_img1(bc_img[0].cpu().data, self.save_dir, self.save_dir+'/SR_Bicubic_%s' % img_name)
        srutils.plot_test_single_result([recon_imgs[0].cpu(), bc_img[0].cpu()], [0, 0], [0, 0], save_path=self.save_dir+'/SR_%s' % img_name)