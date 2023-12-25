import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import collections
try:
    import accimage
except ImportError:
    accimage = None

import yaml
from scipy import signal
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.

    See :class:`~torchvision.transforms.ToPIlImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not(_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        if npimg.dtype == np.int16:
            expected_mode = 'I;16'
        if npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)


def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not(_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.float().div(255)

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)
####################
# blur kernel and PCA
####################


def isogkern(kernlen, std):
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d = gkern2d/np.sum(gkern2d)
    return gkern2d


def anisogkern(kernlen, std1, std2, angle):
    gkern1d_1 = signal.gaussian(kernlen, std=std1).reshape(kernlen, 1)
    gkern1d_2 = signal.gaussian(kernlen, std=std2).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d_1, gkern1d_2)
    gkern2d = gkern2d/np.sum(gkern2d)
    return gkern2d


def PCA(data, k=2):
    X = torch.from_numpy(data)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(torch.t(X))
    return U[:, :k] # PCA matrix

def cal_sigma(sig_x, sig_y, radians):
    D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    U = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), 1 * np.cos(radians)]])
    sigma = np.dot(U, np.dot(D, U.T))
    return sigma


def anisotropic_gaussian_kernel(l, sigma_matrix, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((l * l, 1)), yy.reshape(l * l, 1))).reshape(l, l, 2)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(xy, inverse_sigma) * xy, 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def isotropic_gaussian_kernel(l, sigma, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def random_anisotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, scaling=3, l=21, tensor=False):
    pi = np.random.random() * math.pi * 2 - math.pi
    x = np.random.random() * (sig_max - sig_min) + sig_min
    y = np.clip(np.random.random() * scaling * x, sig_min, sig_max)
    sig = cal_sigma(x, y, pi)
    k = anisotropic_gaussian_kernel(l, sig, tensor=tensor)
    return k


def random_isotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, l=21, tensor=False):
    x = np.random.random() * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k


def stable_isotropic_gaussian_kernel(sig=2.6, l=21, tensor=False):
    x = sig
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k


def random_gaussian_kernel(l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, tensor=False):
    if np.random.random() < rate_iso:
        return random_isotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, tensor=tensor)
    else:
        return random_anisotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, scaling=scaling, tensor=tensor)


def stable_gaussian_kernel(l=21, sig=2.6, tensor=False):
    return stable_isotropic_gaussian_kernel(sig=sig, l=l, tensor=tensor)


def random_batch_kernel(batch, l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, tensor=True):
    batch_kernel = np.zeros((batch, l, l))
    for i in range(batch):
        batch_kernel[i] = random_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling, tensor=False)
    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel


def stable_batch_kernel(batch, l=21, sig=2.6, tensor=True):
    batch_kernel = np.zeros((batch, l, l))
    for i in range(batch):
        batch_kernel[i] = stable_gaussian_kernel(l=l, sig=sig, tensor=False)
    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel


def b_GPUVar_Bicubic(variable, scale):
    tensor = variable.cpu().data
    B, C, H, W = tensor.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_view = tensor.view((B*C, 1, H, W))
    re_tensor = torch.zeros((B*C, 1, H_new, W_new))
    for i in range(B*C):
        img = to_pil_image(tensor_view[i])
        re_tensor[i] = to_tensor(resize(img, (H_new, W_new), interpolation=Image.BICUBIC))
    re_tensor_view = re_tensor.view((B, C, H_new, W_new))
    return re_tensor_view


def b_CPUVar_Bicubic(variable, scale):
    tensor = variable.data
    B, C, H, W = tensor.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_v = tensor.view((B*C, 1, H, W))
    re_tensor = torch.zeros((B*C, 1, H_new, W_new))
    for i in range(B*C):
        img = to_pil_image(tensor_v[i])
        re_tensor[i] = to_tensor(resize(img, (H_new, W_new), interpolation=Image.BICUBIC))
    re_tensor_v = re_tensor.view((B, C, H_new, W_new))
    return re_tensor_v


def random_batch_noise(batch, high, rate_cln=1.0):
    noise_level = np.random.uniform(size=(batch, 1)) * high
    noise_mask = np.random.uniform(size=(batch, 1))
    noise_mask[noise_mask < rate_cln] = 0
    noise_mask[noise_mask >= rate_cln] = 1
    return noise_level * noise_mask


def b_GaussianNoising(tensor, sigma, mean=0.0, noise_size=None, min=0.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.mul(torch.FloatTensor(np.random.normal(loc=mean, scale=1.0, size=size)), sigma.view(sigma.size() + (1, 1)))
    return torch.clamp(noise + tensor, min=min, max=max)


class BatchSRKernel(object):
    def __init__(self, l=21, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3):
        self.l = l
        self.sig = sig
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.rate = rate_iso
        self.scaling = scaling

    def __call__(self, random, batch, tensor=False):
        if random == True: #random kernel
            return random_batch_kernel(batch, l=self.l, sig_min=self.sig_min, sig_max=self.sig_max, rate_iso=self.rate,
                                       scaling=self.scaling, tensor=tensor)
        else: #stable kernel
            return stable_batch_kernel(batch, l=self.l, sig=self.sig, tensor=tensor)


class PCAEncoder(nn.Module):
    def __init__(self, weight, cuda=False):
        super().__init__()
        self.weight = weight #[l^2, k]
        self.size = self.weight.size()
        if cuda:
            self.weight = Variable(self.weight).cuda()
        else:
            self.weight = Variable(self.weight)

    def __call__(self, batch_kernel):
        B, H, W = batch_kernel.size() #[B, l, l]
        return torch.bmm(batch_kernel.view((B, 1, H * W)), self.weight.expand((B, ) + self.size)).view((B, -1))


class BatchBlur(nn.Module):
    def __init__(self, l=15):
        super(BatchBlur, self).__init__()
        self.l = l
        if l % 2 == 1:
            self.pad = nn.ReflectionPad2d(l // 2)
        else:
            self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        # self.pad = nn.ZeroPad2d(l // 2)

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = pad.view((C * B, 1, H_p, W_p))
            kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        else:
            input_CBHW = pad.view((1, C * B, H_p, W_p))
            kernel_var = kernel.contiguous().view((B, 1, self.l, self.l)).repeat(1, C, 1, 1).view((B * C, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, groups=B*C).view((B, C, H, W))


class SRMDPreprocessing(object):
    def __init__(self, scale, random, pca_matrix=None, kernel=21, noise=True, cuda=False, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, rate_cln=0.2, noise_high=0.08):
        self.encoder = PCAEncoder(pca_matrix).cuda() if cuda else PCAEncoder(pca_matrix)
        self.kernel_gen = BatchSRKernel(l=kernel, sig=sig, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling)
        self.blur = BatchBlur(l=kernel)
        self.l = kernel
        self.noise = noise
        self.scale = scale
        self.cuda = cuda
        self.rate_cln = rate_cln
        self.noise_high = noise_high
        self.random = random

    def __call__(self, hr_tensor, kernel=False):
        ### hr_tensor is tensor, not cuda tensor
        B, C, H, W = hr_tensor.size()
        b_kernels = Variable(self.kernel_gen(self.random, B, tensor=True)).cuda() if self.cuda else Variable(self.kernel_gen(self.random, B, tensor=True))
        # blur
        if self.cuda:
            hr_blured_var = self.blur(Variable(hr_tensor).cuda(), b_kernels)
        else:
            hr_blured_var = self.blur(Variable(hr_tensor), b_kernels)
        # kernel encode

        # Down sample
        if self.cuda:
            lr_blured_t = b_GPUVar_Bicubic(hr_blured_var, self.scale)
        else:
            lr_blured_t = b_CPUVar_Bicubic(hr_blured_var, self.scale)

        # Noisy
        if self.noise:
            Noise_level = torch.FloatTensor(random_batch_noise(B, self.noise_high, self.rate_cln))
            lr_noised_t = b_GaussianNoising(lr_blured_t, Noise_level)
        else:
            Noise_level = torch.zeros((B, 1))
            lr_noised_t = lr_blured_t

        if self.cuda:
            Noise_level = Variable(Noise_level).cuda()
            lr_re = Variable(lr_noised_t).cuda()
        else:
            Noise_level = Variable(Noise_level)
            lr_re = Variable(lr_noised_t)

        # B x self.para_input
        kernel_code = self.encoder(b_kernels)
        re_code = (
            torch.cat([kernel_code, Noise_level * 10], dim=1)
            if self.noise
            else kernel_code
        )
        if self.cuda:
            re_code = re_code.cuda()
        return (lr_re, re_code, b_kernels) if kernel else (lr_re, re_code)
        # return (lr_re, b_kernels) if kernel else (lr_re)

class BlurPreprocessing(object):
    def __init__(self, scale, random, kernel=21, cuda=False, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, rate_cln=0.2, noise_high=0.08):
        self.kernel_gen = BatchSRKernel(l=kernel, sig=sig, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling)
        self.blur = BatchBlur(l=kernel)
        self.l = kernel
        self.scale = scale
        self.cuda = cuda
        self.rate_cln = rate_cln
        self.noise_high = noise_high
        self.random = random

    def __call__(self, hr_tensor, noise=False):
        ### hr_tensor is tensor, not cuda tensor
        B, C, H, W = hr_tensor.size()
        b_kernels = Variable(self.kernel_gen(self.random, B, tensor=True)).cuda() if self.cuda else Variable(self.kernel_gen(self.random, B, tensor=True))
        # blur
        if self.cuda:
            hr_blured_var = self.blur(Variable(hr_tensor).cuda(), b_kernels)
        else:
            hr_blured_var = self.blur(Variable(hr_tensor), b_kernels)
        # Noisy
        if noise:
            Noise_level = torch.FloatTensor(random_batch_noise(B, self.noise_high, self.rate_cln))
            lr_noised_t = b_GaussianNoising(hr_blured_var, Noise_level)
        else:
            lr_noised_t = hr_blured_var

        return lr_noised_t

# class SRMDPreprocessing(object):
#     def __init__(
#         self, scale, pca_matrix,
#         ksize=21, code_length=10,
#         random_kernel = True, noise=False, cuda=False, random_disturb=False,
#         sig=0, sig_min=0, sig_max=0, rate_iso=1.0, rate_cln=1, noise_high=0,
#         stored_kernel=False, pre_kernel_path = None
#     ):
#         self.encoder = PCAEncoder(pca_matrix).cuda() if cuda else PCAEncoder(pca)
#
#         self.kernel_gen = (
#             BatchSRKernel(
#                 l=ksize,
#                 sig=sig, sig_min=sig_min, sig_max=sig_max,
#                 rate_iso=rate_iso, random_disturb=random_disturb,
#             ) if not stored_kernel else
#             BatchBlurKernel(pre_kernel_path)
#         )
#
#         self.blur = BatchBlur(l=ksize)
#         self.para_in = code_length
#         self.l = ksize
#         self.noise = noise
#         self.scale = scale
#         self.cuda = cuda
#         self.rate_cln = rate_cln
#         self.noise_high = noise_high
#         self.random = random_kernel
#
#     def __call__(self, hr_tensor, kernel=False):
#         # hr_tensor is tensor, not cuda tensor
#
#         hr_var = Variable(hr_tensor).cuda() if self.cuda else Variable(hr_tensor)
#         device = hr_var.device
#         B, C, H, W = hr_var.size()
#
#         b_kernels = Variable(self.kernel_gen(self.random, B, tensor=True)).to(device)
#         hr_blured_var = self.blur(hr_var, b_kernels)
#
#         # B x self.para_input
#         kernel_code = self.encoder(b_kernels)
#
#         # Down sample
#         if self.scale != 1:
#             lr_blured_t = b_GPUVar_Bicubic(hr_blured_var, self.scale)
#         else:
#             lr_blured_t = b_CPUVar_Bicubic(hr_blured_var, self.scale)
#
#         # Noisy
#         if self.noise:
#             Noise_level = torch.FloatTensor(
#                 random_batch_noise(B, self.noise_high, self.rate_cln)
#             )
#             lr_noised_t = b_GaussianNoising(lr_blured_t, self.noise_high)
#         else:
#             Noise_level = torch.zeros((B, 1))
#             lr_noised_t = lr_blured_t
#
#         Noise_level = Variable(Noise_level).cuda()
#         re_code = (
#             torch.cat([kernel_code, Noise_level * 10], dim=1)
#             if self.noise
#             else kernel_code
#         )
#         lr_re = Variable(lr_noised_t).to(device)
#
#         return (lr_re, re_code, b_kernels) if kernel else (lr_re, re_code)

# class IsoGaussian(object):
#     def __init__(self, scale, para_input=10, kernel=21, noise=False, cuda=False, sig_min=1.8, sig_max=3.2, noise_high=0.0):
#         self.encoder = PCAEncoder(pca, cuda=cuda)
#         self.blur = BatchBlur(l=kernel)
#         self.min = sig_min
#         self.max = sig_max
#         self.para_in = para_input
#         self.l = kernel
#         self.noise = noise
#         self.scale = scale
#         self.cuda = cuda
#         self.noise_high = noise_high
#
#     def __call__(self, hr_tensor):
#         B, C, H, W = hr_tensor.size()
#         kernel_width = np.random.uniform(low=self.min, high=self.max, size=(B, 1))
#         batch_kernel = np.zeros((B, self.l, self.l))
#         for i in range(B):
#             batch_kernel[i] = isotropic_gaussian_kernel(self.l, kernel_width[i], tensor=False)
#         kernels = Variable(torch.FloatTensor(batch_kernel))
#
#         if self.cuda:
#             hr_blured_var = self.blur(Variable(hr_tensor).cuda(), kernels.cuda())
#         else:
#             hr_blured_var = self.blur(Variable(hr_tensor), kernels)
#
#         # kernel encode
#         # kernel_code = Variable(torch.FloatTensor(kernel_width))  # B x self.para_input
#         kernel_code = self.encoder(kernels)
#         if self.cuda:
#             lr_blured_t = b_GPUVar_Bicubic(hr_blured_var, self.scale)
#         else:
#             lr_blured_t = b_CPUVar_Bicubic(hr_blured_var, self.scale)
#
#         if self.noise:
#             lr_noised_t = b_GaussianNoising(lr_blured_t, self.noise_high)
#         else:
#             lr_noised_t = lr_blured_t
#
#         if self.cuda:
#             re_code = kernel_code.cuda()
#             lr_re = Variable(lr_noised_t).cuda()
#         else:
#             re_code = kernel_code
#             lr_re = Variable(lr_noised_t)
#         return lr_re, re_code

####################
# miscellaneous
####################

# def cal_sigma(sig_x, sig_y, radians):
#     sig_x = sig_x.view(-1, 1, 1)
#     sig_y = sig_y.view(-1, 1, 1)
#     radians = radians.view(-1, 1, 1)
#
#     D = torch.cat([F.pad(sig_x ** 2, [0, 1, 0, 0]), F.pad(sig_y ** 2, [1, 0, 0, 0])], 1)
#     U = torch.cat([torch.cat([radians.cos(), -radians.sin()], 2),
#                    torch.cat([radians.sin(), radians.cos()], 2)], 1)
#     sigma = torch.bmm(U, torch.bmm(D, U.transpose(1, 2)))
#
#     return sigma
#
#
# def anisotropic_gaussian_kernel(batch, kernel_size, covar):
#     ax = torch.arange(kernel_size).float().cuda() - kernel_size // 2
#
#     xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
#     yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
#     xy = torch.stack([xx, yy], -1).view(batch, -1, 2)
#
#     inverse_sigma = torch.inverse(covar)
#     kernel = torch.exp(- 0.5 * (torch.bmm(xy, inverse_sigma) * xy).sum(2)).view(batch, kernel_size, kernel_size)
#
#     return kernel / kernel.sum([1, 2], keepdim=True)
#
#
# def isotropic_gaussian_kernel(batch, kernel_size, sigma):
#     ax = torch.arange(kernel_size).float().cuda() - kernel_size//2
#     xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
#     yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
#     kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma.view(-1, 1, 1) ** 2))
#
#     return kernel / kernel.sum([1,2], keepdim=True)
#
#
# def random_anisotropic_gaussian_kernel(batch=1, kernel_size=21, lambda_min=0.2, lambda_max=4.0):
#     theta = torch.rand(batch).cuda() * math.pi
#     lambda_1 = torch.rand(batch).cuda() * (lambda_max - lambda_min) + lambda_min
#     lambda_2 = torch.rand(batch).cuda() * (lambda_max - lambda_min) + lambda_min
#
#     covar = cal_sigma(lambda_1, lambda_2, theta)
#     kernel = anisotropic_gaussian_kernel(batch, kernel_size, covar)
#     return kernel
#
#
# def stable_anisotropic_gaussian_kernel(kernel_size=21, theta=0, lambda_1=0.2, lambda_2=4.0):
#     theta = torch.ones(1).cuda() * theta * math.pi
#     lambda_1 = torch.ones(1).cuda() * lambda_1
#     lambda_2 = torch.ones(1).cuda() * lambda_2
#
#     covar = cal_sigma(lambda_1, lambda_2, theta)
#     kernel = anisotropic_gaussian_kernel(1, kernel_size, covar)
#     return kernel
#
#
# def random_isotropic_gaussian_kernel(batch=1, kernel_size=21, sig_min=0.2, sig_max=4.0):
#     x = torch.rand(batch).cuda() * (sig_max - sig_min) + sig_min
#     k = isotropic_gaussian_kernel(batch, kernel_size, x)
#     return k
#
#
# def stable_isotropic_gaussian_kernel(kernel_size=21, sig=4.0):
#     x = torch.ones(1).cuda() * sig
#     k = isotropic_gaussian_kernel(1, kernel_size, x)
#     return k
#
#
# def random_gaussian_kernel(batch, kernel_size=21, blur_type='iso_gaussian', sig_min=0.2, sig_max=4.0, lambda_min=0.2, lambda_max=4.0):
#     if blur_type == 'iso_gaussian':
#         return random_isotropic_gaussian_kernel(batch=batch, kernel_size=kernel_size, sig_min=sig_min, sig_max=sig_max)
#     elif blur_type == 'aniso_gaussian':
#         return random_anisotropic_gaussian_kernel(batch=batch, kernel_size=kernel_size, lambda_min=lambda_min, lambda_max=lambda_max)
#
#
# def stable_gaussian_kernel(kernel_size=21, blur_type='iso_gaussian', sig=2.6, lambda_1=0.2, lambda_2=4.0, theta=0):
#     if blur_type == 'iso_gaussian':
#         return stable_isotropic_gaussian_kernel(kernel_size=kernel_size, sig=sig)
#     elif blur_type == 'aniso_gaussian':
#         return stable_anisotropic_gaussian_kernel(kernel_size=kernel_size, lambda_1=lambda_1, lambda_2=lambda_2, theta=theta)
#
#
# # implementation of matlab bicubic interpolation in pytorch
# class bicubic(nn.Module):
#     def __init__(self):
#         super(bicubic, self).__init__()
#
#     def cubic(self, x):
#         absx = torch.abs(x)
#         absx2 = torch.abs(x) * torch.abs(x)
#         absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)
#
#         condition1 = (absx <= 1).to(torch.float32)
#         condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)
#
#         f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
#         return f
#
#     def contribute(self, in_size, out_size, scale):
#         kernel_width = 4
#         if scale < 1:
#             kernel_width = 4 / scale
#         x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32).cuda()
#         x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32).cuda()
#
#         u0 = x0 / scale + 0.5 * (1 - 1 / scale)
#         u1 = x1 / scale + 0.5 * (1 - 1 / scale)
#
#         left0 = torch.floor(u0 - kernel_width / 2)
#         left1 = torch.floor(u1 - kernel_width / 2)
#
#         P = np.ceil(kernel_width) + 2
#
#         indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()
#         indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()
#
#         mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
#         mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)
#
#         if scale < 1:
#             weight0 = scale * self.cubic(mid0 * scale)
#             weight1 = scale * self.cubic(mid1 * scale)
#         else:
#             weight0 = self.cubic(mid0)
#             weight1 = self.cubic(mid1)
#
#         weight0 = weight0 / (torch.sum(weight0, 2).unsqueeze(2))
#         weight1 = weight1 / (torch.sum(weight1, 2).unsqueeze(2))
#
#         indice0 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice0), torch.FloatTensor([in_size[0]]).cuda()).unsqueeze(0)
#         indice1 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice1), torch.FloatTensor([in_size[1]]).cuda()).unsqueeze(0)
#
#         kill0 = torch.eq(weight0, 0)[0][0]
#         kill1 = torch.eq(weight1, 0)[0][0]
#
#         weight0 = weight0[:, :, kill0 == 0]
#         weight1 = weight1[:, :, kill1 == 0]
#
#         indice0 = indice0[:, :, kill0 == 0]
#         indice1 = indice1[:, :, kill1 == 0]
#
#         return weight0, weight1, indice0, indice1
#
#     def forward(self, input, scale=1/4):
#         b, c, h, w = input.shape
#
#         weight0, weight1, indice0, indice1 = self.contribute([h, w], [int(h * scale), int(w * scale)], scale)
#         weight0 = weight0[0]
#         weight1 = weight1[0]
#
#         indice0 = indice0[0].long()
#         indice1 = indice1[0].long()
#
#         out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
#         out = (torch.sum(out, dim=3))
#         A = out.permute(0, 1, 3, 2)
#
#         out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
#         out = out.sum(3).permute(0, 1, 3, 2)
#
#         return out
#
#
# class Gaussin_Kernel(object):
#     def __init__(self, kernel_size=21, blur_type='iso_gaussian',
#                  sig=2.6, sig_min=0.2, sig_max=4.0,
#                  lambda_1=0.2, lambda_2=4.0, theta=0, lambda_min=0.2, lambda_max=4.0):
#         self.kernel_size = kernel_size
#         self.blur_type = blur_type
#
#         self.sig = sig
#         self.sig_min = sig_min
#         self.sig_max = sig_max
#
#         self.lambda_1 = lambda_1
#         self.lambda_2 = lambda_2
#         self.theta = theta
#         self.lambda_min = lambda_min
#         self.lambda_max = lambda_max
#
#     def __call__(self, batch, random):
#         # random kernel
#         if random == True:
#             return random_gaussian_kernel(batch, kernel_size=self.kernel_size, blur_type=self.blur_type,
#                                           sig_min=self.sig_min, sig_max=self.sig_max,
#                                           lambda_min=self.lambda_min, lambda_max=self.lambda_max)
#
#         # stable kernel
#         else:
#             return stable_gaussian_kernel(kernel_size=self.kernel_size, blur_type=self.blur_type,
#                                           sig=self.sig,
#                                           lambda_1=self.lambda_1, lambda_2=self.lambda_2, theta=self.theta)
#
# class BatchBlur(nn.Module):
#     def __init__(self, kernel_size=21):
#         super(BatchBlur, self).__init__()
#         self.kernel_size = kernel_size
#         if kernel_size % 2 == 1:
#             self.pad = nn.ReflectionPad2d(kernel_size//2)
#         else:
#             self.pad = nn.ReflectionPad2d((kernel_size//2, kernel_size//2-1, kernel_size//2, kernel_size//2-1))
#
#     def forward(self, input, kernel):
#         B, C, H, W = input.size()
#         input_pad = self.pad(input)
#         H_p, W_p = input_pad.size()[-2:]
#
#         if len(kernel.size()) == 2:
#             input_CBHW = input_pad.view((C * B, 1, H_p, W_p))
#             kernel = kernel.contiguous().view((1, 1, self.kernel_size, self.kernel_size))
#
#             return F.conv2d(input_CBHW, kernel, padding=0).view((B, C, H, W))
#         else:
#             input_CBHW = input_pad.view((1, C * B, H_p, W_p))
#             kernel = kernel.contiguous().view((B, 1, self.kernel_size, self.kernel_size))
#             kernel = kernel.repeat(1, C, 1, 1).view((B * C, 1, self.kernel_size, self.kernel_size))
#
#             return F.conv2d(input_CBHW, kernel, groups=B*C).view((B, C, H, W))
#
#
# class SRMDPreprocessing(object):
#     def __init__(self,
#                  scale,
#                  mode='bicubic',
#                  kernel_size=21,
#                  blur_type='iso_gaussian',
#                  sig=2.6,
#                  sig_min=0.2,
#                  sig_max=4.0,
#                  lambda_1=0.2,
#                  lambda_2=4.0,
#                  theta=0,
#                  lambda_min=0.2,
#                  lambda_max=4.0,
#                  noise=0.0
#                  ):
#         '''
#         # sig, sig_min and sig_max are used for isotropic Gaussian blurs
#         During training phase (random=True):
#             the width of the blur kernel is randomly selected from [sig_min, sig_max]
#         During test phase (random=False):
#             the width of the blur kernel is set to sig
#
#         # lambda_1, lambda_2, theta, lambda_min and lambda_max are used for anisotropic Gaussian blurs
#         During training phase (random=True):
#             the eigenvalues of the covariance is randomly selected from [lambda_min, lambda_max]
#             the angle value is randomly selected from [0, pi]
#         During test phase (random=False):
#             the eigenvalues of the covariance are set to lambda_1 and lambda_2
#             the angle value is set to theta
#         '''
#         self.kernel_size = kernel_size
#         self.scale = scale
#         self.mode = mode
#         self.noise = noise
#
#         self.gen_kernel = Gaussin_Kernel(
#             kernel_size=kernel_size, blur_type=blur_type,
#             sig=sig, sig_min=sig_min, sig_max=sig_max,
#             lambda_1=lambda_1, lambda_2=lambda_2, theta=theta, lambda_min=lambda_min, lambda_max=lambda_max
#         )
#         self.blur = BatchBlur(kernel_size=kernel_size)
#         self.bicubic = bicubic()
#
#     def __call__(self, hr_tensor, random=True):
#         with torch.no_grad():
#             # only downsampling
#             if self.gen_kernel.sig == 0:
#                 B, N, C, H, W = hr_tensor.size()
#                 hr_blured = hr_tensor.view(-1, C, H, W)
#                 b_kernels = None
#
#             # gaussian blur + downsampling
#             else:
#                 B, C, H, W = hr_tensor.size()
#                 b_kernels = self.gen_kernel(B, random)  # B degradations
#
#                 # blur
#                 hr_blured = self.blur(hr_tensor.view(B, -1, H, W), b_kernels)
#                 hr_blured = hr_blured.view(-1, C, H, W)  # BN, C, H, W
#
#             # downsampling
#             if self.mode == 'bicubic':
#                 lr_blured = self.bicubic(hr_blured, scale=1/self.scale)
#             elif self.mode == 's-fold':
#                 lr_blured = hr_blured.view(-1, C, H//self.scale, self.scale, W//self.scale, self.scale)[:, :, :, 0, :, 0]
#
#
#             # add noise
#             if self.noise > 0:
#                 _, C, H_lr, W_lr = lr_blured.size()
#                 noise_level = torch.rand(B, 1, 1, 1, 1).to(lr_blured.device) * self.noise if random else self.noise
#                 noise = torch.randn_like(lr_blured).view(-1, C, H_lr, W_lr).mul_(noise_level).view(-1, C, H_lr, W_lr)
#                 lr_blured.add_(noise)
#
#             lr_blured = torch.clamp(lr_blured.round(), 0, 255)
#
#
#             return lr_blured.view(B, N, C, H//int(self.scale), W//int(self.scale)), b_kernels


import os
from skimage.io import imread, imsave

from torchvision.utils import make_grid
def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

import cv2
def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)

def img2tensor(img):
    '''
    # BGR to RGB, HWC to CHW, numpy to tensor
    Input: img(H, W, C), [0,255], np.uint8 (default)
    Output: 3D(C,H,W), RGB order, float tensor
    '''
    img = img.astype(np.float32) / 255.
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    return img

def create_downsampling_dataset(datapath, dstpath, scale):
    # 读取分类名称
    class_names = os.listdir(datapath)
    if not os.path.isdir(dstpath):
        # 创建新文件夹
        os.mkdir(dstpath)
        for class_name in class_names:
            class_subdir = os.path.join(dstpath, class_name)
            os.mkdir(class_subdir)

    # preprocessing = SRMDPreprocessing(scale, mode='bicubic', kernel_size=21, blur_type='aniso_gaussian')
    for root, _, filenames in os.walk(datapath):
        if filenames:
            class_name = os.path.basename(root)
            # 随机
            filenames = np.random.permutation(filenames)
            target_dir = os.path.join(dstpath, class_name)
            for filename in filenames:
                filepath = os.path.join(root, filename)
                image = imread(filepath)

                # LR_blur, by random gaussian kernel
                img_HR = img2tensor(image)
                C, H, W = img_HR.size()
                # image = interpolation(image, scale, interpolation='bicubic')
                # prepro = SRMDPreprocessing(scale, mode='bicubic', kernel_size=21, blur_type='aniso_gaussian')
                prepro = SRMDPreprocessing(scale, random=True, kernel=21, noise=False,
                                        cuda=False, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3,
                                        rate_cln=0.2, noise_high=0.0) #random(sig_min, sig_max) | stable kernel(sig)
                LR_img = prepro(img_HR.view(1, C, H, W))
                image_LR_blur = tensor2img(LR_img)
                if H!=W:
                    image_LR_blur = np.array(resize(to_pil_image(image_LR_blur), 64))

                basename, _ = os.path.splitext(filename)
                # TIFF转化为png
                target_filename = os.path.join(target_dir, basename + '_downsample.png')
                imsave(target_filename, image_LR_blur)

import numpy as np
import scipy.io as io
from scipy import misc
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import random
import h5py
import json

def bandmean(input):
    # Calculate the mean of each channel for normalization
    C = input.shape[2]
    MEAN_ARRAY = np.ndarray(shape=(C,), dtype=float)
    for i in range(C):
        MEAN_ARRAY[i] = np.mean(input[:, :, i])
    return MEAN_ARRAY

from skimage import transform
def interpolation(img, scale_factor, interpolation='bicubic'):
    data = list()
    H = img.shape[0]
    W = img.shape[1]
    C = img.shape[2]
    h = int(H * scale_factor)
    w = int(W * scale_factor)
    for i in range(C):
        # data.append(misc.imresize(img[:, :, i], [h, w], interp=interpolation, mode="F")[:, :, None])
        data.append(transform.resize(img[:, :, i], [h, w], order=3)[:, :, None])
    img_out = np.concatenate(data, axis=2)
    img_out[img_out > 1] = 1.0
    img_out[img_out < 0] = 0.0
    return img_out

def prepare_dataset_train_test(dataname, datapath, save_path, patch_size=64, stride=32, scale_factor=4, test_ratio=0.25):
    GT_Patch, MS_Patch, Cubic_Patch = [], [], []
    if dataname == "Chikusei":
        mat = h5py.File(datapath)
        input_mat = np.transpose(mat['chikusei'])
        img = input_mat[107:2411, 144: 2192, :]
    elif dataname == "IndianPines":
        mat = io.loadmat(datapath)
        input_mat = mat['indianpines']
        img = input_mat
    elif dataname == "Pavia":
        mat = io.loadmat(datapath)
        input_mat = mat['pavia']
        img = input_mat
    elif dataname == "PaviaU":
        mat = io.loadmat(datapath)
        input_mat = mat['paviaU']
        img = input_mat
    del mat, input_mat

    # normalization
    img = img.astype(float)
    img -= np.min(img)
    img /= np.max(img)

    mean_arrays = bandmean(img)
    means_json = {str(i): mean_arrays[i] for i in range(len(mean_arrays))}
    QIstr = save_path+'/'+dataname+'_x'+str(scale_factor) + '/band_means.txt'
    meanfile = open(QIstr, 'w')
    meanfile.write(str(mean_arrays))
    json.dump(means_json, meanfile)
    meanfile.close()

    H = img.shape[0]
    W = img.shape[1]
    C = img.shape[2]
    pat_col_num = list(range(0, H - patch_size + 1, stride))
    pat_row_num = list(range(0, W - patch_size + 1, stride))
    colnums = len(pat_col_num)
    rownums = len(pat_row_num)
    total_num = 0
    split_h = int(H / 2)
    split_w = int(W / 2)
    for i in range(colnums):
        for j in range(rownums):
            # curr_inp = Patch(i, j)
            up = pat_col_num[i]
            down = up + patch_size
            left = pat_row_num[j]
            right = left + patch_size
            gt = img[up:down, left:right, :]
            # ms = interpolation(gt, 1.0/scale_factor, interpolation='bicubic')
            # ms_bicubic = interpolation(ms, scale_factor, interpolation='bicubic')

            # LR_blur, by random gaussian kernel
            img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(gt, (2, 0, 1)))).float()
            C, H, W = img_HR.size()
            # image = interpolation(image, scale, interpolation='bicubic')
            # prepro = SRMDPreprocessing(scale, mode='bicubic', kernel_size=21, blur_type='aniso_gaussian')
            prepro = SRMDPreprocessing(scale_factor, random=True, kernel=11, noise=False,
                                       cuda=False, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3,
                                       rate_cln=0.2, noise_high=0.0)  # random(sig_min, sig_max) | stable kernel(sig)
            LR_img = prepro(img_HR.view(1, C, H, W))
            image_LR_blur = make_grid(LR_img, nrow=int(math.sqrt(1)), normalize=False).numpy()
            image_LR_blur = np.transpose(image_LR_blur, (1, 2, 0))

            ms_bicubic = interpolation(image_LR_blur, scale_factor, interpolation='bicubic')

            if down < split_h and right < split_w:
                filepath = save_path + '/' + dataname + '_srmd_x' + str(scale_factor) + '/test/'
            else:
                filepath = save_path+'/'+dataname+'_srmd_x'+str(scale_factor)+'/train/'
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            io.savemat(filepath + 'patch_' + str(i) + '_' + str(j) + '.mat',
                       {'gt': gt, 'ms': image_LR_blur, 'ms_bicubic': ms_bicubic})
            total_num += 1
import os
import time
import torch
import numpy as np
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from torch.nn import functional as F
from scipy.ndimage import measurements, interpolation

# from ZSSRforKernelGAN.ZSSR import ZSSR


def move2cpu(d):
    """Move data from gpu to cpu"""
    return d.detach().cpu().float().numpy()


def tensor2im(im_t):
    """Copy the tensor to the cpu & convert to range [0,255]"""
    im_np = np.clip(np.round((np.transpose(move2cpu(im_t).squeeze(0), (1, 2, 0)) + 1) / 2.0 * 255.0), 0, 255)
    return im_np.astype(np.uint8)


def im2tensor(im_np):
    """Copy the image to the gpu & converts to range [-1,1]"""
    im_np = im_np / 255.0 if im_np.dtype == 'uint8' else im_np
    return torch.FloatTensor(np.transpose(im_np, (2, 0, 1)) * 2.0 - 1.0).unsqueeze(0)


def map2tensor(gray_map):
    """Move gray maps to GPU, no normalization is done"""
    return torch.FloatTensor(gray_map).unsqueeze(0).unsqueeze(0)


def resize_tensor_w_kernel(im_t, k, sf=None):
    """Convolves a tensor with a given bicubic kernel according to scale factor"""
    # Expand dimensions to fit convolution: [out_channels, in_channels, k_height, k_width]
    k = k.expand(im_t.shape[1], im_t.shape[1], k.shape[0], k.shape[1])
    # Calculate padding
    padding = (k.shape[-1] - 1) // 2
    return F.conv2d(im_t, k, stride=round(sf), padding=padding)


def read_image(path):
    """Loads an image"""
    im = Image.open(path).convert('RGB')
    im = np.array(im, dtype=np.uint8)
    return im


def rgb2gray(im):
    """Convert and RGB image to gray-scale"""
    return np.dot(im, [0.299, 0.587, 0.114]) if len(im.shape) == 3 else im


def swap_axis(im):
    """Swap axis of a tensor from a 3 channel tensor to a batch of 3-single channel and vise-versa"""
    return im.transpose(0, 1) if type(im) == torch.Tensor else np.moveaxis(im, 0, 1)


def shave_a2b(a, b):
    """Given a big image or tensor 'a', shave it symmetrically into b's shape"""
    # If dealing with a tensor should shave the 3rd & 4th dimension, o.w. the 1st and 2nd
    is_tensor = (type(a) == torch.Tensor)
    r = 2 if is_tensor else 0
    c = 3 if is_tensor else 1
    # Calculate the shaving of each dimension
    shave_r, shave_c = max(0, a.shape[r] - b.shape[r]), max(0, a.shape[c] - b.shape[c])
    return a[:, :, shave_r // 2:a.shape[r] - shave_r // 2 - shave_r % 2, shave_c // 2:a.shape[c] - shave_c // 2 - shave_c % 2] if is_tensor \
        else a[shave_r // 2:a.shape[r] - shave_r // 2 - shave_r % 2, shave_c // 2:a.shape[c] - shave_c // 2 - shave_c % 2]


def create_gradient_map(im, window=5, percent=.97):
    """Create a gradient map of the image blurred with a rect of size window and clips extreme values"""
    # Calculate gradients
    gx, gy = np.gradient(rgb2gray(im))
    # Calculate gradient magnitude
    gmag, gx, gy = np.sqrt(gx ** 2 + gy ** 2), np.abs(gx), np.abs(gy)
    # Pad edges to avoid artifacts in the edge of the image
    gx_pad, gy_pad, gmag = pad_edges(gx, int(window)), pad_edges(gy, int(window)), pad_edges(gmag, int(window))
    lm_x, lm_y, lm_gmag = clip_extreme(gx_pad, percent), clip_extreme(gy_pad, percent), clip_extreme(gmag, percent)
    # Sum both gradient maps
    grads_comb = lm_x / lm_x.sum() + lm_y / lm_y.sum() + gmag / gmag.sum()
    # Blur the gradients and normalize to original values
    loss_map = convolve2d(grads_comb, np.ones(shape=(window, window)), 'same') / (window ** 2)
    # Normalizing: sum of map = numel
    return loss_map / np.mean(loss_map)


def create_probability_map(loss_map, crop):
    """Create a vector of probabilities corresponding to the loss map"""
    # Blur the gradients to get the sum of gradients in the crop
    blurred = convolve2d(loss_map, np.ones([crop // 2, crop // 2]), 'same') / ((crop // 2) ** 2)
    # Zero pad s.t. probabilities are NNZ only in valid crop centers
    prob_map = pad_edges(blurred, crop // 2)
    # Normalize to sum to 1
    prob_vec = prob_map.flatten() / prob_map.sum() if prob_map.sum() != 0 else np.ones_like(prob_map.flatten()) / prob_map.flatten().shape[0]
    return prob_vec


def pad_edges(im, edge):
    """Replace image boundaries with 0 without changing the size"""
    zero_padded = np.zeros_like(im)
    zero_padded[edge:-edge, edge:-edge] = im[edge:-edge, edge:-edge]
    return zero_padded


def clip_extreme(im, percent):
    """Zeroize values below the a threshold and clip all those above"""
    # Sort the image
    im_sorted = np.sort(im.flatten())
    # Choose a pivot index that holds the min value to be clipped
    pivot = int(percent * len(im_sorted))
    v_min = im_sorted[pivot]
    # max value will be the next value in the sorted array. if it is equal to the min, a threshold will be added
    v_max = im_sorted[pivot + 1] if im_sorted[pivot + 1] > v_min else v_min + 10e-6
    # Clip an zeroize all the lower values
    return np.clip(im, v_min, v_max) - v_min


def post_process_k(k, n):
    """Move the kernel to the CPU, eliminate negligible values, and centralize k"""
    k = move2cpu(k)
    # Zeroize negligible values
    significant_k = zeroize_negligible_val(k, n)
    # Force centralization on the kernel
    centralized_k = kernel_shift(significant_k, sf=2)
    # return shave_a2b(centralized_k, k)
    return centralized_k


def zeroize_negligible_val(k, n):
    """Zeroize values that are negligible w.r.t to values in k"""
    # Sort K's values in order to find the n-th largest
    k_sorted = np.sort(k.flatten())
    # Define the minimum value as the 0.75 * the n-th largest value
    k_n_min = 0.75 * k_sorted[-n - 1]
    # Clip values lower than the minimum value
    filtered_k = np.clip(k - k_n_min, a_min=0, a_max=100)
    # Normalize to sum to 1
    return filtered_k / filtered_k.sum()


def create_penalty_mask(k_size, penalty_scale):
    """Generate a mask of weights penalizing values close to the boundaries"""
    center_size = k_size // 2 + k_size % 2
    mask = create_gaussian(size=k_size, sigma1=k_size, is_tensor=False)
    mask = 1 - mask / np.max(mask)
    margin = (k_size - center_size) // 2 - 1
    mask[margin:-margin, margin:-margin] = 0
    return penalty_scale * mask


def create_gaussian(size, sigma1, sigma2=-1, is_tensor=False):
    """Return a Gaussian"""
    func1 = [np.exp(-z ** 2 / (2 * sigma1 ** 2)) / np.sqrt(2 * np.pi * sigma1 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    func2 = func1 if sigma2 == -1 else [np.exp(-z ** 2 / (2 * sigma2 ** 2)) / np.sqrt(2 * np.pi * sigma2 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    return torch.FloatTensor(np.outer(func1, func2)).cuda() if is_tensor else np.outer(func1, func2)


def nn_interpolation(im, sf):
    """Nearest neighbour interpolation"""
    pil_im = Image.fromarray(im)
    return np.array(pil_im.resize((im.shape[1] * sf, im.shape[0] * sf), Image.NEAREST), dtype=im.dtype)


def analytic_kernel(k):
    """Calculate the X4 kernel from the X2 kernel (for proof see appendix in paper)"""
    k_size = k.shape[0]
    # Calculate the big kernels size
    big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
    # Loop over the small kernel to fill the big one
    for r in range(k_size):
        for c in range(k_size):
            big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k
    # Crop the edges of the big kernel to ignore very small values and increase run time of SR
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    # Normalize to 1
    return cropped_big_k / cropped_big_k.sum()


def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel :
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between odd and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second term ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) // 2 + 0.5 * (np.array(sf) - (np.array(kernel.shape) % 2))
    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass
    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    kernel = np.pad(kernel, np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')

    # Finally shift the kernel and return
    kernel = interpolation.shift(kernel, shift_vec)

    return kernel


def save_final_kernel(k_2, conf):
    """saves the final kernel and the analytic kernel to the results folder"""
    sio.savemat(os.path.join(conf.output_dir_path, '%s_kernel_x2.mat' % conf.img_name), {'Kernel': k_2})
    if conf.X4:
        k_4 = analytic_kernel(k_2)
        sio.savemat(os.path.join(conf.output_dir_path, '%s_kernel_x4.mat' % conf.img_name), {'Kernel': k_4})

# from ZSSR import ZSSR
# def run_zssr(k_2, scale):
#     """Performs ZSSR with estimated kernel for wanted scale factor"""
#     if conf.do_ZSSR:
#         start_time = time.time()
#         print('~' * 30 + '\nRunning ZSSR X%d...' % (4 if conf.X4 else 2))
#         if conf.X4:
#             sr = ZSSR(conf.input_image_path, scale_factor=[[2, 2], [4, 4]], kernels=[k_2, analytic_kernel(k_2)], is_real_img=conf.real_image, noise_scale=conf.noise_scale).run()
#         else:
#             sr = ZSSR(conf.input_image_path, scale_factor=2, kernels=[k_2], is_real_img=conf.real_image, noise_scale=conf.noise_scale).run()
#         max_val = 255 if sr.dtype == 'uint8' else 1.
#         plt.imsave(os.path.join(conf.output_dir_path, 'ZSSR_%s.png' % conf.img_name), sr, vmin=0, vmax=max_val, dpi=1)
#         runtime = int(time.time() - start_time)
#         print('Completed! runtime=%d:%d\n' % (runtime // 60, runtime % 60) + '~' * 30)

if __name__ == '__main__':
    # create_downsampling_dataset("/home/zju/lyd/dataset/UCMerced_LandUse/", "/home/zju/lyd/dataset/UCMerced_LandUse_srmd_x2/", 2)
    create_downsampling_dataset("/home/zju/lyd/dataset/UCMerced_LandUse/", "/home/zju/lyd/dataset/UCMerced_LandUse_srmd_x4/", 4)
    # prepare_dataset_train_test('Pavia', '/home/zju/lyd/dataset/HSI/Pavia.mat', '/home/zju/lyd/dataset/HSI/', patch_size=64, stride=32, scale_factor=4, test_ratio=0.25)
    # prepare_dataset_train_test('Pavia', '/home/zju/lyd/dataset/HSI/Pavia.mat', '/home/zju/lyd/dataset/HSI/', patch_size=128, stride=64, scale_factor=8, test_ratio=0.25)