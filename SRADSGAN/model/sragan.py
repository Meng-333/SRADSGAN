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
from skimage.measure import compare_nrmse
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.models import vgg19

from utils import utils as srutils
from utils.logger import Logger, PrintLogger
from data import *
# from loss import GANLoss
# from architecture import RRDBNet, Discriminator_VGG_256, VGGFeatureExtractor
from model.base_networks import *
from data.data import get_RGB_trainDataset, get_RGB_testDataset

# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss

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

class Block(nn.Module):
    def __init__(
        self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(Block, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv2d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv2d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        vgg19_model = vgg19(pretrained=True)

        # Extracts features at the 11th layer
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:12])

    def forward(self, img):
        out = self.feature_extractor(img)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.Conv2d(in_features, in_features, 3, 1, 1),
                        nn.BatchNorm2d(in_features),
                        # nn.utils.weight_norm(),
                        nn.ReLU(),
                        nn.Conv2d(in_features, in_features, 3, 1, 1),
                        nn.BatchNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, buildingblock, in_channels=3, out_channels=3, n_residual_blocks=12, n_basic_blocks=1,   #默认n_residual_blocks=12  n_basic_blocks=1
                 rla_mode='CA-SA', bla_mode='CA-SA', ga_mode='CA-SA', pool_mode='Avg|Max', addconv=True, upscale_factor=3):
        super(GeneratorResNet, self).__init__()

        self.ga_mode = ga_mode
        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            # nn.ReLU(inplace=True)
            # nn.PReLU()
            nn.LeakyReLU(inplace=True)
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(
                buildingblock(BasicBlock, n_blocks=n_basic_blocks, nc=64, gc=32, kernel_size=3, stride=1, padding=1, norm_type=None,
                                                    act_type='lrelu', mode='CNA', rla_mode=rla_mode, bla_mode=bla_mode, pool_mode=pool_mode, addconv=addconv))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64))

        if self.ga_mode.find('CA') != -1:
            self.ca = CAM_Module(64)
        if self.ga_mode.find('SA') != -1:
            self.sa = PAM_Module(64)
        if self.ga_mode.find('-') != -1 and addconv:
            self.conv = nn.Conv2d(64, 64, kernel_size=1, bias=True)
        if self.ga_mode.find('|') != -1:
            self.conv = nn.Conv2d(64*2, 64, kernel_size=1, bias=True)
        self.addconv = addconv
        # Upsampling layers
        upsampling = []
        upsampling_two = [nn.Conv2d(64, 64 * 4, 3, 1, 1),
                          nn.BatchNorm2d(64 * 4),
                          nn.PixelShuffle(upscale_factor=2),
                          nn.LeakyReLU(inplace=True)]
        upsampling_three = [nn.Conv2d(64, 64 * 9, 3, 1, 1),
                            nn.BatchNorm2d(64 * 9),
                            nn.PixelShuffle(upscale_factor=3),
                            nn.LeakyReLU(inplace=True)]
        if (upscale_factor & (upscale_factor - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling += upsampling_two
        elif upscale_factor % 3 == 0:
            for _ in range(int(math.log(upscale_factor, 3))):
                upsampling += upsampling_three

                # for out_features in range(1):
        #     upsampling += [ nn.Conv2d(64, 64*9, 3, 1, 1),
        #                     nn.BatchNorm2d(64*9),
        #                     nn.PixelShuffle(upscale_factor=3),
        #                     nn.LeakyReLU(inplace=True)
        #                     ]

        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)

        if self.ga_mode == 'CA':
            out = self.ca(out)
        elif self.ga_mode == 'SA':
            out = self.sa(out)
        elif self.ga_mode == 'CA-SA':
            out = self.ca(out)
            out = self.sa(out)
            if self.addconv:
                out = self.conv(out)
        elif self.ga_mode == 'SA-CA':
            out = self.sa(out)
            out = self.ca(out)
            if self.addconv:
                out = self.conv(out)
        elif self.ga_mode == 'CA|SA':
            saout = self.sa(out)
            caout = self.ca(out)
            out = self.conv(torch.cat([caout,saout], dim=1))

        out = self.upsampling(out)
        out = self.conv3(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, attention=True):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for layer, out_filters, stride, normalize in [(1, 64, 1, False),
                                                (2, 64, 2, True),
                                                (3, 128, 1, True),
                                                (4, 128, 2, True),
                                                (5, 256, 1, True),
                                                (6, 256, 2, True),
                                                (7, 512, 1, True),
                                                (8, 512, 2, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            if attention:
                if layer==6:
                    layers.append(ChannelAttention(256))
                    layers.append(SpatialAttention())
                if layers==8:
                    layers.append(CAM_Module(512))
                    layers.append(PAM_Module(512))
            in_filters = out_filters

        # Output layer
        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

class SRAGAN(object):
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

        self.lr_transforms = [
            transforms.CenterCrop(self.crop_size),
            transforms.Resize(self.crop_size // self.scale_factor),
            transforms.ToTensor()
        ]
        self.hr_transforms = [
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor()
        ]
        # self.lr_transforms = [transforms.Resize((self.hr_height // self.scale_factor, self.hr_height // self.scale_factor), Image.BICUBIC),
        #                  transforms.ToTensor(),
        #                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        #
        # self.hr_transforms = [transforms.Resize((self.hr_height, self.hr_height), Image.BICUBIC),
        #                  transforms.ToTensor(),
        #                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

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

    def compute_gradient_penalty(self, discriminator, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        #cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if self.gpu_mode else torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = discriminator(interpolates)
        # fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)

        # add by lyd
        Tensor = torch.cuda.FloatTensor if self.gpu_mode else torch.Tensor
        grad_outputs = Tensor(d_interpolates.size())
        grad_outputs.resize_(d_interpolates.size()).fill_(1.0)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                  grad_outputs=grad_outputs, create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def gradient_penalty(self, discriminator, real_samples, fake_samples, grad_penalty_Lp_norm='L2', penalty_type='LS'):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Best setting (novel Hinge Linfinity gradient penalty)
        #grad_penalty_Lp_norm = 'Linf'
        #penalty_type = 'hinge'
        # Default setting from WGAN-GP and most cases (L2 gradient penalty)
        #grad_penalty_Lp_norm = 'L2'
        #penalty_type = 'LS'
        # Calculate gradient
        #penalty_weight = 20 # 10 is the more usual choice

        #cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if self.gpu_mode else torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = discriminator(interpolates)
        # fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)

        Tensor = torch.cuda.FloatTensor if self.gpu_mode else torch.Tensor
        grad_outputs = Tensor(d_interpolates.size())
        grad_outputs.resize_(d_interpolates.size()).fill_(1.0)
        #grad_outputs = torch.ones(param.batch_size)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,grad_outputs=grad_outputs, create_graph=True, retain_graph=True,only_inputs=True)[0]
        #grad = grad.view(current_batch_size,-1)

        if grad_penalty_Lp_norm == 'Linf': # Linfinity gradient norm penalty (Corresponds to L1 margin, BEST results)
          grad_abs = torch.abs(gradients) # Absolute value of gradient
          grad_norm , _ = torch.max(grad_abs,1)
        elif grad_penalty_Lp_norm == 'L1': # L1 gradient norm penalty (Corresponds to Linfinity margin, WORST results)
          grad_norm = gradients.norm(1,1)
        else: # L2 gradient norm penalty (Corresponds to L2 margin, this is what people generally use)
          grad_norm = gradients.norm(2,1)

        if penalty_type == 'LS': # The usual choice, penalize values below 1 and above 1 (too constraining to properly estimate the Wasserstein distance)
          constraint = (grad_norm-1).pow(2)
        elif penalty_type == 'hinge': # Penalize values above 1 only (best choice)
          constraint = torch.nn.ReLU()(grad_norm - 1)

        constraint = constraint.mean()
        gradient_penalty = constraint
        gradient_penalty.backward(retain_graph=True)

        return gradient_penalty

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
            # test_set = get_test_sets(self.data_dir, self.test_dataset, self.scale_factor, is_gray=is_gray)
            # test_set = get_test_sets_by_cropsize(self.data_dir, self.test_dataset, self.crop_size, self.scale_factor, is_gray=is_gray)
            test_set = get_RGB_testDataset(self.data_dir, self.test_dataset, self.crop_size, self.scale_factor, is_gray=is_gray)
            return DataLoader(dataset=test_set, num_workers=self.num_threads, batch_size=self.test_batch_size, shuffle=False, drop_last=True)
        # if dataset == 'train':
        #     print('Loading train dct_datasets...')
        #     return DataLoader(ImageDataset(os.path.join(self.data_dir, self.train_dataset[0]), lr_transforms=self.lr_transforms,
        #                                 hr_transforms=self.hr_transforms),
        #         batch_size=self.batch_size, shuffle=True, num_workers=self.n_cpu)
        # elif dataset == 'test':
        #     print('Loading test dct_datasets...')
        #     return DataLoader(ImageDataset(os.path.join(self.data_dir, self.test_dataset[0]), lr_transforms=self.lr_transforms,
        #                             hr_transforms=self.hr_transforms),
        #     batch_size=self.batch_size, shuffle=True, num_workers=self.n_cpu)

    def train(self):
        # os.makedirs('images', exist_ok=True)
        # os.makedirs('saved_models', exist_ok=True)

        # torch.cuda.set_device(0)
        # torch.backends.cudnn.benchmark = True
        # device_ids = [0, 1, 2]
        device_ids = [0]
        cuda = True if torch.cuda.is_available() else False
        # cuda = self.gpu_mode
        self.device = torch.device('cuda' if cuda else 'cpu')

        # Calculate output of image discriminator (PatchGAN)
        # patch_h, patch_w = int(self.crop_size / 2 ** self.scale_factor), int(self.crop_size / 2 ** self.scale_factor)
        patch_h, patch_w = math.ceil(self.crop_size / 2 ** 4), math.ceil(self.crop_size / 2 ** 4)
        patch = (self.batch_size, 1, patch_h, patch_w)

        # Initialize generator and discriminator
        self.generator = GeneratorResNet(ResidualBlock_Block_WithAttention, n_residual_blocks=12, n_basic_blocks=5,   #默认n_residual_blocks=12  n_basic_blocks=5
                                         rla_mode='CA-SA', bla_mode='CA-SA', ga_mode='CA-SA', pool_mode='Avg|Max',
                                         upscale_factor=self.scale_factor)
        # self.discriminator = Discriminator(norm_type='batch', use_spectralnorm=False, attention=False)
        self.discriminator = Discriminator()
        self.feature_extractor = FeatureExtractor()
        # self.feature_extractor = VGGFeatureExtractor(feature_layer=34, use_bn=False, use_input_norm=True, device=self.device)

        # srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='Generator', input_size=(1,3,int(256/self.scale_factor),int(256/self.scale_factor)))
        srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='Generator')
        srutils.print_network_to_file(self.discriminator, save_dir=self.save_dir, tag='Discriminator')
        srutils.print_network_to_file(self.feature_extractor, save_dir=self.save_dir, tag='FeatureExtractor')


        # Losses
        # self.criterion_GAN = torch.nn.MSELoss().to(self.device)
        if self.loss_Lp_norm=="L1":
            self.criterion_content = torch.nn.L1Loss().to(self.device)
        else:
            self.criterion_content = torch.nn.MSELoss().to(self.device)
        self.criterion_raGAN = GANLoss(gan_type='wgan-gp', real_label_val=1.0, fake_label_val=0.0).to(self.device)

        if len(device_ids) > 1:
            self.generator = nn.DataParallel(self.generator).to(self.device)
            self.discriminator = nn.DataParallel(self.discriminator).to(self.device)
            self.feature_extractor = nn.DataParallel(self.feature_extractor).to(self.device)

        if cuda:
            self.generator = self.generator.cuda(device_ids[0])
            self.discriminator = self.discriminator.cuda(device_ids[0])
            self.feature_extractor = self.feature_extractor.cuda(device_ids[0])
            # self.criterion_GAN = self.criterion_GAN.cuda(device_ids[0])
            self.criterion_content = self.criterion_content.cuda(device_ids[0])
            self.criterion_raGAN = self.criterion_raGAN.cuda(device_ids[0])

        model_dir = os.path.join(self.save_dir, 'model')
        if self.epoch != 0:
            # Load pretrained models
            # model_dict_G = torch.load(model_dir+'/generator_param_epoch_%d.pkl' % self.epoch)
            # trans_param_G = self.generator.state_dict()
            # for item, value in model_dict_G.items():
            #     name = '.'.join(item.split('.')[1:])
            #     trans_param_G[name] = value
            #
            # model_dict_D = torch.load(model_dir+'/discriminator_param_epoch_%d.pkl' % self.epoch)
            # trans_param_D = self.discriminator.state_dict()
            # for item, value in model_dict_D.items():
            #     name = '.'.join(item.split('.')[1:])
            #     trans_param_D[name] = value
            #
            # self.generator.load_state_dict(trans_param_G)
            # self.discriminator.load_state_dict(trans_param_D)
            # self.generator.load_state_dict(torch.load(model_dir+'/generator_param_epoch_%d.pkl' % self.epoch))
            # self.discriminator.load_state_dict(torch.load(model_dir+'/discriminator_param_epoch_%d.pkl' % self.epoch))
            G_model_path = model_dir+'/generator_param_epoch_%d.pkl' % self.epoch
            D_model_path = model_dir+'/discriminator_param_epoch_%d.pkl' % self.epoch
            self.load_epoch_network(load_path=G_model_path, network=self.generator, strict=True)
            self.load_epoch_network(load_path=D_model_path, network=self.discriminator, strict=True)
        else:
            # load_path_G = "pretrained_models/SRResNet_param_epoch_64.pkl"
            # print('loading model for G [{:s}] ...'.format(load_path_G))
            # # self.generator.load_state_dict(torch.load(load_path_G), strict=True)
            # self.load_epoch_network(load_path=load_path_G, network=self.generator)
            # Initialize weights
            self.generator.apply(srutils.weights_init_normal)
            self.discriminator.apply(srutils.weights_init_normal)
            # G_model_path = '/home/zju/lyd/SuperResolution-1/Result/SRAWGAN-2/model/generator_param_epoch_48.pkl'
            # D_model_path = '/home/zju/lyd/SuperResolution-1/Result/SRAWGAN-2/model/discriminator_param_epoch_48.pkl'
            # self.load_epoch_network(load_path=G_model_path, network=self.generator, strict=False)
            # self.load_epoch_network(load_path=D_model_path, network=self.discriminator, strict=True)

        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        # optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=self.lr)
        # optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.lr)

        # Multi-GPU Training
        # if torch.cuda.device_count() > 1:
        #     self.generator = nn.DataParallel(self.generator, device_ids=device_ids)
        #     self.discriminator = nn.DataParallel(self.discriminator, device_ids=device_ids)
        #     self.feature_extractor = nn.DataParallel(self.feature_extractor, device_ids=device_ids)
        #     optimizer_G = nn.DataParallel(optimizer_G, device_ids=device_ids)
        #     optimizer_D = nn.DataParallel(optimizer_D, device_ids=device_ids)

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.batch_size, self.num_channels, self.crop_size // self.scale_factor, self.crop_size // self.scale_factor)
        input_hr = Tensor(self.batch_size, self.num_channels, self.crop_size, self.crop_size)
        input_bc = Tensor(self.batch_size, self.num_channels, self.crop_size, self.crop_size)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones(patch)), requires_grad=False)
        fake = Variable(Tensor(np.zeros(patch)), requires_grad=False)
        if cuda:
            valid = valid.cuda()
            fake = fake.cuda()
        #load train dataset
        dataloader = self.load_dataset(dataset='train', max_samples=self.max_train_samples)
        #test_data_loader = self.load_dataset(dataset='test')

        # set the logger
        # G_log_dir = os.path.join(self.save_dir, 'G_logs')
        # if not os.path.exists(G_log_dir):
        #     os.mkdir(G_log_dir)
        # G_logger = Logger(G_log_dir)
        #
        # D_log_dir = os.path.join(self.save_dir, 'D_logs')
        # if not os.path.exists(D_log_dir):
        #     os.mkdir(D_log_dir)
        # D_logger = Logger(D_log_dir)

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
            # # if (epoch + 1) % 20 == 0:
            #     for param_group in optimizer_D.param_groups:
            #         param_group["lr"] /= 2.0
            #     print("Learning rate decay: lr={}".format(optimizer_D.param_groups[0]["lr"]))

            epoch_loss_G = 0
            epoch_loss_D = 0
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
                # pixel loss
                pixel_loss_G = self.criterion_content(gen_hr, imgs_hr)
                # Content loss
                gen_features = self.feature_extractor(gen_hr)
                real_features = Variable(self.feature_extractor(imgs_hr).data, requires_grad=False)
                loss_content = self.criterion_content(gen_features, real_features)
                # Adversarial loss
                if self.relative:
                    pred_g_fake = self.discriminator(gen_hr)
                    pred_d_real = self.discriminator(imgs_hr).detach()
                    loss_gan = (self.criterion_raGAN(pred_d_real - torch.mean(pred_g_fake), False) +
                                self.criterion_raGAN(pred_g_fake - torch.mean(pred_d_real), True)) / 2
                else:
                    # loss_gan = -torch.mean(self.discriminator(gen_hr))
                    gen_validity = self.discriminator(gen_hr)
                    loss_gan = self.criterion_raGAN(gen_validity, True)

                # Total loss
                # loss_G = loss_content + 1e-3 * loss_GAN
                loss_G = pixel_loss_G + self.weight_content * loss_content + self.weight_gan * loss_gan
                # loss_G = loss_content + 0.01 * pixel_loss_G + 0.005 * loss_gan
                # loss_G = mse_loss_G + 2e-6 * loss_content + 1e-3 * loss_gan
                # loss_G = mse_loss_G + 1e-3 * loss_content + 1e-2 * loss_gan

                loss_G.backward()
                optimizer_G.step()
                # optimizer_G.module.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Loss of real and fake images
                if self.relative:
                    pred_d_real = self.discriminator(imgs_hr)
                    pred_d_fake = self.discriminator(gen_hr.detach())  # detach to avoid BP to G
                    l_d_real = self.criterion_raGAN(pred_d_real - torch.mean(pred_d_fake), True)
                    l_d_fake = self.criterion_raGAN(pred_d_fake - torch.mean(pred_d_real), False)
                    loss_D = (l_d_real + l_d_fake) / 2
                else:
                    # loss_D = -torch.mean(self.discriminator(imgs_hr)) + torch.mean(self.discriminator(gen_hr.detach()))
                    loss_real = self.criterion_raGAN(self.discriminator(imgs_hr), True)
                    loss_fake = self.criterion_raGAN(self.discriminator(gen_hr.detach()), False)
                    loss_D = loss_real + loss_fake
                # Gradient penalty
                # gradient_penalty = self.compute_gradient_penalty(self.discriminator, imgs_hr.data, gen_hr.detach().data)
                if self.gp:
                    gradient_penalty = self.gradient_penalty(self.discriminator, imgs_hr.data, gen_hr.detach().data,
                                                             grad_penalty_Lp_norm=self.grad_penalty_Lp_norm, penalty_type=self.penalty_type)
                    loss_D += self.lambda_gp * gradient_penalty

                loss_D.backward()
                optimizer_D.step()
                # optimizer_D.module.step()

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.clip_value, self.clip_value)

                # --------------
                #  Log Progress
                # --------------

                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                      (epoch, self.num_epochs, i, len(dataloader),
                       loss_D.item(), loss_G.item()))

                epoch_loss_G += loss_G.data.item()
                epoch_loss_D += loss_D.data.item()
                # tensorboard logging
                self.logger.scalar_summary('loss_G', loss_G.item(), step + 1)
                self.logger.scalar_summary('loss_D', loss_D.item(), step + 1)

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
                    print_rlt['D_lr'] = optimizer_D.param_groups[0]['lr']
                    print_rlt['time'] = time_elapsed
                    print_rlt['G_loss'] = loss_G.item()
                    print_rlt['D_loss'] = loss_D.item()
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
            avg_loss_D.append(epoch_loss_D / len(dataloader))

            val_psnr, val_ssim, val_ergas, val_lpips = self.validate(epoch=epoch, mode='train', save_img=((epoch + 1) % self.save_epochs == 0))

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
            self.save_epoch_network(save_dir=model_dir, network=self.discriminator,
                                    network_label='discriminator', iter_label=epoch + 1)
            # global epoch
            epoch += 1
            if val_loss_no_improve_count >= val_loss_noimprove_max_count:
                G_model_path = model_dir + '/generator_param_epoch_%d.pkl' % (val_loss_best_step + 1)
                self.load_epoch_network(load_path=G_model_path, network=self.generator)
                # for param_group in optimizer_G.module.param_groups:
                #     param_group["lr"] /= 2.0
                # print("Learning rate decay: lr={}".format(optimizer_G.module.param_groups[0]["lr"]))
                # if self.lr < 0.0001:
                #     for param_group in optimizer_D.module.param_groups:
                #         param_group["lr"] /= 2.0
                #     print("Learning rate decay: lr={}".format(optimizer_D.module.param_groups[0]["lr"]))
                for param_group in optimizer_G.param_groups:
                    param_group["lr"] /= 2.0
                print("optimizer_G_Learning rate decay: lr={}".format(optimizer_G.param_groups[0]["lr"]))
                if self.lr < 0.0001:
                    for param_group in optimizer_D.param_groups:
                        param_group["lr"] /= 2.0
                    print("optimizer_D_Learning rate decay: lr={}".format(optimizer_D.param_groups[0]["lr"]))
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
            #     torch.save(discriminator.state_dict(), 'saved_models/discriminator_%d.pth' % epoch)
            # Save trained parameters of model
            # if (epoch + 1) % self.save_epochs == 0:
            #     # self.save_model(epoch + 1)
            #     self.save_epoch_network(save_dir=model_dir, network=self.generator, network_label='generator', iter_label=epoch+1)
            #     self.save_epoch_network(save_dir=model_dir, network=self.discriminator, network_label='discriminator', iter_label=epoch+1)
            torch.cuda.empty_cache()
        # Plot avg. loss
        srutils.plot_loss([avg_loss_G, avg_loss_D], self.num_epochs, save_dir=self.save_dir)
        srutils.plot_loss([avg_psnr], self.num_epochs, save_dir=self.save_dir, label='PSNR')
        srutils.plot_loss([avg_ssim], self.num_epochs, save_dir=self.save_dir, label='SSIM')
        srutils.plot_loss([avg_ergas], self.num_epochs, save_dir=self.save_dir, label='ERGAS')
        srutils.plot_loss([avg_lpips], self.num_epochs, save_dir=self.save_dir, label='LPIPS')
        print("Training is finished.")

        # Save final trained parameters of model
        self.save_model(epoch=None)

    def validate(self, epoch=0, mode='test', save_img=False):
        # networks
        cuda = True if torch.cuda.is_available() else False
        device_ids = [0,1,2]

        # load model
        if mode == 'test':
            # networks
            self.generator = GeneratorResNet(upscale_factor=self.scale_factor)
            self.load_epoch_model(epoch)

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
            torch.save(self.discriminator.state_dict(), model_dir + '/discriminator_param_epoch_%d.pkl' % epoch)
        else:
            torch.save(self.generator.state_dict(), model_dir + '/generator_param.pkl')
            torch.save(self.discriminator.state_dict(), model_dir + '/discriminator_param.pkl')

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
            # model_dict_G = torch.load(model_dir + '/generator_param_epoch_%d.pkl' % self.epoch)
            # trans_param_G = self.generator.state_dict()
            # for item, value in model_dict_G.items():
            #     name = '.'.join(item.split('.')[1:])
            #     trans_param_G[name] = value
            # self.generator.load_state_dict(trans_param_G)
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
        self.generator = GeneratorResNet(ResidualBlock_Block_WithAttention, n_residual_blocks=12, n_basic_blocks=5,    # 默认n_residual_blocks=12   n_basic_blocks=5
                                         rla_mode='CA-SA', bla_mode='CA-SA', ga_mode='CA-SA', pool_mode='Avg|Max',
                                         addconv=True, upscale_factor=self.scale_factor)
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
        print_rlt['sragan_mse'] = avg_srcnn_mse
        print_rlt['sragan_psnr'] = avg_srcnn_psnr
        print_rlt['sragan_ssim'] = avg_srcnn_ssim
        print_rlt['sragan_ergas'] = avg_srcnn_ergas
        print_rlt['sragan_lpips'] = avg_srcnn_lpips
        self.logger.print_format_results('val', print_rlt)
        print('-----------------------------------')
        del test_data_loader
        torch.cuda.empty_cache()
        return avg_srcnn_psnr, avg_srcnn_ssim, avg_srcnn_ergas, avg_srcnn_lpips

    def mfeNew_validateByClass(self, epoch, save_img=False, modelpath=None):
        # networks
        cuda = True if torch.cuda.is_available() else False
        # load model
        self.generator = GeneratorResNet(ResidualBlock_Block_WithAttention, n_residual_blocks=12, n_basic_blocks=5,    # 默认n_residual_blocks=12   n_basic_blocks=5
                                         rla_mode='CA-SA', bla_mode='CA-SA', ga_mode='CA-SA', pool_mode='Avg|Max',
                                         addconv=True, upscale_factor=self.scale_factor)

        if modelpath is not None:
            self.generator.load_state_dict(torch.load(modelpath), strict=False)
        srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='SRAGAN')

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
            print_rlt['sragan_mse'] = avg_srcnn_mse
            print_rlt['sragan_psnr'] = avg_srcnn_psnr
            print_rlt['sragan_ssim'] = avg_srcnn_ssim
            print_rlt['sragan_ergas'] = avg_srcnn_ergas
            print_rlt['sragan_lpips'] = avg_srcnn_lpips
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
        print_rlt['sragan_mse'] = average_srrgan_mse
        print_rlt['sragan_psnr'] = average_srrgan_psnr
        print_rlt['sragan_ssim'] = average_srrgan_ssim
        print_rlt['sragan_ergas'] = average_srrgan_ergas
        print_rlt['sragan_lpips'] = average_srrgan_lpips
        self.logger.print_format_results('val', print_rlt)
        print('-----------------------------------')
        torch.cuda.empty_cache()

    def mfe_test_single(self, img_fn, modelpath=None):
        # networks
        cuda = True if torch.cuda.is_available() else False
        # Initialize generator
        self.generator = GeneratorResNet(ResidualBlock_Block_WithAttention, n_residual_blocks=12, n_basic_blocks=5,    # 默认n_residual_blocks=12   n_basic_blocks=5
                                         rla_mode='CA-SA', bla_mode='CA-SA', ga_mode='CA-SA', pool_mode='Avg|Max',
                                         addconv=True, upscale_factor=self.scale_factor)

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
        srutils.save_img1(recon_imgs[0].cpu().data, self.save_dir, self.save_dir+'/SR_SRAGAN_%s' % img_name)
        srutils.save_img1(bc_img[0].cpu().data, self.save_dir, self.save_dir+'/SR_Bicubic_%s' % img_name)
        srutils.plot_test_single_result([recon_imgs[0].cpu(), bc_img[0].cpu()], [0, 0], [0, 0], save_path=self.save_dir+'/SR_%s' % img_name)

