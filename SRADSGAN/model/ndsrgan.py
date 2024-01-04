"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""
# import argparse
import os
import numpy as np
import math
import time
from collections import OrderedDict
import sys
from PIL import Image
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch

from skimage.measure import compare_ssim
from skimage.measure import compare_mse
from skimage.measure import compare_psnr
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.models import vgg19

from utils import utils as srutils
# import utils.utils as srutils
from utils.logger import Logger, PrintLogger
from data.data import get_training_datasets, get_test_datasets, get_RGB_trainDataset, get_RGB_testDataset

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        vgg19_model = vgg19(pretrained=True)

        # Extracts features at the 11th layer
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:12])

    def forward(self, img):
        out = self.feature_extractor(img)
        return out

def CL(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))

class DenseBlock(nn.Module):
    def __init__(self, nf, nc, CL_num=4):
        super(DenseBlock, self).__init__()
        layer = []
        for j in range(CL_num):
            layer.append(CL(nc * j + nf, nc))
        self.CL_blocks = nn.Sequential(*layer)
        self.conv = nn.Conv2d(nc * CL_num + nf, nf, 3, 1, 1)

    def forward(self, x):
        out1 = x
        for CL_block in self.CL_blocks:
            y = CL_block(x)
            x = torch.cat((x, y), dim=1)
        x = self.conv(x)
        x = out1 + x * 0.2
        return x

class DCRDB(nn.Module):
    def __init__(self, nf, nc):
        super(DCRDB, self).__init__()
        layer = []
        self.RDB1 = DenseBlock(nf, nc)
        self.RDB2 = DenseBlock(nf, nc)
        self.RDB3 = DenseBlock(nf, nc)
        self.conv = nn.Conv2d(nf, nf, 3, 1, 1)

    def forward(self, x):
        out1 = self.RDB1(x)
        out2 = self.RDB2(x + 0.2 * out1)
        out3 = self.RDB3(x + 0.2 * out1 + 0.2 * out2)
        out4 = self.conv(x + 0.2 * out1 + 0.2 * out2 + 0.2 * out3)
        return out4 * 0.2 + x

class DRRDBnet(nn.Module):
    def __init__(self, nf, nc):
        super(DRRDBnet, self).__init__()
        self.DRRDB1 = DCRDB(nf, nc)
        self.DRRDB2 = DCRDB(nf, nc)
        self.DRRDB3 = DCRDB(nf, nc)
        self.DRRDB4 = DCRDB(nf, nc)
        self.DRRDB5 = DCRDB(nf, nc)
        self.DRRDB6 = DCRDB(nf, nc)
        self.DRRDB7 = DCRDB(nf, nc)
        self.DRRDB8 = DCRDB(nf, nc)
        self.DRRDB9 = DCRDB(nf, nc)
        self.DRRDB10 = DCRDB(nf, nc)
        self.DRRDB11 = DCRDB(nf, nc)
        self.DRRDB12 = DCRDB(nf, nc)
        self.DRRDB13 = DCRDB(nf, nc)
        self.DRRDB14 = DCRDB(nf, nc)
        self.DRRDB15 = DCRDB(nf, nc)
        self.DRRDB16 = DCRDB(nf, nc)
        self.DRRDB17 = DCRDB(nf, nc)
        self.DRRDB18 = DCRDB(nf, nc)
        self.DRRDB19 = DCRDB(nf, nc)
        self.DRRDB20 = DCRDB(nf, nc)
        self.DRRDB21 = DCRDB(nf, nc)
        self.DRRDB22 = DCRDB(nf, nc)
        self.DRRDB23 = DCRDB(nf, nc)

    def forward(self, x):
        m1 = self.DRRDB1(x)
        m2 = self.DRRDB2(x + 0.2 * m1)
        m3 = self.DRRDB3(x + 0.2 * m1 + 0.2 * m2)
        m4 = self.DRRDB4(x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3)
        m5 = self.DRRDB5(x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4)
        m6 = self.DRRDB6(x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5)
        m7 = self.DRRDB7(x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6)
        m8 = self.DRRDB8(x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7)
        m9 = self.DRRDB9(x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8)
        m10 = self.DRRDB10(x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9)
        m11 = self.DRRDB11(x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10)
        m12 = self.DRRDB12(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11)
        m13 = self.DRRDB13(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12)
        m14 = self.DRRDB14(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13)
        m15 = self.DRRDB15(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14)
        m16 = self.DRRDB16(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14 + 0.2 * m15)
        m17 = self.DRRDB17(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14 + 0.2 * m15 + 0.2 * m16)
        m18 = self.DRRDB18(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14 + 0.2 * m15 + 0.2 * m16 + 0.2 * m17)
        m19 = self.DRRDB19(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14 + 0.2 * m15 + 0.2 * m16 + 0.2 * m17 + 0.2 * m18)
        m20 = self.DRRDB20(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14 + 0.2 * m15 + 0.2 * m16 + 0.2 * m17 + 0.2 * m18 + 0.2 * m19)
        m21 = self.DRRDB21(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14 + 0.2 * m15 + 0.2 * m16 + 0.2 * m17 + 0.2 * m18 + 0.2 * m19 + 0.2 * m20)
        m22 = self.DRRDB22(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14 + 0.2 * m15 + 0.2 * m16 + 0.2 * m17 + 0.2 * m18 + 0.2 * m19 + 0.2 * m20 + 0.2 * m21)
        m23 = self.DRRDB23(
            x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14 + 0.2 * m15 + 0.2 * m16 + 0.2 * m17 + 0.2 * m18 + 0.2 * m19 + 0.2 * m20 + 0.2 * m21 + 0.2 * m22)
        m24 = x + 0.2 * m1 + 0.2 * m2 + 0.2 * m3 + 0.2 * m4 + 0.2 * m5 + 0.2 * m6 + 0.2 * m7 + 0.2 * m8 + 0.2 * m9 + 0.2 * m10 + 0.2 * m11 + 0.2 * m12 + 0.2 * m13 + 0.2 * m14 + 0.2 * m15 + 0.2 * m16 + 0.2 * m17 + 0.2 * m18 + 0.2 * m19 + 0.2 * m20 + 0.2 * m21 + 0.2 * m22 + 0.2 * m23
        return m24

class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, nf=64, nc=32, upscale_factor=3):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, nf, 3, 1, 1),
        )

        # Residual blocks
        self.DCRDB_block = DRRDBnet(nf=nf, nc=nc)

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

        # Upsampling layers
        upsampling = []
        upsampling_two = [nn.UpsamplingNearest2d(scale_factor=2),
                      nn.Conv2d(nf, nf, 3, 1, 1),
                      nn.LeakyReLU(0.2, inplace=True)]
        upsampling_three = [nn.UpsamplingNearest2d(scale_factor=3),
                      nn.Conv2d(nf, nf, 3, 1, 1),
                      nn.LeakyReLU(0.2, inplace=True)]

        # upsampling_two = [nn.functional.interpolate(input, scale_factor=2, mode='nearest'),
        #               nn.Conv2d(nf, nf, 3, 1, 1),
        #               nn.LeakyReLU(0.2, inplace=True)]
        # upsampling_three = [nn.functional.interpolate(input, scale_factor=3, mode='nearest'),
        #               nn.Conv2d(nf, nf, 3, 1, 1),
        #               nn.LeakyReLU(0.2, inplace=True)]

        if (upscale_factor & (upscale_factor - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling += upsampling_two
        elif upscale_factor % 3 == 0:
            for _ in range(int(math.log(upscale_factor, 3))):
                upsampling += upsampling_three

        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(nf, out_channels, 3, 1, 1))

    def forward(self, x):
        out = self.conv1(x)
        trunk = self.conv2(self.DCRDB_block(out))
        out = out + trunk
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for out_filters, stride, normalize in [(64, 2, False),
                                                (128, 2, True),
                                                (256, 2, True),
                                                (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        # Output layer
        layers.append(nn.Conv2d(out_filters, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

class NDSRGAN(object):
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

        self.log_dict = OrderedDict()


        self.cuda = True if torch.cuda.is_available() else False
        from utils import PerceptualSimilarity
        self.PerceptualModel = PerceptualSimilarity.PerceptualLoss(model='net-lin', net='alex', use_gpu=self.cuda)


        # set the logger
        log_dir = os.path.join(self.root_dir, self.save_dir, 'logs')
        log_freq = 200
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.logger = Logger(self.model_name, log_dir, log_freq)

    def load_dataset(self, dataset='train'):
        if self.num_channels == 1:
            is_gray = True
        else:
            is_gray = False

        if dataset == 'train':
            print('Loading train datasets...')
            # train_set = get_training_set_randomcrop(self.data_dir, self.train_dataset, self.crop_size, self.scale_factor,
            #                               is_gray=is_gray)
            train_set = get_RGB_trainDataset(self.data_dir, self.train_dataset, self.crop_size, self.scale_factor, is_gray=is_gray)
            # print('srgan______________________train_set', train_set)
            return DataLoader(dataset=train_set, num_workers=self.num_threads, batch_size=self.batch_size,
                              shuffle=True, drop_last=True)
        elif dataset == 'test':
            print('Loading test datasets...')
            # test_set = get_test_sets(self.data_dir, self.test_dataset, self.scale_factor, is_gray=is_gray)
            test_set = get_RGB_testDataset(self.data_dir, self.test_dataset, self.crop_size, self.scale_factor,
                                                 is_gray=is_gray)
            return DataLoader(dataset=test_set, num_workers=self.num_threads,
                              batch_size=self.test_batch_size,
                              shuffle=False, drop_last=True)

    def train(self):
        cuda = True if torch.cuda.is_available() else False

        # Calculate output of image discriminator (PatchGAN)
        # patch_h, patch_w = int(self.crop_size / 2 ** 4), int(self.crop_size / 2 ** 4)
        # patch_h, patch_w = math.ceil(self.crop_size / 2 ** 4), math.ceil(self.crop_size / 2 ** 4)
        patch_h, patch_w = int(self.crop_size / 2 ** 3)-2, int(self.crop_size / 2 ** 3)-2
        patch = (self.batch_size, 1, patch_h, patch_w)

        # Initialize generator and discriminator
        self.generator = GeneratorResNet(in_channels=3, out_channels=3, nf=64, nc=32, upscale_factor=self.scale_factor)
        self.discriminator = Discriminator()
        self.feature_extractor = FeatureExtractor()

        # Losses
        self.criterion_GAN = torch.nn.SmoothL1Loss()
        # self.criterion_content = torch.nn.L1Loss()   # MAE L1
        # self.criterion_content = torch.nn.MSELoss()   # MSE  L2

        self.criterion_content = torch.nn.SmoothL1Loss()  #  smoothL1

        if cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
            self.feature_extractor = self.feature_extractor.cuda()
            self.criterion_GAN = self.criterion_GAN.cuda()
            self.criterion_content = self.criterion_content.cuda()

        model_dir = os.path.join(self.save_dir, 'model')
        if self.epoch != 0:
            # Load pretrained models
            self.generator.load_state_dict(torch.load(model_dir + '/generator_param_epoch_%d.pkl' % self.epoch))
            self.discriminator.load_state_dict(torch.load(model_dir + '/discriminator_param_epoch_%d.pkl' % self.epoch))
        else:
            # Initialize weights
            self.generator.apply(weights_init_normal)
            self.discriminator.apply(weights_init_normal)

        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        # optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=self.lr)
        # optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.lr)

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.batch_size, self.num_channels, self.crop_size // self.scale_factor, self.crop_size // self.scale_factor)
        input_hr = Tensor(self.batch_size, self.num_channels, self.crop_size, self.crop_size)
        input_bc = Tensor(self.batch_size, self.num_channels, self.crop_size, self.crop_size)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones(patch)), requires_grad=False)
        fake = Variable(Tensor(np.zeros(patch)), requires_grad=False)

        # load train dataset
        dataloader = self.load_dataset(dataset='train')
        print('Number of val images in [{:s}]: {:d}'.format("TrainingSet", len(dataloader)))
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
        step = 0
        start_time = time.time()
        global epoch
        epoch = self.epoch


        tensor_to_img = ToPILImage()
        for epoch in range(self.epoch, self.num_epochs):
            # learning rate is decayed by a factor of 10 every 20 epochs
            if (epoch + 1) % 12 == 0:
                for param_group in optimizer_G.param_groups:
                    param_group["lr"] /= 2.0
                print("Learning rate decay: lr={}".format(optimizer_G.param_groups[0]["lr"]))
            epoch_loss_G = 0
            epoch_loss_D = 0
            for i, (input, target, bc_input, imgpath) in enumerate(dataloader):
                #print('______________srgan______________318', input.shape, target.shape, bc_input.shape, imgpath)

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
                # Adversarial loss
                gen_validity = self.discriminator(gen_hr)
                loss_GAN = self.criterion_GAN(gen_validity, valid)

                # Content loss
                gen_features = self.feature_extractor(gen_hr)
                real_features = Variable(self.feature_extractor(imgs_hr).data, requires_grad=False)
                loss_content = self.criterion_content(gen_features, real_features)

                # add by lyd
                #loss_gan = torch.mean(self.discriminator(gen_hr))
                mse_loss_G = self.criterion_GAN(gen_hr, imgs_hr)
                # Total loss
                # loss_G = loss_content + 1e-3 * loss_GAN
                loss_G = 1e-2 * mse_loss_G + loss_content + 2.5e-3 * loss_GAN

                loss_G.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Loss of real and fake images
                loss_real = self.criterion_GAN(self.discriminator(imgs_hr), valid)
                loss_fake = self.criterion_GAN(self.discriminator(gen_hr.detach()), fake)

                # Gradient penalty
                # gradient_penalty = self.compute_gradient_penalty(self.discriminator, imgs_hr.data, gen_hr.detach().data)

                # Total loss
                loss_D = (loss_real + loss_fake) / 2
                #loss_D = -torch.mean(self.discriminator(imgs_hr)) + torch.mean(
                #     self.discriminator(gen_hr.detach()))  # + self.lambda_gp * gradient_penalty

                loss_D.backward()
                optimizer_D.step()

                # --------------
                #  Log Progress
                # --------------

                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                      (epoch, self.num_epochs, i, len(dataloader),
                       loss_D.item(), loss_G.item()))

                epoch_loss_G += loss_G.item()
                epoch_loss_D += loss_D.item()
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
                    # srutils.plot_test_result_by_name(result_imgs, psnrs, ssims, batches_done, imgpath[0], save_dir=self.save_dir,
                    #                           is_training=True, show=False)

                    indicators = {"PSNR": psnrs, "SSIM": ssims, "ERGAS": ergas, "LPIPS": lpips}
                    srutils.plot_result_by_name(result_imgs, indicators, batches_done, imgpath[0],
                                                     save_dir=self.save_dir, is_training=True, show=False)

                    # set output log
                    time_elapsed = time.time() - start_time
                    print_rlt = OrderedDict()
                    print_rlt['model'] = self.model_name
                    print_rlt['epoch'] = epoch
                    print_rlt['iters'] = step
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

            self.validate(epoch=epoch, mode='train')
            # avg. loss per epoch
            avg_loss_G.append(epoch_loss_G / len(dataloader))
            avg_loss_D.append(epoch_loss_D / len(dataloader))

            # if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            #     # Save model checkpoints
            #     torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % epoch)
            #     torch.save(discriminator.state_dict(), 'saved_models/discriminator_%d.pth' % epoch)
            # Save trained parameters of model
            if (epoch + 1) % self.save_epochs == 0:
                self.save_model(epoch + 1)
                epoch += 1

        # Plot avg. loss
        srutils.plot_loss([avg_loss_G, avg_loss_D], self.num_epochs, save_dir=self.save_dir)
        print("Training is finished.")

        # Save final trained parameters of model
        self.save_model(epoch=None)

    def validate(self, epoch=0, mode='test'):
        # networks
        cuda = True if torch.cuda.is_available() else False

        # load model
        if mode == 'test':
            # networks
            self.generator = GeneratorResNet(in_channels=3, out_channels=3, nf=64, nc=32, upscale_factor=self.scale_factor)
            self.load_model()

        if self.gpu_mode:
            self.generator.cuda()

        # load dataset
        test_data_loader = self.load_dataset(dataset='test')
        print('Number of val images in [{:s}]: {:d}'.format("TestingSet", len(test_data_loader)))

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        #print('_______________Tensor', Tensor)
        input_lr = Tensor(self.test_batch_size, self.num_channels, self.hr_height // self.scale_factor,
                          self.hr_width // self.scale_factor)
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
        loss_mse = 0.0
        for iter, (input, target, bc_input, imgpath) in enumerate(test_data_loader):
            #print('______________test_data_loader______________495', input.shape, target.shape, bc_input.shape, imgpath)
            # Configure model input
            imgs_lr = Variable(input_lr.copy_(input))
            #print('before____________________imgs_lr', imgs_lr, imgs_lr.shape,imgs_lr.size)
            #print('____________________imgs_lr', imgs_lr, imgs_lr.shape,imgs_lr.size)
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

                if (epoch + 1) % self.save_epochs == 0 and iter % 50 == 0:
                    # save result images
                    result_imgs = [imgs_hr[i].cpu(), imgs_lr[i].cpu(), imgs_bc[i].cpu(), recon_imgs[i].cpu()]
                    psnrs = [None, None, bc_psnr, recon_psnr]
                    ssims = [None, None, bc_ssim, recon_ssim]
                    ergas = [None, None, bc_ergas, recon_ergas]
                    lpips = [None, None, bc_lpips, recon_lpips]
                    # srutils.plot_test_result_by_name(result_imgs, psnrs, ssims, epoch, imgpath[i], save_dir=self.save_dir)

                    indicators = {"PSNR": psnrs, "SSIM": ssims, "ERGAS": ergas, "LPIPS": lpips}
                    srutils.plot_result_by_name(result_imgs, indicators, epoch, imgpath[i], save_dir=self.save_dir)

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
                # loss_mse += mse
                print("Saving %d test result images..." % img_num)

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
        self.generator = GeneratorResNet(in_channels=3, out_channels=3, nf=64, nc=32, upscale_factor=self.scale_factor)
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


                # if save_img and iter % 50 == 0:
                #     # save result images
                #     result_imgs = [imgs_hr[i].cpu(), imgs_lr[i].cpu(), imgs_bc[i].cpu(), recon_imgs[i].cpu()]
                #     psnrs = [None, None, bc_psnr, recon_psnr]
                #     ssims = [None, None, bc_ssim, recon_ssim]
                #     ergas = [None, None, bc_ergas, recon_ergas]
                #     lpips = [None, None, bc_lpips, recon_lpips]
                #     indicators = {"PSNR": psnrs, "SSIM": ssims, "ERGAS": ergas, "LPIPS": lpips}
                #     # srutils.plot_test_result_by_name(result_imgs, psnrs, ssims, epoch, imgpath[i],
                #                                      # save_dir=self.save_dir)
                #     srutils.plot_result_by_name(result_imgs, indicators, epoch, imgpath[i],
                #                                 save_dir=self.save_dir)

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
        print_rlt['ndsrgan_mse'] = avg_srcnn_mse
        print_rlt['ndsrgan_psnr'] = avg_srcnn_psnr
        print_rlt['ndsrgan_ssim'] = avg_srcnn_ssim
        print_rlt['ndsrgan_ergas'] = avg_srcnn_ergas
        print_rlt['ndsrgan_lpips'] = avg_srcnn_lpips
        self.logger.print_format_results('val', print_rlt)
        print('-----------------------------------')
        del test_data_loader
        torch.cuda.empty_cache()
        return avg_srcnn_psnr, avg_srcnn_ssim, avg_srcnn_ergas, avg_srcnn_lpips

    def mfeNew_validateByClass(self, epoch, save_img=False, modelpath=None):
        # networks
        cuda = True if torch.cuda.is_available() else False
        # load model
        self.generator = GeneratorResNet(in_channels=3, out_channels=3, nf=64, nc=32, upscale_factor=self.scale_factor)

        if modelpath is not None:
            self.generator.load_state_dict(torch.load(modelpath), strict=False)
        srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='NDSRGAN')

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
            print_rlt['ndsrgan_mse'] = avg_srcnn_mse
            print_rlt['ndsrgan_psnr'] = avg_srcnn_psnr
            print_rlt['ndsrgan_ssim'] = avg_srcnn_ssim
            print_rlt['ndsrgan_ergas'] = avg_srcnn_ergas
            print_rlt['ndsrgan_lpips'] = avg_srcnn_lpips
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
        print_rlt['ndsrgan_mse'] = average_srrgan_mse
        print_rlt['ndsrgan_psnr'] = average_srrgan_psnr
        print_rlt['ndsrgan_ssim'] = average_srrgan_ssim
        print_rlt['ndsrgan_ergas'] = average_srrgan_ergas
        print_rlt['ndsrgan_lpips'] = average_srrgan_lpips
        self.logger.print_format_results('val', print_rlt)
        print('-----------------------------------')
        torch.cuda.empty_cache()

    def mfe_test_single(self, img_fn, modelpath=None):
        # networks
        cuda = True if torch.cuda.is_available() else False
        # Initialize generator
        self.generator = GeneratorResNet(in_channels=3, out_channels=3, nf=64, nc=32, upscale_factor=self.scale_factor)

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
        srutils.save_img1(recon_imgs[0].cpu().data, self.save_dir, self.save_dir+'/SR_NDSRGAN_%s' % img_name)
        srutils.save_img1(bc_img[0].cpu().data, self.save_dir, self.save_dir+'/SR_Bicubic_%s' % img_name)
        srutils.plot_test_single_result([recon_imgs[0].cpu(), bc_img[0].cpu()], [0, 0], [0, 0], save_path=self.save_dir+'/SR_%s' % img_name)


