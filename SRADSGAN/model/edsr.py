import os
import sys
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.base_networks import *
from torch.utils.data import DataLoader
from data import *
from utils import utils as srutils
from utils.logger import Logger, PrintLogger
from torchvision.transforms import *

from collections import OrderedDict
import time
from PIL import Image
from skimage.measure import compare_ssim
from skimage.measure import compare_mse
from skimage.measure import compare_psnr
import numpy as np
from data.data import get_RGB_trainDataset, get_RGB_testDataset


class Net(torch.nn.Module):
    def __init__(self, num_channels, base_filter, num_residuals, upscale_factor=3):
        super(Net, self).__init__()

        self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, activation=None, norm=None)

        resnet_blocks = []
        for _ in range(num_residuals):
            resnet_blocks.append(ResnetBlock(base_filter, norm=None))
        self.residual_layers = nn.Sequential(*resnet_blocks)

        self.mid_conv = ConvBlock(base_filter, base_filter, 3, 1, 1, activation=None, norm=None)

        # self.upscale4x = nn.Sequential(
        #     Upsample2xBlock(base_filter, base_filter, upsample='ps', activation=None, norm=None),
        #     Upsample2xBlock(base_filter, base_filter, upsample='ps', activation=None, norm=None),
        # )

        # Upsampling layers
        upsampling = []
        upsampling_two = [nn.Conv2d(256, 256 * 4, 3, 1, 1),
                          # nn.BatchNorm2d(64 * 4),
                          nn.PixelShuffle(upscale_factor=2),
                          nn.LeakyReLU(inplace=True)]
        upsampling_three = [nn.Conv2d(256, 256 * 9, 3, 1, 1),
                            # nn.BatchNorm2d(64 * 9),
                            nn.PixelShuffle(upscale_factor=3),
                            nn.LeakyReLU(inplace=True)]
        if (upscale_factor & (upscale_factor - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling += upsampling_two
        elif upscale_factor % 3 == 0:
            for _ in range(int(math.log(upscale_factor, 3))):
                upsampling += upsampling_three

        self.upsampling = nn.Sequential(*upsampling)

        self.output_conv = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None)

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            srutils.weights_init_normal(m, mean=mean, std=std)

    def forward(self, x):
        out = self.input_conv(x)
        residual = out
        out = self.residual_layers(out)
        out = self.mid_conv(out)
        out = torch.add(out, residual)
        # out = self.upscale4x(out)
        out = self.upsampling(out)
        out = self.output_conv(out)
        return out


class EDSR(object):
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

        self.log_dict = OrderedDict()

        self.cuda = True if torch.cuda.is_available() else False
        # import PerceptualSimilarity
        from utils import PerceptualSimilarity
        self.PerceptualModel = PerceptualSimilarity.PerceptualLoss(model='net-lin', net='alex', use_gpu=self.cuda)
        # set the logger
        log_dir = os.path.join(self.root_dir, self.save_dir, 'logs')
        log_freq = 200
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.logger = Logger(self.model_name, log_dir, log_freq)

    def load_dataset(self, dataset='train', max_samples=10000):
        if self.num_channels == 1:
            is_gray = True
        else:
            is_gray = False

        if dataset == 'train':
            print('Loading train dct_datasets...')
            train_set = get_RGB_trainDataset(self.data_dir, self.train_dataset, self.crop_size,
                                                    self.scale_factor, is_gray=is_gray)
            return DataLoader(dataset=train_set, num_workers=self.num_threads, batch_size=self.batch_size,
                              shuffle=True)
        elif dataset == 'test':
            print('Loading test dct_datasets...')
            test_set = get_RGB_testDataset(self.data_dir, self.test_dataset, self.crop_size, self.scale_factor,
                                                 is_gray=is_gray)
            return DataLoader(dataset=test_set, num_workers=self.num_threads,
                              batch_size=self.test_batch_size,
                              shuffle=False)

    def train(self):
        device_ids = [0]
        cuda = True if torch.cuda.is_available() else False
        # cuda = self.gpu_mode
        self.device = torch.device('cuda' if cuda else 'cpu')

        # Initialize generator and discriminator
        self.generator = Net(num_channels=self.num_channels, base_filter=256, num_residuals=32, upscale_factor=self.scale_factor)   #默认num_residuals=32

        srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='EDSR')

        # Losses
        # self.criterion_GAN = torch.nn.MSELoss().to(self.device)
        if self.loss_Lp_norm == "L1":
            self.criterion_content = torch.nn.L1Loss().to(self.device)
        else:
            self.criterion_content = torch.nn.MSELoss().to(self.device)  # L2

        if len(device_ids) > 1:
            self.generator = nn.DataParallel(self.generator).to(self.device)

        if cuda:
            self.generator = self.generator.cuda(device_ids[0])
            self.criterion_content = self.criterion_content.cuda(device_ids[0])

        model_dir = os.path.join(self.save_dir, 'model')
        if self.epoch != 0:
            # Load pretrained models
            G_model_path = model_dir + '/generator_param_epoch_%d.pkl' % self.epoch
            self.load_epoch_network(load_path=G_model_path, network=self.generator)
        else:
            self.generator.apply(srutils.weights_init_normal)

        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.batch_size, self.num_channels, self.crop_size // self.scale_factor,
                          self.crop_size // self.scale_factor)
        input_hr = Tensor(self.batch_size, self.num_channels, self.crop_size, self.crop_size)
        input_bc = Tensor(self.batch_size, self.num_channels, self.crop_size, self.crop_size)

        # load train dataset
        dataloader = self.load_dataset(dataset='train', max_samples=self.max_train_samples)

        sys.stdout = PrintLogger(self.save_dir)
        # ----------
        #  Training
        # ----------
        print('Training is started.')
        avg_loss_G = []
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
                mse_loss_G = self.criterion_content(gen_hr, imgs_hr)

                # Total loss
                loss_G = mse_loss_G

                loss_G.backward()
                optimizer_G.step()
                # optimizer_G.module.step()

                print("[Epoch %d/%d] [Batch %d/%d] [G loss: %f]" %
                      (epoch, self.num_epochs, i, len(dataloader),
                       loss_G.item()))

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
                    # recon_lpips = srutils.compare_lpips(imgs_hr[0], gen_hr[0], use_gpu=cuda)
                    recon_lpips = self.PerceptualModel.forward(gen_hr[0], imgs_hr[0]).detach().item()

                    # mse1 = compare_mse(bc_img, hr_img)
                    bc_psnr = compare_psnr(bc_img, hr_img)
                    bc_ssim = compare_ssim(bc_img, hr_img, multichannel=True)
                    bc_ergas = srutils.compare_ergas2(hr_img, bc_img, scale=self.scale_factor)
                    # bc_lpips = srutils.compare_lpips(imgs_hr[0], imgs_bc[0], use_gpu=cuda)
                    bc_lpips = self.PerceptualModel.forward(imgs_bc[0], imgs_hr[0]).detach().item()

                    result_imgs = [imgs_hr[0].cpu(), imgs_lr[0].cpu(), imgs_bc[0].cpu(), gen_hr[0].cpu()]
                    psnrs = [None, None, bc_psnr, recon_psnr]
                    ssims = [None, None, bc_ssim, recon_ssim]
                    ergas = [None, None, bc_ergas, recon_ergas]
                    lpips = [None, None, bc_lpips, recon_lpips]
                    indicators = {"PSNR": psnrs, "SSIM": ssims, "ERGAS": ergas, "LPIPS": lpips}
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
                    print_rlt['G_lr'] = optimizer_G.param_groups[0]['lr']
                    print_rlt['D_lr'] = 0#optimizer_D.param_groups[0]['lr']
                    print_rlt['time'] = time_elapsed
                    print_rlt['G_loss'] = loss_G.item()
                    print_rlt['D_loss'] = 0#loss_D.item()
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
                del imgs_hr, imgs_bc, imgs_lr, gen_hr#, gen_features, real_features
                torch.cuda.empty_cache()

            # avg. loss per epoch
            avg_loss_G.append(epoch_loss_G / len(dataloader))

            val_psnr, val_ssim, val_ergas, val_lpips = self.validate(epoch=epoch, mode='train',
                                                                     save_img=((epoch + 1) % self.save_epochs == 0))

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
                self.lr /= 2.0
                G_model_path = model_dir + '/generator_param_epoch_%d.pkl' % (val_loss_best_step + 1)
                self.load_epoch_network(load_path=G_model_path, network=self.generator)
                for param_group in optimizer_G.param_groups:
                    param_group["lr"] /= 2.0
                print("Learning rate decay: lr={}".format(optimizer_G.param_groups[0]["lr"]))
                # global epoch
                self.lr /= 2.0
                epoch = val_loss_best_step + 1
                val_loss_no_improve_count = 0
            torch.cuda.empty_cache()
        # Plot avg. loss
        srutils.plot_loss([avg_loss_G], self.num_epochs, save_dir=self.save_dir)
        print("Training is finished.")

        # Save final trained parameters of model
        self.save_model(epoch=None)

    def validate(self, epoch=0, mode='test', save_img=False):
        # networks
        cuda = True if torch.cuda.is_available() else False
        device_ids = [0, 1]

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
                recon_lpips = self.PerceptualModel.forward(recon_imgs[i], imgs_hr[i]).detach().item()

                bc_mse = compare_mse(bc_img, gt_img)
                bc_psnr = compare_psnr(bc_img, gt_img)
                bc_ssim = compare_ssim(bc_img, gt_img, multichannel=True)
                bc_ergas = srutils.compare_ergas2(gt_img, bc_img, scale=self.scale_factor)
                # bc_lpips = srutils.compare_lpips(gt_img, bc_img, use_gpu=cuda)
                bc_lpips = self.PerceptualModel.forward(imgs_bc[i], imgs_hr[i]).detach().item()

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
        self.generator = Net(num_channels=self.num_channels, base_filter=256, num_residuals=32,
                             upscale_factor=self.scale_factor)  # 默认num_residuals=32
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
        print_rlt['edsr_mse'] = avg_srcnn_mse
        print_rlt['edsr_psnr'] = avg_srcnn_psnr
        print_rlt['edsr_ssim'] = avg_srcnn_ssim
        print_rlt['edsr_ergas'] = avg_srcnn_ergas
        print_rlt['edsr_lpips'] = avg_srcnn_lpips
        self.logger.print_format_results('val', print_rlt)
        print('-----------------------------------')
        del test_data_loader
        torch.cuda.empty_cache()
        return avg_srcnn_psnr, avg_srcnn_ssim, avg_srcnn_ergas, avg_srcnn_lpips

    def mfeNew_validateByClass(self, epoch, save_img=False, modelpath=None):
        # networks
        cuda = True if torch.cuda.is_available() else False
        # load model
        self.generator = Net(num_channels=self.num_channels, base_filter=256, num_residuals=32,
                             upscale_factor=self.scale_factor)  # 默认num_residuals=32

        if modelpath is not None:
            self.generator.load_state_dict(torch.load(modelpath), strict=False)
        srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='EDSR_validateByClass')

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
            print_rlt['edsr_mse'] = avg_srcnn_mse
            print_rlt['edsr_psnr'] = avg_srcnn_psnr
            print_rlt['edsr_ssim'] = avg_srcnn_ssim
            print_rlt['edsr_ergas'] = avg_srcnn_ergas
            print_rlt['edsr_lpips'] = avg_srcnn_lpips
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
        print_rlt['edsr_mse'] = average_srrgan_mse
        print_rlt['edsr_psnr'] = average_srrgan_psnr
        print_rlt['edsr_ssim'] = average_srrgan_ssim
        print_rlt['edsr_ergas'] = average_srrgan_ergas
        print_rlt['edsr_lpips'] = average_srrgan_lpips
        self.logger.print_format_results('val', print_rlt)
        print('-----------------------------------')
        torch.cuda.empty_cache()

    def mfe_test_single(self, img_fn, modelpath=None):
        # networks
        cuda = True if torch.cuda.is_available() else False
        # Initialize generator
        self.generator = Net(num_channels=self.num_channels, base_filter=256, num_residuals=32,
                             upscale_factor=self.scale_factor)  # 默认num_residuals=32

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
        srutils.save_img1(recon_imgs[0].cpu().data, self.save_dir, self.save_dir+'/SR_EDSR_%s' % img_name)
        srutils.save_img1(bc_img[0].cpu().data, self.save_dir, self.save_dir+'/SR_Bicubic_%s' % img_name)
        srutils.plot_test_single_result([recon_imgs[0].cpu(), bc_img[0].cpu()], [0, 0], [0, 0], save_path=self.save_dir+'/SR_%s' % img_name)