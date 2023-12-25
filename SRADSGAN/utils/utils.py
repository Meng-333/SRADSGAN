import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import math
from math import log10
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import os
import imageio
# from scipy.misc import imsave
# from edge_detector import edge_detect
import sewar


# import thop


class LayerActivations:
    features = None

    def __init__(self, layer):
        # for name, midlayer in model.__modules.items()
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


# For logger
def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = torch.from_numpy(x).cuda()
    return Variable(x)


# Plot losses
def plot_loss(avg_losses, num_epochs, save_dir='', label='loss', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    temp = 0.0
    for i in range(len(avg_losses)):
        temp = max(np.max(avg_losses[i]), temp)
    ax.set_ylim(0, temp * 1.1)
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss values')

    if len(avg_losses) == 1:
        plt.plot(avg_losses[0], label=label)
    else:
        plt.plot(avg_losses[0], label='G_loss')
        plt.plot(avg_losses[1], label='D_loss')
    plt.legend()

    # save figure
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_fn = label + '_values_epoch_{:d}'.format(num_epochs) + '.png'
    save_fn = os.path.join(save_dir, save_fn)
    plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


# Make gif
def make_gif(dataset, num_epochs, save_dir='results/'):
    gen_image_plots = []
    for epoch in range(num_epochs):
        # plot for generating gif
        save_fn = save_dir + 'Result_epoch_{:d}'.format(epoch + 1) + '.png'
        gen_image_plots.append(imageio.imread(save_fn))

    imageio.mimsave(save_dir + dataset + '_result_epochs_{:d}'.format(num_epochs) + '.gif', gen_image_plots, fps=5)


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
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


def weights_init_kaming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


def save_img(img, img_num, save_dir='', is_training=False):
    # img.clamp(0, 1)
    if list(img.shape)[0] == 3:
        save_img = img * 255.0
        save_img = save_img.clamp(0, 255).numpy().transpose(1, 2, 0).astype(np.uint8)
        # img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)
    else:
        save_img = img.squeeze().clamp(0, 1).numpy()

    # save img
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if is_training:
        save_fn = save_dir + '/SR_result_epoch_{:d}'.format(img_num) + '.png'
    else:
        save_fn = save_dir + '/SR_result_{:d}'.format(img_num) + '.png'
    # imsave(save_fn, save_img)
    imageio.imsave(save_fn, save_img)

def show_attention_on_image(image, mask, save_dir, img_path):
    mask = np.uint8(255 * mask.detach().numpy())
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    attention = heatmap + np.float32(image)
    attention = attention / np.max(attention)
    # save img
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # imsave(img_path, save_img)
    # imageio.imsave(img_path, save_img)
    cv2.imwrite(img_path, np.uint8(255*attention))

def save_img1(img, save_dir, img_path, cuda=True):
    # img.clamp(0, 1)
    if list(img.shape)[0] == 3:
        if cuda:
            img = img * 255.0
        save_img = img.clamp(0, 255).detach().numpy().transpose(1, 2, 0).astype(np.uint8)
        # save_img = (((img - img.min()) * 255) / (img.max() - img.min())).detach().numpy().transpose(1, 2, 0).astype(np.uint8)
    else:
        # save_img = img.squeeze().clamp(0, 1).detach().numpy()
        if not cuda:
            save_img = (((img - img.min()) * 255) / (img.max() - img.min())).detach().numpy().astype(np.uint8)
        else:
            save_img = (((img - img.min())) / (img.max() - img.min())).detach().numpy().astype(np.uint8)

    # save img
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # imsave(img_path, save_img)
    imageio.imsave(img_path, save_img)


def plot_test_result(imgs, psnrs, img_num, save_dir='', is_training=False, show_label=True, show=False):
    size = list(imgs[0].shape)
    if show_label:
        h = 3
        w = h * len(imgs)
    else:
        h = size[2] / 100
        w = size[1] * len(imgs) / 100

    fig, axes = plt.subplots(1, len(imgs), figsize=(w, h))
    # axes.axis('off')
    for i, (ax, img, psnr) in enumerate(zip(axes.flatten(), imgs, psnrs)):
        ax.axis('off')
        ax.set_adjustable('box')
        if list(img.shape)[0] == 3:
            # Scale to 0-255
            # img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)
            img *= 255.0
            img = img.clamp(0, 255).numpy().transpose(1, 2, 0).astype(np.uint8)

            ax.imshow(img, cmap=None, aspect='equal')
        else:
            # img = ((img - img.min()) / (img.max() - img.min())).numpy().transpose(1, 2, 0)
            img = img.squeeze().clamp(0, 1).numpy()
            ax.imshow(img, cmap='gray', aspect='equal')

        if show_label:
            ax.axis('on')
            if i == 0:
                ax.set_xlabel('HR image')
            elif i == 1:
                ax.set_xlabel('LR image')
            elif i == 2:
                ax.set_xlabel('Bicubic (PSNR: %.2fdB)' % psnr)
            elif i == 3:
                ax.set_xlabel('SR image (PSNR: %.2fdB)' % psnr)

    if show_label:
        plt.tight_layout()
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(bottom=0)
        plt.subplots_adjust(top=1)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(left=0)

    # save figure
    result_dir = os.path.join(save_dir, 'plot')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if is_training:
        save_fn = result_dir + '/Train_result_epoch_{:d}'.format(img_num) + '.png'
    else:
        save_fn = result_dir + '/Test_result_{:d}'.format(img_num) + '.png'
    plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def mfe_plot_test_result2(imgs, mses, psnrs, ssims, ergas, lpips, img_num, save_dir='', is_training=False, show_label=True, show=False):
    size = list(imgs[0].shape)
    if show_label:
        h = 3
        w = h * len(imgs)
    else:
        h = size[2] / 100
        w = size[1] * len(imgs) / 100

    fig, axes = plt.subplots(1, len(imgs), figsize=(w, h))
    # axes.axis('off')
    for i, (ax, img, mse, psnr, ssim, erga, lpip) in enumerate(zip(axes.flatten(), imgs, mses, psnrs, ssims, ergas, lpips)):
        ax.axis('off')
        ax.set_adjustable('box')
        if list(img.shape)[0] == 3:
            # Scale to 0-255
            # img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)
            img *= 255.0
            img = img.clamp(0, 255).detach().numpy().transpose(1, 2, 0).astype(np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
        else:
            # img = ((img - img.min()) / (img.max() - img.min())).numpy().transpose(1, 2, 0)
            img = img.squeeze().clamp(0, 1).detach().numpy()
            ax.imshow(img, cmap='gray', aspect='equal')

        if show_label:
            ax.axis('on')
            if i == 0:
                ax.set_xlabel('HR image')
            elif i == 1:
                ax.set_xlabel('LR image')
            elif i == 2:
                ax.set_xlabel('Bicubic (MSE: %.2fdB)\n (PSNR: %.2fdB)\n (SSIM: %.2f)\n (ERGA: %.2f)\n (LPIP: %.2f)' % (mse, psnr, ssim, erga, lpip))
            elif i == 3:
                ax.set_xlabel('SR image (MSE: %.2fdB)\n (PSNR: %.2fdB)\n (SSIM: %.2f)\n (ERGA: %.2f)\n (LPIP: %.2f)' % (mse, psnr, ssim, erga, lpip))

    if show_label:
        plt.tight_layout()
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(bottom=0)
        plt.subplots_adjust(top=1)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(left=0)

    # save figure
    result_dir = os.path.join(save_dir, 'plot')
    if is_training:
        save_fn = result_dir + '/Train_result_{:d}'.format(img_num) + '.png'
    else:
        result_dir = os.path.join(save_dir, 'testplot')
        save_fn = result_dir + '/Test_result_{:d}'.format(img_num) + '.png'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()




def plot_test_result2(imgs, psnrs, ssims, img_num, save_dir='', is_training=False, show_label=True, show=False):
    size = list(imgs[0].shape)
    if show_label:
        h = 3
        w = h * len(imgs)
    else:
        h = size[2] / 100
        w = size[1] * len(imgs) / 100

    fig, axes = plt.subplots(1, len(imgs), figsize=(w, h))
    # axes.axis('off')
    for i, (ax, img, psnr, ssim) in enumerate(zip(axes.flatten(), imgs, psnrs, ssims)):
        ax.axis('off')
        ax.set_adjustable('box')
        if list(img.shape)[0] == 3:
            # Scale to 0-255
            # img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)
            img *= 255.0
            img = img.clamp(0, 255).detach().numpy().transpose(1, 2, 0).astype(np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
        else:
            # img = ((img - img.min()) / (img.max() - img.min())).numpy().transpose(1, 2, 0)
            img = img.squeeze().clamp(0, 1).detach().numpy()
            ax.imshow(img, cmap='gray', aspect='equal')

        if show_label:
            ax.axis('on')
            if i == 0:
                ax.set_xlabel('HR image')
            elif i == 1:
                ax.set_xlabel('LR image')
            elif i == 2:
                ax.set_xlabel('Bicubic (PSNR: %.2fdB)\n (SSIM: %.2f)' % (psnr, ssim))
            elif i == 3:
                ax.set_xlabel('SR image (PSNR: %.2fdB)\n (SSIM: %.2f)' % (psnr, ssim))

    if show_label:
        plt.tight_layout()
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(bottom=0)
        plt.subplots_adjust(top=1)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(left=0)

    # save figure
    result_dir = os.path.join(save_dir, 'plot')
    if is_training:
        save_fn = result_dir + '/Train_result_{:d}'.format(img_num) + '.png'
    else:
        result_dir = os.path.join(save_dir, 'testplot')
        save_fn = result_dir + '/Test_result_{:d}'.format(img_num) + '.png'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def plot_test_result_by_path(imgs, psnrs, ssims, save_path, show_label=True, show=False):
    size = list(imgs[0].shape)
    if show_label:
        h = 3
        w = h * len(imgs)
    else:
        h = size[2] / 100
        w = size[1] * len(imgs) / 100

    fig, axes = plt.subplots(1, len(imgs), figsize=(w, h))
    # axes.axis('off')
    for i, (ax, img, psnr, ssim) in enumerate(zip(axes.flatten(), imgs, psnrs, ssims)):
        ax.axis('off')
        ax.set_adjustable('box')
        if list(img.shape)[0] == 3:
            # Scale to 0-255
            # img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)
            img *= 255.0
            img = img.clamp(0, 255).detach().numpy().transpose(1, 2, 0).astype(np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
        else:
            # img = ((img - img.min()) / (img.max() - img.min())).numpy().transpose(1, 2, 0)
            img = img.squeeze().clamp(0, 1).detach().numpy()
            ax.imshow(img, cmap='gray', aspect='equal')

        if show_label:
            ax.axis('on')
            if i == 0:
                ax.set_xlabel('HR image')
            elif i == 1:
                ax.set_xlabel('LR image')
            elif i == 2:
                ax.set_xlabel('Bicubic (PSNR: %.2fdB)\n (SSIM: %.2f)' % (psnr, ssim))
            elif i == 3:
                ax.set_xlabel('SR image (PSNR: %.2fdB)\n (SSIM: %.2f)' % (psnr, ssim))

    if show_label:
        plt.tight_layout()
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(bottom=0)
        plt.subplots_adjust(top=1)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(left=0)

    # save figure
    plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_test_result_by_name(imgs, psnrs, ssims, img_num, imagepath, save_dir='', is_training=False, show_label=True,
                             show=False):
    size = list(imgs[0].shape)
    if show_label:
        h = 3
        w = h * len(imgs)
    else:
        h = size[2] / 100
        w = size[1] * len(imgs) / 100

    fig, axes = plt.subplots(1, len(imgs), figsize=(w, h))
    #axes.axis('off')
    for i, (ax, img, psnr, ssim) in enumerate(zip(axes.flatten(), imgs, psnrs, ssims)):
        ax.axis('off')
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.set_adjustable('box')
        # ax.set_adjustable('box-forced')
        if list(img.shape)[0] == 3:
            # Scale to 0-255
            # img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)
            img *= 255.0
            img = img.clamp(0, 255).detach().numpy().transpose(1, 2, 0).astype(np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
        else:
            # img = ((img - img.min()) / (img.max() - img.min())).numpy().transpose(1, 2, 0)
            img = img.squeeze().clamp(0, 1).detach().numpy()
            ax.imshow(img, cmap='gray', aspect='equal')

        if show_label:
            ax.axis('on')
            if i == 0:
                ax.set_xlabel('HR image')
            elif i == 1:
                ax.set_xlabel('LR image')
            elif i == 2:
                ax.set_xlabel('Bicubic (PSNR: %.4fdB)\n (SSIM: %.5f)' % (psnr, ssim))
            elif i == 3:
                ax.set_xlabel('SR image (PSNR: %.4fdB)\n (SSIM: %.5f)' % (psnr, ssim))

    if show_label:
        plt.tight_layout()
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(bottom=0)
        plt.subplots_adjust(top=1)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(left=0)

    img_name = os.path.splitext(os.path.basename(imagepath))[0]

    # save figure
    result_dir = os.path.join(save_dir, 'plot', img_name)
    if not is_training:
        result_dir = os.path.join(save_dir, 'testplot', img_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    save_img_path = os.path.join(result_dir, '{:s}_{:d}.png'.format(img_name, img_num))
    plt.savefig(save_img_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_result_by_name(imgs, indicators, img_num, imagepath, save_dir='', is_training=False, is_validation=False,
                        classname=None, show_label=True, show=False):
    size = list(imgs[0].shape)
    if show_label:
        h = 3
        w = h * len(imgs)
    else:
        h = size[2] / 100
        w = size[1] * len(imgs) / 100

    fig, axes = plt.subplots(1, len(imgs), figsize=(w, h))

    keys = list(indicators.keys())
    keyscount = len(keys)
    # axes.axis('off')
    for i, (ax, img) in enumerate(zip(axes.flatten(), imgs)):
        ax.axis('off')
        ax.set_adjustable('box')
        # ax.set_adjustable('box-forced')
        if list(img.shape)[0] == 3:
            # Scale to 0-255
            # img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)
            img *= 255.0
            img = img.clamp(0, 255).detach().numpy().transpose(1, 2, 0).astype(np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
        else:
            # img = ((img - img.min()) / (img.max() - img.min())).numpy().transpose(1, 2, 0)
            img = img.squeeze().clamp(0, 1).detach().numpy()
            ax.imshow(img, cmap='gray', aspect='equal')

        if show_label:
            ax.axis('on')
            if i == 0:
                ax.set_xlabel('HR image')
            elif i == 1:
                ax.set_xlabel('LR image')
            elif i == 2:
                label = "Bicubic"
                for j in range(keyscount):
                    label += '(%s: %.4f)\n' % (keys[j], indicators.get(keys[j])[i])
                ax.set_xlabel(label)
            elif i == 3:
                label = "SR image"
                for j in range(keyscount):
                    label += '(%s: %.4f)\n' % (keys[j], indicators.get(keys[j])[i])
                ax.set_xlabel(label)

    if show_label:
        plt.tight_layout()
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(bottom=0)
        plt.subplots_adjust(top=1)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(left=0)

    img_name = os.path.splitext(os.path.basename(imagepath))[0]

    # save figure
    result_dir = os.path.join(save_dir, 'plot', img_name)
    if not is_training:
        result_dir = os.path.join(save_dir, 'testplot', img_name)
        if is_validation:
            result_dir = os.path.join(save_dir, 'validate', classname)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    save_img_path = os.path.join(result_dir, '{:s}_{:d}.png'.format(img_name, img_num))
    plt.savefig(save_img_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_test_result_by_class(imgs, psnrs, ssims, img_num, classname, imagepath, save_dir='', is_training=False,
                              show_label=True, show=False):
    size = list(imgs[0].shape)
    if show_label:
        h = 3
        w = h * len(imgs)
    else:
        h = size[2] / 100
        w = size[1] * len(imgs) / 100

    fig, axes = plt.subplots(1, len(imgs), figsize=(w, h))
    # axes.axis('off')
    for i, (ax, img, psnr, ssim) in enumerate(zip(axes.flatten(), imgs, psnrs, ssims)):
        ax.axis('off')
        ax.set_adjustable('box')
        if list(img.shape)[0] == 3:
            # Scale to 0-255
            # img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)
            img *= 255.0
            img = img.clamp(0, 255).detach().numpy().transpose(1, 2, 0).astype(np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
        else:
            # img = ((img - img.min()) / (img.max() - img.min())).numpy().transpose(1, 2, 0)
            img = img.squeeze().clamp(0, 1).detach().numpy()
            ax.imshow(img, cmap='gray', aspect='equal')

        if show_label:
            ax.axis('on')
            if i == 0:
                ax.set_xlabel('HR image')
            elif i == 1:
                ax.set_xlabel('LR image')
            elif i == 2:
                ax.set_xlabel('Bicubic (PSNR: %.3fdB)\n (SSIM: %.5f)' % (psnr, ssim))
            elif i == 3:
                ax.set_xlabel('SR image (PSNR: %.3fdB)\n (SSIM: %.5f)' % (psnr, ssim))

    if show_label:
        plt.tight_layout()
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(bottom=0)
        plt.subplots_adjust(top=1)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(left=0)

    img_name = os.path.splitext(os.path.basename(imagepath))[0]

    # save figure
    result_dir = os.path.join(save_dir, 'plot', img_name)
    if not is_training:
        result_dir = os.path.join(save_dir, 'validate', classname, img_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    save_img_path = os.path.join(result_dir, '{:s}_{:d}.png'.format(img_name, img_num))
    plt.savefig(save_img_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_test_single_result(imgs, psnrs, ssims, save_path='', is_training=False, show_label=True, show=False):
    size = list(imgs[0].shape)
    if show_label:
        h = 3
        w = h * len(imgs)
    else:
        h = size[2] / 100
        w = size[1] * len(imgs) / 100

    fig, axes = plt.subplots(1, len(imgs), figsize=(w, h))
    # axes.axis('off')
    for i, (ax, img, psnr, ssim) in enumerate(zip(axes.flatten(), imgs, psnrs, ssims)):
        ax.axis('off')
        ax.set_adjustable('box')
        if list(img.shape)[0] == 3:
            # Scale to 0-255
            # img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)
            img *= 255.0
            img = img.clamp(0, 255).detach().numpy().transpose(1, 2, 0).astype(np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
        else:
            # img = ((img - img.min()) / (img.max() - img.min())).numpy().transpose(1, 2, 0)
            img = img.squeeze().clamp(0, 1).detach().numpy()
            ax.imshow(img, cmap='gray', aspect='equal')

        if show_label:
            ax.axis('on')
            if i == 0:
                ax.set_xlabel('HR image')
            elif i == 1:
                ax.set_xlabel('LR image')
            elif i == 2:
                ax.set_xlabel('Bicubic (PSNR: %.3fdB)\n (SSIM: %.5f)' % (psnr, ssim))
            elif i == 3:
                ax.set_xlabel('SR image (PSNR: %.3fdB)\n (SSIM: %.5f)' % (psnr, ssim))

    if show_label:
        plt.tight_layout()
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(bottom=0)
        plt.subplots_adjust(top=1)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(left=0)

    # save figure
    plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()

def shave(imgs, border_size=0):
    size = list(imgs.shape)
    if len(size) == 4:
        shave_imgs = torch.FloatTensor(size[0], size[1], size[2] - border_size * 2, size[3] - border_size * 2)
        for i, img in enumerate(imgs):
            shave_imgs[i, :, :, :] = img[:, border_size:-border_size, border_size:-border_size]
        return shave_imgs
    else:
        return imgs[:, border_size:-border_size, border_size:-border_size]


def PSNR(pred, gt):
    pred = pred.clamp(0, 1)
    gt = gt.clamp(0, 1)
    # pred = (pred - pred.min()) / (pred.max() - pred.min())

    diff = pred - gt
    mse = np.mean(diff.numpy() ** 2)
    if mse == 0:
        return 100
    return 10 * log10(1.0 / mse)


def norm(img, vgg=False):
    if vgg:
        # normalize for pre-trained vgg model
        # https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        # normalize [-1, 1]
        transform = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
    return transform(img)


def norm2(imgs, vgg=False):
    if vgg:
        # normalize for pre-trained vgg model
        # https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        # normalize [-1, 1]
        transform = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
    size = list(imgs.shape)
    if len(size) == 4:
        norm_imgs = torch.FloatTensor(size[0], size[1], size[2], size[3])
        for i, img in enumerate(imgs):
            norm_imgs[i, :, :, :] = transform(img)
        return norm_imgs
    else:
        return transform(imgs)


def denorm(img, vgg=False):
    if vgg:
        transform = transforms.Normalize(mean=[-2.118, -2.036, -1.804],
                                         std=[4.367, 4.464, 4.444])
        return transform(img)
    else:
        out = (img + 1) / 2
        return out.clamp(0, 1)


def img_interp(imgs, scale_factor, interpolation='bicubic'):
    if interpolation == 'bicubic':
        interpolation = Image.BICUBIC
    elif interpolation == 'bilinear':
        interpolation = Image.BILINEAR
    elif interpolation == 'nearest':
        interpolation = Image.NEAREST

    size = list(imgs.shape)

    if len(size) == 4:
        target_height = int(size[2] * scale_factor)
        target_width = int(size[3] * scale_factor)
        interp_imgs = torch.FloatTensor(size[0], size[1], target_height, target_width)
        for i, img in enumerate(imgs):
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize((target_height, target_width),
                                                              interpolation=interpolation),
                                            transforms.ToTensor()])

            interp_imgs[i, :, :, :] = transform(img)
        return interp_imgs
    else:
        target_height = int(size[1] * scale_factor)
        target_width = int(size[2] * scale_factor)
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((target_height, target_width), interpolation=interpolation),
                                        transforms.ToTensor()])
        return transform(imgs)


####################
# add for esrgan
####################
import os
import mpmath
from datetime import datetime
import numpy as np
import cv2
from skimage.measure import compare_ssim
from torchvision.utils import make_grid
import json


####################
# miscellaneous
####################
def saveOpt(opt):
    dump_dir = opt.root_dir + opt.save_dir
    dump_path = os.path.join(dump_dir, 'options.json')
    if os.path.exists(dump_path):
        new_name = dump_path + '_archived_%s.json' % get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(dump_path, new_name)
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


####################
# image convert
####################
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


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


def add_noise(x, noise='.'):
    if noise is not '.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x


def gaussian_noise(img: np.ndarray, mean, std):
    imgtype = img.dtype
    gauss = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = np.clip((1 + gauss) * img.astype(np.float32), 0, 255)
    return noisy.astype(imgtype)


def poisson_noise(img):
    imgtype = img.dtype
    img = img.astype(np.float32) / 255.0
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = 255 * np.clip(np.random.poisson(img.astype(np.float32) * vals) / float(vals), 0, 1)
    return noisy.astype(imgtype)


def salt_and_pepper(img, prob=0.01):
    ''' Adds "Salt & Pepper" noise to an image.
        prob: probability (threshold) that controls level of noise
    '''
    imgtype = img.dtype
    rnd = np.random.rand(img.shape[0], img.shape[1])
    noisy = img.copy()
    noisy[rnd < prob / 2] = 0.0
    noisy[rnd > 1 - prob / 2] = 255.0
    return noisy.astype(imgtype)


####################
# metric
####################


def psnr(img1, img2):
    assert img1.dtype == img2.dtype == np.uint8, 'np.uint8 is supposed.'
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2, multichannel=False):
    assert img1.dtype == img2.dtype == np.uint8, 'np.uint8 is supposed.'
    return compare_ssim(img1, img2, multichannel=multichannel)


def compare_ergas(img1, img2, scale=4):
    return sewar.full_ref.ergas(img1, img2, r=scale, ws=8)

from skimage.measure import compare_mse
def compare_ergas1(img1, img2, scale=4):
    # rmse_map = np.zeros(img1.shape[2])
    # vals = np.zeros(img1.shape[2])
    channel = img1.shape[2]
    rmse_sum = 0
    for i in range(channel):
        rmse = compare_mse(img1[:, :, i], img2[:, :, i])
        mean2 = np.mean(img1[:, :, i])**2
        rmse_sum += rmse/mean2
    ergas = 100.0*np.sqrt(rmse_sum/channel)/scale
    return ergas

def compare_ergas2(img1, img2, scale=4):
    # rmse_map = np.zeros(img1.shape[2])
    # vals = np.zeros(img1.shape[2])
    channel = img1.shape[2]
    mse = compare_mse(img1, img2)
    # rmse_sum = np.mean(np.square(img1 - img2), dtype=np.float64)
    mean2 = np.mean(img1, dtype=np.float64)**2
    ergas = 100.0*np.sqrt(mse/mean2/channel)/scale
    return ergas

def compare_lpips(img1, img2, use_gpu=True):
    import PerceptualSimilarity
    model = PerceptualSimilarity.PerceptualLoss(model='net-lin', net='alex', use_gpu=use_gpu)
    d = model.forward(img1, img2)
    return d.detach().item()


def getFlopsAndParams(model, input_size):
    # from torchvision.models import resnet50
    from thop import profile
    from thop import clever_format
    # model = resnet50()
    input = torch.randn(input_size)
    total_ops, total_params = profile(model, inputs=(input,))
    total_ops, total_params = clever_format([total_ops, total_params], "%.3f")
    return [total_ops, total_params]


# helper printing function that can be used by subclasses
def get_network_description(network):
    if isinstance(network, torch.nn.DataParallel):
        network = network.module
    s = str(network)
    n = sum(map(lambda x: x.numel(), network.parameters()))
    return s, n


def print_network_to_file(model, save_dir, tag, input_size=(1, 3, 64, 64)):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    s, n = get_network_description(model)
    total_ops, total_params = getFlopsAndParams(model, input_size)
    print('Number of parameters in Model: {:,d}'.format(n))
    message = '-------------- ' + tag + ' --------------\n' + s + '\n'
    message = message + 'Total ops: ' + total_ops + '\n'
    message = message + 'Total params: ' + total_params + '\n'
    message = message + '----------------------------\n'
    network_path = os.path.join(save_dir, 'network.txt')
    with open(network_path, 'a+') as f:
        f.write(message)




####################
# GANLoss
####################
def gradient_penalty(self, discriminator, real_samples, fake_samples,
                     grad_penalty_Lp_norm='L2', penalty_type='LS'):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Best setting (novel Hinge Linfinity gradient penalty)
    # grad_penalty_Lp_norm = 'Linf'
    # penalty_type = 'hinge'
    # Default setting from WGAN-GP and most cases (L2 gradient penalty)
    # grad_penalty_Lp_norm = 'L2'
    # penalty_type = 'LS'
    # Calculate gradient
    # penalty_weight = 20 # 10 is the more usual choice

    # cuda = True if torch.cuda.is_available() else False
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
    # grad_outputs = torch.ones(param.batch_size)

    # Get gradient w.r.t. interpolates
    gradients = \
        torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=grad_outputs, create_graph=True,
                            retain_graph=True, only_inputs=True)[0]
    # grad = grad.view(current_batch_size,-1)

    if grad_penalty_Lp_norm == 'Linf':  # Linfinity gradient norm penalty (Corresponds to L1 margin, BEST results)
        grad_abs = torch.abs(gradients)  # Absolute value of gradient
        grad_norm, _ = torch.max(grad_abs, 1)
    elif grad_penalty_Lp_norm == 'L1':  # L1 gradient norm penalty (Corresponds to Linfinity margin, WORST results)
        grad_norm = gradients.norm(1, 1)
    else:  # L2 gradient norm penalty (Corresponds to L2 margin, this is what people generally use)
        grad_norm = gradients.norm(2, 1)

    if penalty_type == 'LS':  # The usual choice, penalize values below 1 and above 1 (too constraining to properly estimate the Wasserstein distance)
        constraint = (grad_norm - 1).pow(2)
    elif penalty_type == 'hinge':  # Penalize values above 1 only (best choice)
        constraint = torch.nn.ReLU()(grad_norm - 1)

    constraint = constraint.mean()
    gradient_penalty = constraint
    gradient_penalty.backward(retain_graph=True)

    return gradient_penalty


# '--penalty', type=float, default=20, help='Gradient penalty parameter for Gradien penalties')
# '--grad_penalty', type='bool', default=False, help='If True, use gradient penalty of WGAN-GP but with whichever loss_D chosen. No need to set this true with WGAN-GP.')
# '--no_grad_penalty', type='bool', default=False, help='If True, do not use gradient penalty when using WGAN-GP (If you want to try Spectral normalized WGAN).')
# '--grad_penalty_aug', type='bool', default=False, help='If True, use augmented lagrangian for gradient penalty (aka Sobolev GAN).')
# '--fake_only', type='bool', default=False, help='Using fake data only in gradient penalty')
# '--real_only', type='bool', default=False, help='Using real data only in gradient penalty')
# '--delta', type=float, default=1, help='(||grad_D(x)|| - delta)^2')
# '--rho', type=float, default=.0001, help='learning rate of lagrange multiplier when using augmented lagrangian')
# '--penalty-type', help='Gradient penalty type.  The default ("squared-diff") forces gradient norm *equal* to 1, which is not correct, but is what is done in the original WGAN paper. True Lipschitz constraint is with "clamp"',
# choices=['clamp','squared-clamp','squared-diff','squared','TV','abs','hinge','hinge2'], default='squared-diff')
# '--reduction', help='Summary statistic for gradient penalty over batches (default: "mean")',
# choices=['mean','max','softmax'],default='mean')
# '--l1_margin', help='maximize L-1 margin (equivalent to penalizing L-infinity gradient norm)',action='store_true')
# '--l1_margin_logsumexp', help='maximize L-1 margin using logsumexp to approximate L-infinity gradient norm (equivalent to penalizing L-infinity gradient norm)',action='store_true')
# '--l1_margin_smoothmax', help='maximize L-1 margin using smooth max to approximate L-infinity gradient norm (equivalent to penalizing L-infinity gradient norm)',action='store_true')
# '--linf_margin', help='maximize L-infinity margin (equivalent to penalizing L-1 gradient norm)',action='store_true')
# '--smoothmax', type=float, default=.5, help='parameter for smooth max (higher = less smooth)')
# '--l1_margin_no_abs', help='Only penalize positive gradient (Shouldnt work, but it does)',action='store_true')
# "--no_bias", type='bool', default=False, help="Unbiased estimator when using RaLSGAN (loss_D=12) or RcLSGAN (loss_D=22)")
class GANLoss(torch.nn.Module):
    def __init__(self, loss_D, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        # loss_D == 1:'GAN_'
        # loss_D == 2:'LSGAN_'
        # loss_D == 3:'HingeGAN_'
        # loss_D == 4:'WGANGP_'
        # loss_D == 11:'RaSGAN_'
        # loss_D == 12:'RaLSGAN_'
        # loss_D == 13:'RaHingeGAN_'
        # loss_D == 21:'RcSGAN_'
        # loss_D == 22:'RcLSGAN_'
        # loss_D == 23:'RcHingeGAN_'
        # loss_D == 31:'RpSGAN_'
        # loss_D == 32:'RpLSGAN_'
        # loss_D == 33:'RpHingeGAN_'
        # loss_D == 41:'RpSGAN_MVUE_'
        # loss_D == 42:'RpLSGAN_MVUE_'
        # loss_D == 43:'RpHingeGAN_MVUE_'

        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_D = loss_D

        self.no_bias = False
        self.penalty = 20
        self.grad_penalty = False
        self.no_grad_penalty = False
        self.grad_penalty_aug = False,
        self.fake_only = False
        self.real_only = False
        self.delta = 1
        self.rho = .0001
        self.penalty_type = 'squared-diff'
        self.reduction = 'mean'
        self.l1_margin = True
        self.l1_margin_logsumexp = True
        self.l1_margin_smoothmax = True
        self.linf_margin = True
        self.smoothmax = .5
        self.l1_margin_no_abs = True

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, x, x_fake, y_pred, y_pred_fake, discriminator=None):
        current_batch_size = y_pred.size()[0]
        self.batch = current_batch_size
        real_label = self.get_target_label(y_pred, True)
        fake_label = self.get_target_label(y_pred, False)
        BCE_stable = torch.nn.BCEWithLogitsLoss()
        # z = torch.FloatTensor(current_batch_size, z_size, 1, 1)
        # Uniform weight
        u = torch.FloatTensor(current_batch_size, 1, 1, 1)
        grad_outputs = torch.ones(current_batch_size)
        w_grad = torch.FloatTensor([self.penalty])

        if discriminator is not None:
            if self.loss_D in [1, 2, 3, 4]:
                # Train with real data
                if self.loss_D == 1:
                    errD_real = torch.nn.BCEWithLogitsLoss(y_pred, real_label)
                if self.loss_D == 2:
                    errD_real = torch.mean((y_pred - real_label) ** 2)
                    # a = torch.abs(y_pred - y)
                    # errD_real = torch.mean(a**(1+torch.log(1+a**4)))
                if self.loss_D == 4:
                    errD_real = -torch.mean(y_pred)
                if self.loss_D == 3:
                    errD_real = torch.mean(torch.nn.ReLU()(1.0 - y_pred))

                # Train with fake data
                # Detach y_pred from the neural network G and put it inside D
                if self.loss_D == 1:
                    errD_fake = torch.nn.BCEWithLogitsLoss(y_pred_fake, fake_label)
                if self.loss_D == 2:
                    errD_fake = torch.mean((y_pred_fake) ** 2)
                    # a = torch.abs(y_pred_fake - real_label)
                    # errD_fake = torch.mean(a**(1+torch.log(1+a**2)))
                if self.loss_D == 4:
                    errD_fake = torch.mean(y_pred_fake)
                if self.loss_D == 3:
                    errD_fake = torch.mean(torch.nn.ReLU()(1.0 + y_pred_fake))
                errD = errD_real + errD_fake
                # print(errD)
            else:

                # Relativistic average GANs
                if self.loss_D == 11:
                    errD = BCE_stable(y_pred - torch.mean(y_pred_fake), real_label) + BCE_stable(
                        y_pred_fake - torch.mean(y_pred), fake_label)
                if self.loss_D == 12:
                    if self.no_bias:
                        errD = torch.mean((y_pred - torch.mean(y_pred_fake) - real_label) ** 2) + torch.mean(
                            (y_pred_fake - torch.mean(y_pred) + real_label) ** 2) - (
                                           torch.var(y_pred, dim=0) + torch.var(y_pred_fake, dim=0)) / self.batch_size
                    else:
                        errD = torch.mean((y_pred - torch.mean(y_pred_fake) - real_label) ** 2) + torch.mean(
                            (y_pred_fake - torch.mean(y_pred) + real_label) ** 2)
                if self.loss_D == 13:
                    errD = torch.mean(torch.nn.ReLU()(1.0 - (y_pred - torch.mean(y_pred_fake)))) + torch.mean(
                        torch.nn.ReLU()(1.0 + (y_pred_fake - torch.mean(y_pred))))

                # Relativistic centered GANs
                if self.loss_D == 21:
                    full_mean = (torch.mean(y_pred) + torch.mean(y_pred_fake)) / 2
                    errD = BCE_stable(y_pred - full_mean, real_label) + BCE_stable(y_pred_fake - full_mean, fake_label)
                if self.loss_D == 22:
                    full_mean = (torch.mean(y_pred) + torch.mean(y_pred_fake)) / 2
                    if self.no_bias:
                        errD = torch.mean((y_pred - full_mean - real_label) ** 2) + torch.mean(
                            (y_pred_fake - full_mean + real_label) ** 2) + (
                                           torch.var(y_pred, dim=0) + torch.var(y_pred_fake, dim=0)) / (
                                           2 * self.batch_size)
                    else:
                        errD = torch.mean((y_pred - full_mean - real_label) ** 2) + torch.mean(
                            (y_pred_fake - full_mean + real_label) ** 2)
                if self.loss_D == 23:
                    full_mean = (torch.mean(y_pred) + torch.mean(y_pred_fake)) / 2
                    errD = torch.mean(torch.nn.ReLU()(1.0 - (y_pred - full_mean))) + torch.mean(
                        torch.nn.ReLU()(1.0 + (y_pred_fake - full_mean)))

                # Relativistic paired GANs (Without the MVUE)
                if self.loss_D == 31:
                    errD = 2 * (BCE_stable(y_pred - y_pred_fake, real_label))
                if self.loss_D == 32:
                    errD = 2 * torch.mean((y_pred - y_pred_fake - real_label) ** 2)
                if self.loss_D == 33:
                    errD = 2 * torch.mean(torch.nn.ReLU()(1.0 - (y_pred - y_pred_fake)))

                if self.loss_D in [41, 42, 43]:

                    # Relativistic paired GANs (MVUE, slower)
                    # Creating cartesian product substraction, very demanding sadly O(k^2), where k is the batch size
                    grid_x, grid_y = torch.meshgrid([y_pred, y_pred_fake])
                    y_pred_subst = (grid_x - grid_y)
                    real_label.resize_(current_batch_size, current_batch_size).fill_(1)

                    if self.loss_D == 41:
                        errD = 2 * (BCE_stable(y_pred_subst, real_label))
                    if self.loss_D == 42:
                        errD = 2 * torch.mean((y_pred_subst - real_label) ** 2)
                    if self.loss_D == 43:
                        errD = 2 * torch.mean(torch.nn.ReLU()(1.0 - y_pred_subst))

                errD_real = errD
                errD_fake = errD
            if (self.loss_D in [4] or self.grad_penalty) and (not self.no_grad_penalty):
                # Gradient penalty
                u.resize_(current_batch_size, 1, 1, 1)
                u.uniform_(0, 1)
                if self.real_only:
                    x_both = x.data
                elif self.fake_only:
                    x_both = x_fake.data
                else:
                    x_both = x.data * u + x_fake.data * (1 - u)
                if self.cuda:
                    x_both = x_both.cuda()

                # We only want the gradients with respect to x_both
                x_both = Variable(x_both, requires_grad=True)
                y0 = discriminator(x_both)
                grad = torch.autograd.grad(outputs=y0,
                                           inputs=x_both, grad_outputs=grad_outputs,
                                           retain_graph=True, create_graph=True,
                                           only_inputs=True)[0]
                x_both.requires_grad_(False)
                sh = grad.shape
                grad = grad.view(current_batch_size, -1)

                if self.l1_margin_no_abs:
                    grad_abs = torch.abs(grad)
                else:
                    grad_abs = grad

                if self.l1_margin:
                    grad_norm, _ = torch.max(grad_abs, 1)
                elif self.l1_margin_smoothmax:
                    grad_norm = torch.sum(grad_abs * torch.exp(self.smoothmax * grad_abs)) / torch.sum(
                        torch.exp(self.smoothmax * grad_abs))
                elif self.l1_margin_logsumexp:
                    grad_norm = torch.logsumexp(grad_abs, 1)
                elif self.linf_margin:
                    grad_norm = grad.norm(1, 1)
                else:
                    grad_norm = grad.norm(2, 1)

                if self.penalty_type == 'squared-diff':
                    constraint = (grad_norm - 1).pow(2)
                elif self.penalty_type == 'clamp':
                    constraint = grad_norm.clamp(min=1.) - 1.
                elif self.penalty_type == 'squared-clamp':
                    constraint = (grad_norm.clamp(min=1.) - 1.).pow(2)
                elif self.penalty_type == 'squared':
                    constraint = grad_norm.pow(2)
                elif self.penalty_type == 'TV':
                    constraint = grad_norm
                elif self.penalty_type == 'abs':
                    constraint = torch.abs(grad_norm - 1)
                elif self.penalty_type == 'hinge':
                    constraint = torch.nn.ReLU()(grad_norm - 1)
                elif self.penalty_type == 'hinge2':
                    constraint = (torch.nn.ReLU()(grad_norm - 1)).pow(2)
                else:
                    raise ValueError('penalty type %s is not valid' % self.penalty_type)

                if self.reduction == 'mean':
                    constraint = constraint.mean()
                elif self.reduction == 'max':
                    constraint = constraint.max()
                elif self.reduction == 'softmax':
                    sm = constraint.softmax(0)
                    constraint = (sm * constraint).sum()
                else:
                    raise ValueError('reduction type %s is not valid' % self.reduction)
                if self.print_grad:
                    print(constraint)

                if self.grad_penalty_aug:
                    grad_penalty = (-w_grad * constraint + (self.rho / 2) * (constraint) ** 2)
                else:
                    grad_penalty = self.penalty * constraint
            else:
                grad_penalty = 0.

            loss = errD + grad_penalty
        else:
            if self.loss_D == 1:
                errG = torch.nn.BCEWithLogitsLoss(y_pred_fake, real_label)
            if self.loss_D == 2:
                errG = torch.mean((y_pred_fake - real_label) ** 2)
            if self.loss_D == 4:
                errG = -torch.mean(y_pred_fake)
            if self.loss_D == 3:
                errG = -torch.mean(y_pred_fake)

            # Relativistic average GANs
            if self.loss_D == 11:
                errG = torch.nn.BCEWithLogitsLoss(y_pred - torch.mean(y_pred_fake), fake_label) + BCE_stable(
                    y_pred_fake - torch.mean(y_pred), real_label)
            if self.loss_D == 12:
                if self.no_bias:
                    errG = torch.mean((y_pred - torch.mean(y_pred_fake) + real_label) ** 2) + torch.mean(
                        (y_pred_fake - torch.mean(y_pred) - real_label) ** 2) - (
                                       torch.var(y_pred_fake, dim=0) / self.batch_size)
                else:
                    errG = torch.mean((y_pred - torch.mean(y_pred_fake) + real_label) ** 2) + torch.mean(
                        (y_pred_fake - torch.mean(y_pred) - real_label) ** 2)
            if self.loss_D == 13:
                errG = torch.mean(torch.nn.ReLU()(1.0 + (y_pred - torch.mean(y_pred_fake)))) + torch.mean(
                    torch.nn.ReLU()(1.0 - (y_pred_fake - torch.mean(y_pred))))

            if self.loss_D == 21:
                full_mean = (torch.mean(y_pred) + torch.mean(y_pred_fake)) / 2
                errG = torch.nn.BCEWithLogitsLoss(y_pred - full_mean, fake_label) + torch.nn.BCEWithLogitsLoss(
                    y_pred_fake - full_mean, real_label)
            if self.loss_D == 22:  # (y_hat-1)^2 + (y_hat+1)^2
                full_mean = (torch.mean(y_pred) + torch.mean(y_pred_fake)) / 2
                if self.no_bias:
                    errG = torch.mean((y_pred - full_mean + real_label) ** 2) + torch.mean(
                        (y_pred_fake - full_mean - real_label) ** 2) + (
                                       torch.var(y_pred_fake, dim=0) / (2 * self.batch_size))
                else:
                    errG = torch.mean((y_pred - full_mean + real_label) ** 2) + torch.mean(
                        (y_pred_fake - full_mean - real_label) ** 2)
            if self.loss_D == 23:
                full_mean = (torch.mean(y_pred) + torch.mean(y_pred_fake)) / 2
                errG = torch.mean(torch.nn.ReLU()(1.0 + (y_pred - full_mean))) + torch.mean(
                    torch.nn.ReLU()(1.0 - (y_pred_fake - full_mean)))

            # Relativistic paired GANs (Without the MVUE)
            if self.loss_D == 31:
                errG = torch.nn.BCEWithLogitsLoss(y_pred_fake - y_pred, real_label)
            if self.loss_D == 32:  # (y_hat-1)^2 + (y_hat+1)^2
                errG = torch.mean((y_pred_fake - y_pred - real_label) ** 2)
            if self.loss_D == 33:
                errG = torch.mean(torch.nn.ReLU()(1.0 - (y_pred_fake - y_pred)))

            if self.loss_D in [41, 42, 43]:

                # Relativistic paired GANs
                # Creating cartesian product substraction, very demanding sadly O(k^2), where k is the batch size
                grid_x, grid_y = torch.meshgrid([y_pred, y_pred_fake])
                y_pred_subst = grid_y - grid_x
                real_label.resize_(current_batch_size, current_batch_size).fill_(1)

                if self.loss_D == 41:
                    errG = 2 * torch.nn.BCEWithLogitsLoss(y_pred_subst, real_label)
                if self.loss_D == 42:  # (y_hat-1)^2 + (y_hat+1)^2
                    errG = 2 * torch.mean((y_pred_subst - real_label) ** 2)
                if self.loss_D == 43:
                    errG = 2 * torch.mean(torch.nn.ReLU()(1.0 - y_pred_subst))
            loss = errG

        return loss