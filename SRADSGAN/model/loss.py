import torch
import torch.nn as nn
from torch.autograd import Variable
from model.util import shave_a2b, resize_tensor_w_kernel, create_penalty_mask, map2tensor


# noinspection PyUnresolvedReferences
class GANLoss(nn.Module):
    """D outputs a [0,1] map of size of the input. This map is compared in a pixel-wise manner to 1/0 according to
    whether the input is real (i.e. from the input image) or fake (i.e. from the Generator)"""

    def __init__(self, d_last_layer_size):
        super(GANLoss, self).__init__()
        # The loss function is applied after the pixel-wise comparison to the true label (0/1)
        self.loss = nn.L1Loss(reduction='mean')
        # Make a shape
        d_last_layer_shape = [1, 1, d_last_layer_size, d_last_layer_size]
        # The two possible label maps are pre-prepared
        self.label_tensor_fake = Variable(torch.zeros(d_last_layer_shape), requires_grad=False)
        self.label_tensor_real = Variable(torch.ones(d_last_layer_shape), requires_grad=False)

    def forward(self, d_last_layer, is_d_input_real):
        # Determine label map according to whether current input to discriminator is real or fake
        label_tensor = self.label_tensor_real if is_d_input_real else self.label_tensor_fake
        # Compute the loss
        return self.loss(d_last_layer, label_tensor)


class DownScaleLoss(nn.Module):
    """ Computes the difference between the Generator's downscaling and an ideal (bicubic) downscaling"""

    def __init__(self, scale_factor, cuda=True):
        super(DownScaleLoss, self).__init__()
        self.loss = nn.MSELoss()
        bicubic_k = [[0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [-.0013275146484375, -0.0039825439453130, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0013275146484380, -0.0039825439453125, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625]]
        if cuda:
            self.bicubic_kernel = Variable(torch.Tensor(bicubic_k).cuda(), requires_grad=False)
        else:
            self.bicubic_kernel = Variable(torch.Tensor(bicubic_k), requires_grad=False)
        self.scale_factor = scale_factor

    def forward(self, g_input, g_output, kernel=None):
        if kernel is None:
            downscaled = resize_tensor_w_kernel(im_t=g_output, k=self.bicubic_kernel, sf=self.scale_factor)
        else:
            downscaled = resize_tensor_w_kernel(im_t=g_output, k=kernel, sf=self.scale_factor)
        # Shave the downscaled to fit g_output
        return self.loss(g_input, shave_a2b(downscaled, g_input))


class SumOfWeightsLoss(nn.Module):
    """ Encourages the kernel G is imitating to sum to 1 """

    def __init__(self):
        super(SumOfWeightsLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.ones(1).to(kernel.device), torch.sum(kernel))


class CentralizedLoss(nn.Module):
    """ Penalizes distance of center of mass from K's center"""

    def __init__(self, k_size, scale_factor=4, cuda=True):
        super(CentralizedLoss, self).__init__()
        self.indices = Variable(torch.arange(0., float(k_size)), requires_grad=False)
        wanted_center_of_mass = k_size // 2 + 0.5 * (int(scale_factor) - k_size % 2)
        self.center = Variable(torch.FloatTensor([wanted_center_of_mass, wanted_center_of_mass]), requires_grad=False)
        if cuda:
            self.indices = self.indices.cuda()
            self.center = self.center.cuda()
        self.loss = nn.MSELoss()

    def forward(self, kernel):
        """Return the loss over the distance of center of mass from kernel center """
        r_sum, c_sum = torch.sum(kernel, dim=1).reshape(1, -1), torch.sum(kernel, dim=0).reshape(1, -1)
        return self.loss(torch.stack((torch.matmul(r_sum, self.indices) / torch.sum(kernel),
                                      torch.matmul(c_sum, self.indices) / torch.sum(kernel))), self.center)


class BoundariesLoss(nn.Module):
    """ Encourages sparsity of the boundaries by penalizing non-zeros far from the center """

    def __init__(self, k_size, cuda=True):
        super(BoundariesLoss, self).__init__()
        self.mask = map2tensor(create_penalty_mask(k_size, 30))
        self.zero_label = Variable(torch.zeros(k_size), requires_grad=False)
        if cuda:
            self.mask = self.mask.cuda()
            self.zero_label = self.zero_label.cuda()
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(kernel * self.mask, self.zero_label)


class SparsityLoss(nn.Module):
    """ Penalizes small values to encourage sparsity """
    def __init__(self):
        super(SparsityLoss, self).__init__()
        self.power = 0.2
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.abs(kernel) ** self.power, torch.zeros_like(kernel))

import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import models

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        if len(X.size()) == 4:
            h_relu1 = self.slice1(X)
            h_relu2 = self.slice2(h_relu1)
            h_relu3 = self.slice3(h_relu2)
            h_relu4 = self.slice4(h_relu3)
            h_relu5 = self.slice5(h_relu4)
        else:
            B, N, C, H, W = X.size()
            h_relu1, h_relu2, h_relu3, h_relu4, h_relu5 = [], [], [], [], []
            for j in range(N):
                h_relu1.append(self.slice1(X[:, j, :, :, :].clone().squeeze_(1).detach()))
                h_relu2.append(self.slice2(h_relu1[j]))
                h_relu3.append(self.slice3(h_relu2[j]))
                h_relu4.append(self.slice4(h_relu3[j]))
                h_relu5.append(self.slice5(h_relu4[j]))
            for j in range(N):
                h_relu1[j] = h_relu1[j].unsqueeze_(1)
                h_relu2[j] = h_relu2[j].unsqueeze_(1)
                h_relu3[j] = h_relu3[j].unsqueeze_(1)
                h_relu4[j] = h_relu4[j].unsqueeze_(1)
                h_relu5[j] = h_relu5[j].unsqueeze_(1)
            h_relu1 = torch.cat(h_relu1, 1)
            h_relu2 = torch.cat(h_relu2, 1)
            h_relu3 = torch.cat(h_relu3, 1)
            h_relu4 = torch.cat(h_relu4, 1)
            h_relu5 = torch.cat(h_relu5, 1)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class ContrastLoss(nn.Module):
    def __init__(self, ablation=False, cuda=True):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19()
        if cuda:
            self.vgg = self.vgg.cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss

class NContrastLoss(nn.Module):
    def __init__(self, ablation=False, cuda=True):

        super(NContrastLoss, self).__init__()
        self.vgg = Vgg19()
        if cuda:
            self.vgg = self.vgg.cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):
        B, N, C, H, W = n.size()
        a_vgg, p_vgg = self.vgg(a), self.vgg(p)
        n_vgg = self.vgg(n)
        # n_vgg = self.vgg(n.view(-1, C, H, W))
        # a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0
        # d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())

            # bn, c, h, w = n_vgg[i].size()
            # n_vgg_i = n_vgg[i].detach().view(B, N, c, h, w)
            n_vgg_i = n_vgg[i].detach()
            d_an = 0
            for j in range(N):
                d_an += self.l1(a_vgg[i], n_vgg_i[:, j, :, :, :].clone().squeeze_(1).detach())

            contrastive = d_ap / (d_an + 1e-7)

            loss += self.weights[i] * contrastive
        return loss

class ContrastCosineLoss(nn.Module):
    def __init__(self, ablation=False, cuda=True):

        super(ContrastCosineLoss, self).__init__()
        self.vgg = Vgg19()
        if cuda:
            self.vgg = self.vgg.cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation
        self.T = 0.07

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            # d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            s_ap = F.cosine_similarity(a_vgg[i], p_vgg[i].detach(), dim=1)
            d_ap = (torch.mean(s_ap, dim=[1, 2])/self.T).exp()
            if not self.ab:
                # d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                s_an = F.cosine_similarity(a_vgg[i], n_vgg[i].detach(), dim=1)
                d_an = (torch.mean(s_an, dim=[1, 2])/self.T).exp()
                contrastive = -torch.log(d_ap / (d_an + 1e-7))
            else:
                contrastive = -torch.log(d_ap)
            loss += self.weights[i] * contrastive
        return torch.mean(loss)

class NContrastCosineLoss(nn.Module):
    def __init__(self, ablation=False, cuda=True):

        super(NContrastCosineLoss, self).__init__()
        self.vgg = Vgg19()
        if cuda:
            self.vgg = self.vgg.cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation
        self.T = 0.07

    def forward(self, a, p, n):
        B, N, C, H, W = n.size()
        a_vgg, p_vgg = self.vgg(a), self.vgg(p)
        n_vgg = self.vgg(n)
        loss = 0
        # d_ap, d_an = 0, 0
        for i in range(len(self.weights)):
            s_ap = F.cosine_similarity(a_vgg[i], p_vgg[i].detach(), dim=1)
            d_ap = (torch.mean(s_ap, dim=[1, 2])/self.T).exp()

            # bn, c, h, w = n_vgg[i].size()
            # n_vgg_i = n_vgg[i].detach().view(B, N, c, h, w)
            n_vgg_i = n_vgg[i].detach()
            d_an = 0
            for j in range(N):
                s_an = F.cosine_similarity(a_vgg[i], n_vgg_i[:, j, :, :, :].clone().squeeze_(1).detach(), dim=1)
                d_an += torch.exp(torch.mean(s_an, dim=[1, 2])/self.T)
            contrastive = -torch.log(d_ap / (d_an + d_ap))

            loss = loss + self.weights[i] * contrastive
        loss = torch.mean(loss)
        return loss