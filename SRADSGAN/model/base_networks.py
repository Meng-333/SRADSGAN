import os
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim.optimizer import Optimizer
# from torch.autograd import Variable
# from torch import Tensor
from torch.nn import Parameter

############################################## Normalization Method ##############################################
# Group Normalization
class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias

# adaptive combination of InstanceNorm and LayerNorm
class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out

# combination of InstanceNorm and LayerNorm
class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        return out


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

# spectral normalization on parameters of convs
class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

#######################################################################################################

class DenseBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation='relu', norm='batch'):
        super(DenseBlock, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, dilation=1, bias=True, activation=None, norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias, dilation=dilation)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        elif self.norm == 'group':
            self.bn = GroupNorm(output_size)
        # else:
        #     raise NotImplementedError('Normalization type [{:s}] is not found'.format(self.norm))

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='relu', norm='batch'):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

# build block of ResNet
class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu', norm='batch'):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.norm1 = torch.nn.BatchNorm2d(num_filter)
            self.norm2 = torch.nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.norm1 = torch.nn.InstanceNorm2d(num_filter)
            self.norm2 = torch.nn.InstanceNorm2d(num_filter)
        elif norm == 'group':
            self.norm1 = GroupNorm(num_filter)
            self.norm2 = GroupNorm(num_filter)
        elif norm == 'iln':
            self.norm1 = ILN(num_filter)
            self.norm2 = ILN(num_filter)
        elif norm == 'adailn':
            self.norm1 = adaILN(num_filter)
            self.norm2 = adaILN(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()


    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.norm1(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.norm2(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        return out


class PSBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, scale_factor, kernel_size=3, stride=1, padding=1, bias=True, activation='relu', norm='batch'):
        super(PSBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size * scale_factor**2, kernel_size, stride, padding, bias=bias)
        self.ps = torch.nn.PixelShuffle(scale_factor)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.ps(self.conv(x)))
        else:
            out = self.ps(self.conv(x))

        if self.activation is not None:
            out = self.act(out)
        return out


class Upsample2xBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, upsample='deconv', activation='relu', norm='batch'):
        super(Upsample2xBlock, self).__init__()
        scale_factor = 2
        # 1. Deconvolution (Transposed convolution)
        if upsample == 'deconv':
            self.upsample = DeconvBlock(input_size, output_size,
                                        kernel_size=4, stride=2, padding=1,
                                        bias=bias, activation=activation, norm=norm)

        # 2. Sub-pixel convolution (Pixel shuffler)
        elif upsample == 'ps':
            self.upsample = PSBlock(input_size, output_size, scale_factor=scale_factor,
                                    bias=bias, activation=activation, norm=norm)

        # 3. Resize and Convolution
        elif upsample == 'rnc':
            self.upsample = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                ConvBlock(input_size, output_size,
                          kernel_size=3, stride=1, padding=1,
                          bias=bias, activation=activation, norm=norm)
            )

    def forward(self, x):
        out = self.upsample(x)
        return out

#######################################################################################################

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, pool_mode='Avg|Max'):
        super(ChannelAttention, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # if withmax:
        #     self.max_pool = nn.AdaptiveMaxPool2d(1)
        # else:
        #     self.max_pool = None
        self.pool_mode = pool_mode
        if pool_mode.find('Avg') != -1:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if pool_mode.find('Max') != -1:
            self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        # self.gamma = Parameter(torch.zeros(1))

    def forward(self, x):
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # if self.max_pool is not None:
        #     max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        #     out = avg_out + max_out
        # else:
        #     out = avg_out
        if self.pool_mode == 'Avg':
            out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        elif self.pool_mode == 'Max':
            out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        elif self.pool_mode == 'Avg|Max':
            avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
            out = avg_out + max_out
        out = self.sigmoid(out) * x
        return out

class WideChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(WideChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        expand = 2
        linear = 0.5
        self.fc   = nn.Conv2d(in_planes, in_planes * expand, 1, bias=False)
        self.fc1   = nn.Conv2d(in_planes * expand, int(in_planes * linear), 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(int(in_planes * linear), in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.fc(self.avg_pool(x)))))
        max_out = self.fc2(self.relu1(self.fc1(self.fc(self.max_pool(x)))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, pool_mode='Avg|Max'):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # self.withMaxPooling = withMax
        # if withMax:
        #     self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        # else:
        #     self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.pool_mode = pool_mode
        if pool_mode == 'Avg|Max':
            self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        else:
            self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # if self.withMaxPooling:
        #     max_out, _ = torch.max(x, dim=1, keepdim=True)
        #     out = torch.cat([avg_out, max_out], dim=1)
        # else:
        #     out = avg_out
        if self.pool_mode == 'Avg':
            out = torch.mean(x, dim=1, keepdim=True)
        elif self.pool_mode == 'Max':
            out, _ = torch.max(x, dim=1, keepdim=True)
        elif self.pool_mode == 'Avg|Max':
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out)) * x
        return out

class WideSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(WideSpatialAttention, self).__init__()

        expand = 2
        linear = 0.8
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 2*expand, kernel_size, padding=padding, bias=False)
        self.conv1 = nn.Conv2d(2*expand, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x = self.conv1(x)
        return self.sigmoid(x)

class PAM_Module(nn.Module):
    """ Position attention module"""
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim, light=False):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.light = light
        if light:
            self.conv1x1 = nn.Conv2d(in_dim * 2, in_dim, kernel_size=1, stride=1, bias=True)
            self.relu = nn.ReLU(True)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        if self.light:
            x_avg = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x_max = torch.nn.functional.adaptive_max_pool2d(x, 1)
            x_pool = self.relu(self.conv1x1(torch.cat([x_avg, x_max], 1)))
            proj_query = x_pool.view(m_batchsize, C, -1)
            proj_key = x_pool.view(m_batchsize, C, -1).permute(0, 2, 1)
            energy = torch.bmm(proj_query, proj_key)
            energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
            attention = self.softmax(energy_new)
            # out = attention*x
        else:
            proj_query = x.view(m_batchsize, C, -1)
            proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
            energy = torch.bmm(proj_query, proj_key)
            energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
            attention = self.softmax(energy_new)

        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out

class FLC_Channel_Attention(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(FLC_Channel_Attention, self).__init__()
        # Class Activation Map
        self.gap_fc = nn.Linear(in_dim, 1, bias=False)
        self.gmp_fc = nn.Linear(in_dim, 1, bias=False)
        self.conv1x1 = nn.Conv2d(in_dim * 2, in_dim, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)

        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        attention = self.relu(self.conv1x1(x))

        return attention

class GeometryPrior(torch.nn.Module):
    def __init__(self, k, channels, multiplier=0.5):
        super(GeometryPrior, self).__init__()
        self.channels = channels
        self.k = k
        self.position = 2 * torch.rand(1, 2, k, k, requires_grad=True) - 1
        self.l1 = torch.nn.Conv2d(2, int(multiplier * channels), 1)
        self.l2 = torch.nn.Conv2d(int(multiplier * channels), channels, 1)

    def forward(self, x):
        x = self.l2(torch.nn.functional.relu(self.l1(self.position)))
        return x.view(1, self.channels, 1, self.k ** 2)

class KeyQueryMap(torch.nn.Module):
    def __init__(self, channels, m):
        super(KeyQueryMap, self).__init__()
        self.l = torch.nn.Conv2d(channels, channels // m, 1)

    def forward(self, x):
        return self.l(x)

class AppearanceComposability(torch.nn.Module):
    def __init__(self, k, padding, stride):
        super(AppearanceComposability, self).__init__()
        self.k = k
        self.unfold = torch.nn.Unfold(k, 1, padding, stride)

    def forward(self, x):
        key_map, query_map = x
        k = self.k
        key_map_unfold = self.unfold(key_map)
        query_map_unfold = self.unfold(query_map)
        key_map_unfold = key_map_unfold.view(
            key_map.shape[0], key_map.shape[1],
            -1,
            key_map_unfold.shape[-2] // key_map.shape[1])
        query_map_unfold = query_map_unfold.view(
            query_map.shape[0], query_map.shape[1],
            -1,
            query_map_unfold.shape[-2] // query_map.shape[1])
        return key_map_unfold * query_map_unfold[:, :, :, k ** 2 // 2:k ** 2 // 2 + 1]

class LocalRelationalLayer(torch.nn.Module):
    def __init__(self, channels, k, stride=1, m=None, padding=0):
        super(LocalRelationalLayer, self).__init__()
        self.channels = channels
        self.k = k
        self.stride = stride
        self.m = m or 8
        self.padding = padding
        self.kmap = KeyQueryMap(channels, k)
        self.qmap = KeyQueryMap(channels, k)
        # self.kmap = torch.nn.Conv2d(channels, channels // k, 1)
        # self.qmap = torch.nn.Conv2d(channels, channels // k, 1)
        self.ac = AppearanceComposability(k, padding, stride)
        self.gp = GeometryPrior(k, channels // m)
        self.unfold = torch.nn.Unfold(k, 1, padding, stride)
        self.final1x1 = torch.nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        def combine_prior(appearance_kernel, geometry_kernel):
            return torch.nn.functional.softmax(appearance_kernel + geometry_kernel, dim=-1)
        gpk = self.gp(0)
        km = self.kmap(x)
        qm = self.qmap(x)
        ak = self.ac((km, qm))
        ck = combine_prior(ak, gpk)[:, None, :, :, :]
        x_unfold = self.unfold(x)
        x_unfold = x_unfold.view(x.shape[0], self.m, x.shape[1] // self.m, -1, x_unfold.shape[-2] // x.shape[1])
        pre_output = (ck * x_unfold).view(x.shape[0], x.shape[1], -1, x_unfold.shape[-2] // x.shape[1])
        h_out = (x.shape[2] + 2 * self.padding - 1 * (self.k - 1) - 1) // self.stride + 1
        w_out = (x.shape[3] + 2 * self.padding - 1 * (self.k - 1) - 1) // self.stride + 1
        pre_output = torch.sum(pre_output, dim=-1).view(x.shape[0], x.shape[1], h_out, w_out)
        return self.final1x1(pre_output)


# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
class AugmentedConv(nn.Module):
    # Example Code
    # tmp = torch.randn((16, 3, 32, 32)).to(device)
    # augmented_conv1 = AugmentedConv(in_channels=3, out_channels=20, kernel_size=3, dk=40, dv=4, Nh=4, relative=True, padding=1, stride=2, shape=16).to(device)
    # conv_out1 = augmented_conv1(tmp)
    # print(conv_out1.shape)
    #
    # for name, param in augmented_conv1.named_parameters():
    #     print('parameter name: ', name)
    #
    # augmented_conv2 = AugmentedConv(in_channels=3, out_channels=20, kernel_size=3, dk=40, dv=4, Nh=4, relative=True, padding=1, stride=1, shape=32).to(device)
    # conv_out2 = augmented_conv2(tmp)
    # print(conv_out2.shape)
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, shape=0, relative=False, stride=1):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size, stride=stride, padding=self.padding)

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=stride, padding=self.padding)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = torch.nn.functional.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


#######################################################################################################

class BasicBlock_WithoutAttention(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None):
        super(BasicBlock_WithoutAttention, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# WDSR Block type B
class WDSRBlock_BN(nn.Module):
    def __init__(
        self, n_feats, kernel_size, act=nn.ReLU(True), res_scale=1):
        super(WDSRBlock_BN, self).__init__()
        self.res_scale = res_scale
        expand = 6
        linear = 0.8

        body = [  nn.Conv2d(n_feats, n_feats*expand, 1, padding=1//2),
                  nn.BatchNorm2d(n_feats*expand),
                  act,
                  nn.Conv2d(n_feats * expand, int(n_feats * linear), 1, padding=1 // 2),
                  nn.BatchNorm2d(int(n_feats * linear)),
                  nn.Conv2d(int(n_feats * linear), n_feats, kernel_size, padding=kernel_size // 2),
                  nn.BatchNorm2d(n_feats)  ]

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res

# ESRGAN ResidualDenseBlock
class ResidualDenseBlock_5C(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''
    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, padding=1, \
            norm_type='batch', act_type='relu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = ConvBlock(nc, gc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=act_type, norm=norm_type)
        self.conv2 = ConvBlock(nc+gc, gc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=act_type, norm=norm_type)
        self.conv3 = ConvBlock(nc+2*gc, gc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=act_type, norm=norm_type)
        self.conv4 = ConvBlock(nc+3*gc, gc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=act_type, norm=norm_type)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = ConvBlock(nc+4*gc, gc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=last_act, norm=norm_type)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x

# ResidualDenseBlock with attention mechanism
class ResidualDenseBlock_5C_WithAttention(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''
    def __init__(self, nc, kernel_size=3, gc=32, stride=1, padding=1, bias=True, \
            norm_type=None, act_type='lrelu', mode='CNA', ca=False, sa=False):
        super(ResidualDenseBlock_5C_WithAttention, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = ConvBlock(nc, gc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=act_type, norm=norm_type)
        self.conv2 = ConvBlock(nc+gc, gc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=act_type, norm=norm_type)
        self.conv3 = ConvBlock(nc+2*gc, gc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=act_type, norm=norm_type)
        self.conv4 = ConvBlock(nc+3*gc, gc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=act_type, norm=norm_type)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = ConvBlock(nc+4*gc, nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=last_act, norm=norm_type)

        if ca:
            self.ca = ChannelAttention(nc)
        else:
            self.ca = None
        if sa:
            self.sa = SpatialAttention(kernel_size=kernel_size)
        else:
            self.sa = None
        if act_type == 'relu':
            self.act = torch.nn.ReLU(True)
        elif act_type == 'prelu':
            self.act = torch.nn.PReLU()
        elif act_type == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif act_type == 'tanh':
            self.act = torch.nn.Tanh()
        elif act_type == 'sigmoid':
            self.act = torch.nn.Sigmoid()
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        out = x5.mul(0.2)
        if self.ca is not None:
            out = self.ca(out) * out
        if self.sa is not None:
            out = self.sa(out) * out
        out += x

        if self.act is not None:
            out = self.act(out)
        # out = self.act(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, bias=True, dilation=1, norm_type='batch', act_type=None,
                 la_mode='CA-SA', pool_mode='Avg|Max', addconv=True, downsample=None):
        super(BasicBlock, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
        #              padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
        #              padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv1 = ConvBlock(inplanes, planes, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding, dilation=dilation, activation=act_type, norm=norm_type)
        self.conv2 = ConvBlock(planes, planes, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding, dilation=dilation, activation=None, norm=norm_type)

        # self.cascade = cascading
        # if not cascading:
        #     self.conv = nn.Conv2d(planes * 2, planes, kernel_size=1, bias=True)
        # if ca:
        #     self.ca = ChannelAttention(planes, withmax=withMaxPool)
        # else:
        #     self.ca = None
        # if sa:
        #     self.sa = SpatialAttention(kernel_size=7, withMax=withMaxPool)
        # else:
        #     self.sa = None
        self.la_mode = la_mode
        self.addconv = addconv
        if self.la_mode.find('CA') != -1:
            self.ca = ChannelAttention(planes, pool_mode=pool_mode)
        if self.la_mode.find('SA') != -1:
            self.sa = SpatialAttention(kernel_size=7, pool_mode=pool_mode)
        if self.la_mode.find('|') != -1:
            self.conv = nn.Conv2d(planes * 2, planes, kernel_size=1, bias=True)
        if self.la_mode.find('-') != -1 and addconv:
            self.conv = nn.Conv2d(planes, planes, kernel_size=1, bias=True)

        if act_type == 'relu':
            self.act = torch.nn.ReLU(True)
        elif act_type == 'prelu':
            self.act = torch.nn.PReLU()
        elif act_type == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif act_type == 'tanh':
            self.act = torch.nn.Tanh()
        elif act_type == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            self.act = None

        # downsample = None
        # if stride != 1 or inplanes != planes * self.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(inplanes, planes * self.expansion,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(inplanes * self.expansion),
        #     )
        # self.downsample = downsample

    def forward(self, x):
        # residual = x
        out = self.conv1(x)

        if self.inplanes == self.planes:
            residual = x
        else:
            residual = out

        out = self.conv2(out)

        # if self.cascade:
        #     if self.ca is not None:
        #         out = self.ca(out) * out
        #     if self.sa is not None:
        #         out = self.sa(out) * out
        # else:
        #     if self.ca is not None:
        #         caout = self.ca(out) * out
        #     if self.sa is not None:
        #         saout = self.sa(out) * out
        #     if self.ca is not None and self.sa is not None:
        #         out = self.conv(torch.cat([caout,saout], dim=1))

        if self.la_mode == 'CA':
            out = self.ca(out)
        elif self.la_mode == 'SA':
            out = self.sa(out)
        elif self.la_mode == 'CA-SA':
            out = self.ca(out)
            out = self.sa(out)
            if self.addconv:
                out = self.conv(out)
        elif self.la_mode == 'SA-CA':
            out = self.sa(out)
            out = self.ca(out)
            if self.addconv:
                out = self.conv(out)
        elif self.la_mode == 'CA|SA':
            saout = self.sa(out)
            caout = self.ca(out)
            out = self.conv(torch.cat([caout,saout], dim=1))

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        out += residual
        # out = self.act(out)
        if self.act is not None:
            out = self.act(out)
        return out

class BasicBlock_forDense(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, norm_type=None, act_type=None, la_mode='CA-SA', pool_mode='Avg|Max', downsample=None):
        super(BasicBlock_forDense, self).__init__()

        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
        #              padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
        #              padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv1 = ConvBlock(inplanes, planes, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding, activation=act_type, norm=norm_type, dilation=dilation)
        self.conv2 = ConvBlock(planes, planes, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding, activation=None, norm=norm_type,dilation=dilation)

        # self.cascade = cascading
        # if not cascading:
        #     self.conv = nn.Conv2d(planes*2, planes, kernel_size=1, bias=True)
        #
        # if ca:
        #     self.ca = ChannelAttention(planes)
        # else:
        #     self.ca = None
        # if sa:
        #     self.sa = SpatialAttention(kernel_size=7)
        # else:
        #     self.sa = None

        self.la_mode = la_mode
        if self.la_mode.find('CA') != -1:
            self.ca = ChannelAttention(planes, pool_mode=pool_mode)
            # self.ca = WideChannelAttention(planes)
        if self.la_mode.find('SA') != -1:
            self.sa = SpatialAttention(kernel_size=7, pool_mode=pool_mode)
            # self.sa = WideSpatialAttention(kernel_size=7)
        if self.la_mode.find('|') != -1:
            self.conv = nn.Conv2d(planes * 2, planes, kernel_size=1, bias=True)

        if act_type == 'relu':
            self.act = torch.nn.ReLU(True)
        elif act_type == 'prelu':
            self.act = torch.nn.PReLU()
        elif act_type == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif act_type == 'tanh':
            self.act = torch.nn.Tanh()
        elif act_type == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            self.act = None

        # downsample = None
        # if stride != 1 or inplanes != planes * self.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(inplanes, planes * self.expansion,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(inplanes * self.expansion),
        #     )
        # self.downsample = downsample

    def forward(self, x):
        # residual = x
        out = self.conv1(x)
        residual = out
        out = self.conv2(out)

        # if self.cascade:
        #     if self.ca is not None:
        #         out = self.ca(out) * out
        #     if self.sa is not None:
        #         out = self.sa(out) * out
        # else:
        #     if self.ca is not None:
        #         caout = self.ca(out) * out
        #     if self.sa is not None:
        #         saout = self.sa(out) * out
        #     out = self.conv(torch.cat([caout,saout], dim=1))

        if self.la_mode == 'CA':
            out = self.ca(out)
        elif self.la_mode == 'SA':
            out = self.sa(out)
        elif self.la_mode == 'CA-SA':
            out = self.ca(out)
            out = self.sa(out)
        elif self.la_mode == 'SA-CA':
            out = self.sa(out)
            out = self.ca(out)
        elif self.la_mode == 'CA|SA':
            saout = self.sa(out)
            caout = self.ca(out)
            out = self.conv(torch.cat([caout,saout], dim=1))

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        out += residual
        # out = self.act(out)
        if self.act is not None:
            out = self.act(out)
        return out

class WDBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, norm_type=None, act_type=None, ca=False, sa=False, downsample=None):
        super(WDBasicBlock, self).__init__()

        expand = 6
        linear = 0.8
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
        #              padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
        #              padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv1 = ConvBlock(inplanes, planes*expand, kernel_size=kernel_size, stride=stride, padding=padding, activation=act_type, norm=norm_type)
        self.conv2 = ConvBlock(planes*expand, int(planes*linear), kernel_size=kernel_size, stride=stride, padding=padding, activation=None, norm=norm_type)
        self.conv3 = ConvBlock(int(planes*linear), planes, kernel_size=kernel_size, stride=stride, padding=padding, activation=None, norm=norm_type)

        if ca:
            self.ca = ChannelAttention(planes)
        else:
            self.ca = None
        if sa:
            self.sa = SpatialAttention(kernel_size=7)
        else:
            self.sa = None

        if act_type == 'relu':
            self.act = torch.nn.ReLU(True)
        elif act_type == 'prelu':
            self.act = torch.nn.PReLU()
        elif act_type == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif act_type == 'tanh':
            self.act = torch.nn.Tanh()
        elif act_type == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            self.act = None

        # downsample = None
        # if stride != 1 or inplanes != planes * self.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(inplanes, planes * self.expansion,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(inplanes * self.expansion),
        #     )
        # self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.ca is not None:
            out = self.ca(out) * out
        if self.sa is not None:
            out = self.sa(out) * out

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        out += residual
        # out = self.act(out)
        if self.act is not None:
            out = self.act(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, norm_type=None, act_type='lrelu', ca=False, sa=False, downsample=None):
        super(Bottleneck, self).__init__()
        self.norm = norm_type
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d
        # self.conv1 = ConvBlock(inplanes, planes, kernel_size=1, stride=stride, padding=padding, activation=act_type, norm=norm_type)
        # self.conv2 = ConvBlock(planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, activation=act_type, norm=norm_type)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = self.bn(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = self.bn(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = self.bn(planes * 4)
        # self.relu = nn.ReLU(inplace=True)

        self.activation = act_type
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            self.act = None

        # self.ca = ChannelAttention(planes * 4)
        # self.sa = SpatialAttention()
        if ca:
            self.ca = ChannelAttention(planes*4)
        else:
            self.ca = None
        if sa:
            self.sa = SpatialAttention(kernel_size=3)
        else:
            self.sa = None

        # downsample = None
        # if stride != 1 or inplanes != planes * self.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(inplanes, planes * self.expansion,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(inplanes * self.expansion),
        #     )
        # self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.ca is not None:
            out = self.ca(out) * out
        if self.sa is not None:
            out = self.sa(out) * out

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out

class ResidualDenseBlock_5Block(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''
    def __init__(self, block, nc, gc=32, kernel_size=3, stride=1, bias=True, padding=1, \
            norm_type='batch', act_type='lrelu', mode='CNA', ca=False, sa=False):
        super(ResidualDenseBlock_5Block, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = block(nc, gc, kernel_size=kernel_size, stride=stride, padding=padding, norm_type=norm_type, act_type=act_type, ca=ca, sa=sa)
        self.conv2 = block(nc+gc, gc, kernel_size=kernel_size, stride=stride, padding=padding, norm_type=norm_type, act_type=act_type, ca=ca, sa=sa)
        self.conv3 = block(nc+2*gc, gc, kernel_size=kernel_size, stride=stride, padding=padding, norm_type=norm_type, act_type=act_type, ca=ca, sa=sa)
        self.conv4 = block(nc+3*gc, gc, kernel_size=kernel_size, stride=stride, padding=padding, norm_type=norm_type, act_type=act_type, ca=ca, sa=sa)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = block(nc+4*gc, nc, kernel_size=kernel_size, stride=stride, padding=padding, norm_type=norm_type, act_type=last_act, ca=ca, sa=sa)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x

class ResidualDenseBlock_5Block_WithAttention(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, block, n_blocks=1, nc=64, gc=32, kernel_size=3, stride=1, bias=True, padding=1,
                 norm_type=None, act_type='lrelu', mode='CNA', rla_mode='CA-SA', bla_mode='CA-SA', pool_mode='Avg|Max'):
        super(ResidualDenseBlock_5Block_WithAttention, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = block(nc, gc, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding, norm_type=norm_type,
                           act_type=act_type, la_mode=bla_mode, pool_mode=pool_mode)
        self.conv2 = block(nc + gc, gc, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding, norm_type=norm_type,
                           act_type=act_type, la_mode=bla_mode, pool_mode=pool_mode)
        self.conv3 = block(nc + 2 * gc, gc, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding,
                           norm_type=norm_type, act_type=act_type, la_mode=bla_mode, pool_mode=pool_mode)
        self.conv4 = block(nc + 3 * gc, gc, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding,
                           norm_type=norm_type, act_type=act_type, la_mode=bla_mode, pool_mode=pool_mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = block(nc + 4 * gc, nc, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding,
                           norm_type=norm_type, act_type=last_act, la_mode=bla_mode, pool_mode=pool_mode)

        # self.cascade = cascading
        # if not cascading:
        #     self.conv = nn.Conv2d(nc * 2, nc, kernel_size=1, bias=True)
        # if ca:
        #     self.ca = ChannelAttention(nc)
        # else:
        #     self.ca = None
        # if sa:
        #     self.sa = SpatialAttention(kernel_size=7)
        # else:
        #     self.sa = None

        self.la_mode = rla_mode
        if self.la_mode.find('CA') != -1:
            self.ca = ChannelAttention(nc, pool_mode=pool_mode)
            # self.ca = WideChannelAttention(nc)
        if self.la_mode.find('SA') != -1:
            self.sa = SpatialAttention(kernel_size=3, pool_mode=pool_mode)
            # self.sa = WideSpatialAttention(kernel_size=kernel_size)
        if self.la_mode.find('|') != -1:
            self.conv = nn.Conv2d(nc * 2, nc, kernel_size=1, bias=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        out = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # if self.cascade:
        #     if self.ca is not None:
        #         out = self.ca(out) * out
        #     if self.sa is not None:
        #         out = self.sa(out) * out
        # else:
        #     if self.ca is not None:
        #         caout = self.ca(out) * out
        #     if self.sa is not None:
        #         saout = self.sa(out) * out
        #     if self.ca is not None and self.sa is not None:
        #         out = self.conv(torch.cat([caout, saout], dim=1))
        if self.la_mode == 'CA':
            out = self.ca(out)
        elif self.la_mode == 'SA':
            out = self.sa(out)
        elif self.la_mode == 'CA-SA':
            out = self.ca(out)
            out = self.sa(out)
        elif self.la_mode == 'SA-CA':
            out = self.sa(out)
            out = self.ca(out)
        elif self.la_mode == 'CA|SA':
            saout = self.sa(out)
            caout = self.ca(out)
            out = self.conv(torch.cat([caout,saout], dim=1))
        return out.mul(0.2) + x

class ResidualDenseBlock_5DialtedBlock_WithAttention(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''
    def __init__(self, block, nc, gc=32, kernel_size=3, stride=1, bias=True, padding=1, \
            norm_type='batch', act_type='relu', mode='CNA', rla_mode='CA-SA', bla_mode='CA-SA', pool_mode='Avg|Max'):
        super(ResidualDenseBlock_5DialtedBlock_WithAttention, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = block(nc, gc, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding*1, dilation=1, norm_type=norm_type, act_type=act_type, la_mode=bla_mode, pool_mode=pool_mode)
        self.conv2 = block(nc+gc, gc, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding*4, dilation=4, norm_type=norm_type, act_type=act_type, la_mode=bla_mode, pool_mode=pool_mode)
        self.conv3 = block(nc+2*gc, gc, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding*8, dilation=8, norm_type=norm_type, act_type=act_type, la_mode=bla_mode, pool_mode=pool_mode)
        self.conv4 = block(nc+3*gc, gc, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding*12, dilation=12, norm_type=norm_type, act_type=act_type, la_mode=bla_mode, pool_mode=pool_mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = block(nc+4*gc, nc, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding*2, dilation=2, norm_type=norm_type, act_type=last_act, la_mode=bla_mode, pool_mode=pool_mode)

        # self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
        #                                      nn.Conv2d(nc, gc, 1, stride=1, bias=False),
        #                                      nn.BatchNorm2d(gc),
        #                                      nn.ReLU())
        # if ca:
        #     self.ca = ChannelAttention(nc)
        # else:
        #     self.ca = None
        # if sa:
        #     self.sa = SpatialAttention(kernel_size=kernel_size)
        # else:
        #     self.sa = None
        self.la_mode = rla_mode
        if self.la_mode.find('CA') != -1:
            self.ca = ChannelAttention(nc, pool_mode=pool_mode)
        if self.la_mode.find('SA') != -1:
            self.sa = SpatialAttention(kernel_size=kernel_size, pool_mode=pool_mode)
        if self.la_mode.find('|') != -1:
            self.conv = nn.Conv2d(nc * 2, nc, kernel_size=1, bias=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        # x5 = self.global_avg_pool(x)
        # x5 = F.bilinear(x5, x4.size()[2:], ) #.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        # x5 = nn.functional.upsample_bilinear(x5, x4.size()[2:])
        out = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # out = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # if self.ca is not None:
        #     out = self.ca(out) * out
        # if self.sa is not None:
        #     out = self.sa(out) * out
        if self.la_mode == 'CA':
            out = self.ca(out)
        elif self.la_mode == 'SA':
            out = self.sa(out)
        elif self.la_mode == 'CA-SA':
            out = self.ca(out)
            out = self.sa(out)
        elif self.la_mode == 'SA-CA':
            out = self.sa(out)
            out = self.ca(out)
        elif self.la_mode == 'CA|SA':
            saout = self.sa(out)
            caout = self.ca(out)
            out = self.conv(torch.cat([caout,saout], dim=1))
        return out.mul(0.2) + x

class ResidualBlock_Block_WithAttention(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''
    def __init__(self, block, n_blocks=1, nc=64, gc=32, kernel_size=3, stride=1, bias=True, padding=1,norm_type='batch',
                 act_type='relu', mode='CNA', rla_mode='CA-SA', bla_mode='CA-SA', pool_mode='Avg|Max', addconv=True):
        super(ResidualBlock_Block_WithAttention, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        # self.conv1 = block(nc, nc, kernel_size=kernel_size, stride=stride, padding=padding, norm_type=norm_type, act_type=act_type, la_mode=bla_mode, pool_mode=pool_mode)
        # self.conv2 = block(nc, nc, kernel_size=kernel_size, stride=stride, padding=padding, norm_type=norm_type, act_type=act_type, la_mode=bla_mode, pool_mode=pool_mode)
        # self.conv3 = block(nc, nc, kernel_size=kernel_size, stride=stride, padding=padding, norm_type=norm_type, act_type=act_type, la_mode=bla_mode, pool_mode=pool_mode)
        # self.conv4 = block(nc, nc, kernel_size=kernel_size, stride=stride, padding=padding, norm_type=norm_type, act_type=act_type, la_mode=bla_mode, pool_mode=pool_mode)
        blocks = []
        for _ in range(n_blocks-1):
            blocks.append(block(nc, nc, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding, norm_type=norm_type,
                      act_type=act_type, la_mode=bla_mode, pool_mode=pool_mode, addconv=addconv))
        self.blocks = nn.Sequential(*blocks)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.last_conv = block(nc, nc, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding, norm_type=norm_type,
                               act_type=last_act, la_mode=bla_mode, pool_mode=pool_mode, addconv=addconv)

        # self.cascade = cascading
        # if not cascading:
        #     self.conv = nn.Conv2d(nc * 2, nc, kernel_size=1, bias=True)
        # if ca:
        #     self.ca = ChannelAttention(nc, withmax=withMaxPool)
        # else:
        #     self.ca = None
        # if sa:
        #     self.sa = SpatialAttention(kernel_size=kernel_size, withMax=withMaxPool)
        # else:
        #     self.sa = None

        self.la_mode = rla_mode
        self.addconv = addconv
        if self.la_mode.find('CA') != -1:
            self.ca = ChannelAttention(nc, pool_mode=pool_mode)
        if self.la_mode.find('SA') != -1:
            self.sa = SpatialAttention(kernel_size=7, pool_mode=pool_mode)
        if self.la_mode.find('|') != -1:
            self.conv = nn.Conv2d(nc * 2, nc, kernel_size=1, bias=True)
        if self.la_mode.find('-') != -1 and addconv:
            self.conv = nn.Conv2d(nc, nc, kernel_size=1, bias=True)

    def forward(self, x):
        # x1 = self.conv1(x)
        # x2 = self.conv2(x1)
        # x3 = self.conv3(x2)
        # x4 = self.conv4(x3)
        x4 = self.blocks(x)
        out = self.last_conv(x4)

        # if self.cascade:
        #     if self.ca is not None:
        #         out = self.ca(out) * out
        #     if self.sa is not None:
        #         out = self.sa(out) * out
        # else:
        #     if self.ca is not None:
        #         caout = self.ca(out) * out
        #     if self.sa is not None:
        #         saout = self.sa(out) * out
        #     if self.ca is not None and self.sa is not None:
        #         out = self.conv(torch.cat([caout,saout], dim=1))

        if self.la_mode == 'CA':
            out = self.ca(out)
        elif self.la_mode == 'SA':
            out = self.sa(out)
        elif self.la_mode == 'CA-SA':
            out = self.ca(out)
            out = self.sa(out)
            if self.addconv:
                out = self.conv(out)
        elif self.la_mode == 'SA-CA':
            out = self.sa(out)
            out = self.ca(out)
            if self.addconv:
                out = self.conv(out)
        elif self.la_mode == 'CA|SA':
            saout = self.sa(out)
            caout = self.ca(out)
            out = self.conv(torch.cat([caout,saout], dim=1))
        out += x
        return out

class ResidualBlock_5Block_WithAttention(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, block, nc, kernel_size=3, stride=1, bias=True, padding=1, norm_type='batch', act_type='relu',
                 mode='CNA', rla_mode='CA-SA', bla_mode='CA-SA', pool_mode='Avg|Max'):
        super(ResidualBlock_5Block_WithAttention, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = block(nc, nc, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding, norm_type=norm_type,
                           act_type=act_type, la_mode=bla_mode, pool_mode=pool_mode)
        self.conv2 = block(nc, nc, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding, norm_type=norm_type,
                           act_type=act_type, la_mode=bla_mode, pool_mode=pool_mode)
        self.conv3 = block(nc, nc, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding, norm_type=norm_type,
                           act_type=act_type, la_mode=bla_mode, pool_mode=pool_mode)
        self.conv4 = block(nc, nc, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding, norm_type=norm_type,
                           act_type=act_type, la_mode=bla_mode, pool_mode=pool_mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = block(nc, nc, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding, norm_type=norm_type,
                           act_type=act_type, la_mode=bla_mode, pool_mode=pool_mode)

        # self.cascade = cascading
        # if not cascading:
        #     self.conv = nn.Conv2d(nc * 2, nc, kernel_size=1, bias=True)
        # if ca:
        #     self.ca = ChannelAttention(nc)
        # else:
        #     self.ca = None
        # if sa:
        #     self.sa = SpatialAttention(kernel_size=kernel_size)
        # else:
        #     self.sa = None

        self.la_mode = rla_mode
        if self.la_mode.find('CA') != -1:
            self.ca = ChannelAttention(nc, pool_mode=pool_mode)
        if self.la_mode.find('SA') != -1:
            self.sa = SpatialAttention(kernel_size=3, pool_mode=pool_mode)
        if self.la_mode.find('|') != -1:
            self.conv = nn.Conv2d(nc * 2, nc, kernel_size=1, bias=True)
        # if self.la_mode.find('-') != -1:
        #     self.conv = nn.Conv2d(nc, nc, kernel_size=1, bias=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = self.conv5(x4)

        # if self.cascade:
        #     if self.ca is not None:
        #         out = self.ca(out) * out
        #     if self.sa is not None:
        #         out = self.sa(out) * out
        # else:
        #     if self.ca is not None:
        #         caout = self.ca(out) * out
        #     if self.sa is not None:
        #         saout = self.sa(out) * out
        #     out = self.conv(torch.cat([caout, saout], dim=1))

        if self.la_mode == 'CA':
            out = self.ca(out)
        elif self.la_mode == 'SA':
            out = self.sa(out)
        elif self.la_mode == 'CA-SA':
            out = self.ca(out)
            out = self.sa(out)
            # out = self.conv(out)
        elif self.la_mode == 'SA-CA':
            out = self.sa(out)
            out = self.ca(out)
            out = self.conv(out)
        elif self.la_mode == 'CA|SA':
            saout = self.sa(out)
            caout = self.ca(out)
            out = self.conv(torch.cat([caout, saout], dim=1))

        out += x
        return out

# add by lyd
class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        # self.save_dir = opt['path']['models']  # save models
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_current_learning_rate(self):
        return self.optimizers[0].param_groups[0]['lr']

    # helper printing function that can be used by subclasses
    def get_network_description(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    # helper saving function that can be used by subclasses
    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path), strict=strict)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, norm_type='', use_spectralnorm=False, attention=False):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize, norm_type='batch', spectral_norm=False):
            """Returns layers of each discriminator block"""
            layers = []
            if spectral_norm:
                layers.append(SpectralNorm(nn.Conv2d(in_filters, out_filters, 3, stride, 1)))
            else:
                layers.append(nn.Conv2d(in_filters, out_filters, 3, stride, 1))

            if normalize:
                if norm_type =='batch':
                    layers.append(torch.nn.BatchNorm2d(out_filters))
                elif norm_type == 'instance':
                    layers.append(torch.nn.InstanceNorm2d(out_filters))
                elif norm_type == 'group':
                    layers.append(GroupNorm(out_filters))
                # layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        # for out_filters, stride, normalize in [ (64, 1, False),
        #                                         (64, 2, True),
        #                                         (128, 1, True),
        #                                         (128, 2, True),
        #                                         (256, 1, True),
        #                                         (256, 2, True),
        #                                         (512, 1, True),
        #                                         (512, 2, True),]:
        #     layers.extend(discriminator_block(in_filters, out_filters, stride, normalize, norm_type=norm_type, spectral_norm=use_spectralnorm))
        #     in_filters = out_filters
        for layer, out_filters, stride, normalize in [ (1, 64, 1, False),
                                                (2, 64, 2, True),
                                                (3, 128, 1, True),
                                                (4, 128, 2, True),
                                                (5, 256, 1, True),
                                                (6, 256, 2, True),
                                                (7, 512, 1, True),
                                                (8, 512, 2, True),]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize, norm_type=norm_type, spectral_norm=use_spectralnorm))
            if attention:
                if layer == 6:
                    layers.append(ChannelAttention(256))
                    layers.append(SpatialAttention())
                if layers == 8:
                    layers.append(CAM_Module(512))
                    layers.append(PAM_Module(512))
            in_filters = out_filters
        # Output layer
        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

import math
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, n_colors=3, n_resgroups=5, n_resblocks=10, n_feats=64, kernel_size=3, reduction=4, scale=4,
                 conv=default_conv, res_scale=1):
        super(RCAN, self).__init__()

        act = nn.ReLU(True)

        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)]

        # self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))