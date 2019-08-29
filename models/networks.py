import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init
from torch.optim import lr_scheduler

from .vgg import VGG19

###############################################################################
# Functions
###############################################################################


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
        net = torch.nn.DataParallel(net)
    init_weights(net, init_type)
    return net


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % layer_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def get_self_attention_layer(in_dim):
    self_attn_layer = SelfAttention(in_dim)
    return self_attn_layer


def define_G(input_nc, output_nc, nz, ngf, nencode=4, netG='unet_128', use_spectral_norm=False,
             norm='batch', nl='relu', use_dropout=False, use_attention=False,
             init_type='xavier', gpu_ids=[], where_add='input', upsample='bilinear'):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    if nz == 0:
        where_add = 'input'

    if netG == 'agisnet':
        input_content = input_nc
        input_style = input_nc * nencode
        net = AGISNet(input_content, input_style, output_nc, 6, ngf,
                      norm_layer=norm_layer,  nl_layer=nl_layer,
                      use_dropout=use_dropout, use_attention=use_attention,
                      use_spectral_norm=use_spectral_norm, upsample=upsample)

    elif netG == 'unet_64' and where_add == 'input':
        net = G_Unet_add_input(input_nc, output_nc, nz, 6, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                               use_dropout=use_dropout, use_attention=use_attention,
                               use_spectral_norm=use_spectral_norm, upsample=upsample)
    elif netG == 'unet_128' and where_add == 'input':
        net = G_Unet_add_input(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                               use_dropout=use_dropout, use_attention=use_attention,
                               use_spectral_norm=use_spectral_norm, upsample=upsample)
    elif netG == 'unet_256' and where_add == 'input':
        net = G_Unet_add_input(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                               use_dropout=use_dropout, use_attention=use_attention,
                               use_spectral_norm=use_spectral_norm, upsample=upsample)
    elif netG == 'unet_64' and where_add == 'all':
        net = G_Unet_add_all(input_nc, output_nc, nz, 6, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                             use_dropout=use_dropout, use_attention=use_attention,
                             use_spectral_norm=use_spectral_norm, upsample=upsample)
    elif netG == 'unet_128' and where_add == 'all':
        net = G_Unet_add_all(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                             use_dropout=use_dropout, use_attention=use_attention,
                             use_spectral_norm=use_spectral_norm, upsample=upsample)
    elif netG == 'unet_256' and where_add == 'all':
        net = G_Unet_add_all(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                             use_dropout=use_dropout, use_attention=use_attention,
                             use_spectral_norm=use_spectral_norm, upsample=upsample)
    else:
        raise NotImplementedError(
            'Generator model name [%s] is not recognized' % net)

    return init_net(net, init_type, gpu_ids)


def define_D(input_nc, ndf, netD,
             norm='batch', nl='lrelu', use_spectral_norm=False,
             use_sigmoid=False, init_type='xavier', num_Ds=1, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl = 'lrelu'  # use leaky relu for D
    nl_layer = get_non_linearity(layer_type=nl)

    if netD == 'basic_32':
        net = D_NLayers(input_nc, ndf, n_layers=2, norm_layer=norm_layer,
                        use_spectral_norm=use_spectral_norm, nl_layer=nl_layer, use_sigmoid=use_sigmoid)
    elif netD == 'basic_64':
        net = D_NLayers(input_nc, ndf, n_layers=2, norm_layer=norm_layer,
                        use_spectral_norm=use_spectral_norm, nl_layer=nl_layer, use_sigmoid=use_sigmoid)
    elif netD == 'basic_128':
        net = D_NLayers(input_nc, ndf, n_layers=2, norm_layer=norm_layer,
                        use_spectral_norm=use_spectral_norm, nl_layer=nl_layer, use_sigmoid=use_sigmoid)
    elif netD == 'basic_256':
        net = D_NLayers(input_nc, ndf, n_layers=3, norm_layer=norm_layer,
                        use_spectral_norm=use_spectral_norm, nl_layer=nl_layer, use_sigmoid=use_sigmoid)
    elif netD == 'basic_64_multi':
        net = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=1, norm_layer=norm_layer,
                             use_spectral_norm=use_spectral_norm, use_sigmoid=use_sigmoid, num_D=num_Ds)
    elif netD == 'basic_128_multi':
        net = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=2, norm_layer=norm_layer,
                             use_spectral_norm=use_spectral_norm, use_sigmoid=use_sigmoid, num_D=num_Ds)
    elif netD == 'basic_256_multi':
        net = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=3, norm_layer=norm_layer,
                             use_spectral_norm=use_spectral_norm, use_sigmoid=use_sigmoid, num_D=num_Ds)
    else:
        raise NotImplementedError(
            'Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, gpu_ids)


def define_R(input_nc, ndf, netR, norm='instance', nl='lrelu', use_spectral_norm=False,
             use_sigmoid=False, init_type='xavier', gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl = 'lrelu'
    nl_layer = get_non_linearity(layer_type=nl)

    if netR == 'basic_64':
        net = R_NLayers(input_nc, ndf, n_layers=4, norm_layer=norm_layer,
                        use_spectral_norm=use_spectral_norm, nl_layer=nl_layer, use_sigmoid=use_sigmoid)
    elif netR == 'basic_128':
        net = R_NLayers(input_nc, ndf, n_layers=5, norm_layer=norm_layer,
                        use_spectral_norm=use_spectral_norm, nl_layer=nl_layer, use_sigmoid=use_sigmoid)
    elif netR == 'basic_256':
        net = R_NLayers(input_nc, ndf, n_layers=6, norm_layer=norm_layer,
                        use_spectral_norm=use_spectral_norm, nl_layer=nl_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError(
            'Reviser model name [%s] is not recognized' % net)
    return init_net(net, init_type, gpu_ids)


def define_E(input_nc, output_nc, nef, netE,
             norm='batch', nl='lrelu',
             init_type='xavier', gpu_ids=[], vaeLike=False):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)

    if netE == 'resnet_64':
        net = E_ResNet(input_nc, output_nc, nef, n_blocks=3, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'resnet_128':
        net = E_ResNet(input_nc, output_nc, nef, n_blocks=4, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'resnet_256':
        net = E_ResNet(input_nc, output_nc, nef, n_blocks=5, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'conv_64':
        net = E_NLayers(input_nc, output_nc, nef, n_layers=3, norm_layer=norm_layer,
                        nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'conv_128':
        net = E_NLayers(input_nc, output_nc, nef, n_layers=4, norm_layer=norm_layer,
                        nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'conv_256':
        net = E_NLayers(input_nc, output_nc, nef, n_layers=5, norm_layer=norm_layer,
                        nl_layer=nl_layer, vaeLike=vaeLike)
    else:
        raise NotImplementedError(
            'Encoder model name [%s] is not recognized' % net)

    return init_net(net, init_type, gpu_ids)


class ListModule(object):
    # should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(
                self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class D_NLayersMulti(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_spectral_norm=False,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=1):
        super(D_NLayersMulti, self).__init__()
        # st()
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid, use_spectral_norm)
            self.model = nn.Sequential(*layers)
        else:
            self.model = ListModule(self, 'model')
            layers = self.get_layers(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid, use_spectral_norm)
            self.model.append(nn.Sequential(*layers))

            self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
            for i in range(num_D - 1):
                ndf_i = int(round(ndf / (2**(i + 1))))
                layers = self.get_layers(
                    input_nc, ndf_i, n_layers, norm_layer, use_sigmoid, use_spectral_norm)
                self.model.append(nn.Sequential(*layers))

    def get_layers(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                   use_sigmoid=False, use_spectral_norm=False):
        kw = 4
        padw = 1
        if use_spectral_norm:
            sequence = [SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw,
                                     stride=2, padding=padw)), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                                  stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if use_spectral_norm:
                sequence += [SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                          kernel_size=kw, stride=2, padding=padw))]
            else:
                sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                       kernel_size=kw, stride=2, padding=padw)]
            sequence += [
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        if use_spectral_norm:
            sequence += [SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                      kernel_size=kw, stride=1, padding=padw))]
        else:
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                   kernel_size=kw, stride=1, padding=padw)]
        sequence += [
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        if use_spectral_norm:
            sequence += [SpectralNorm(nn.Conv2d(ndf * nf_mult, 1,
                                      kernel_size=kw, stride=1, padding=padw))]
        else:
            sequence += [nn.Conv2d(ndf * nf_mult, 1,
                                   kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        return sequence

    def forward(self, input):
        if self.num_D == 1:
            return self.model(input)
        result = []
        down = input
        for i in range(self.num_D):
            result.append(self.model[i](down))
            if i != self.num_D - 1:
                down = self.down(down)
        return result


class G_NLayers(nn.Module):
    def __init__(self, output_nc=3, nz=100, ngf=64, n_layers=3,
                 norm_layer=None, nl_layer=None):
        super(G_NLayers, self).__init__()

        kw, s, padw = 4, 2, 1
        sequence = [nn.ConvTranspose2d(
            nz, ngf * 4, kernel_size=kw, stride=1, padding=0, bias=True)]
        if norm_layer is not None:
            sequence += [norm_layer(ngf * 4)]

        sequence += [nl_layer()]

        nf_mult = 4
        nf_mult_prev = 4
        for n in range(n_layers, 0, -1):
            nf_mult_prev = nf_mult
            nf_mult = min(n, 4)
            sequence += [nn.ConvTranspose2d(ngf * nf_mult_prev, ngf * nf_mult,
                                            kernel_size=kw, stride=s, padding=padw, bias=True)]
            if norm_layer is not None:
                sequence += [norm_layer(ngf * nf_mult)]
            sequence += [nl_layer()]

        sequence += [nn.ConvTranspose2d(ngf, output_nc,
                                        kernel_size=4, stride=s, padding=padw, bias=True)]
        sequence += [nn.Tanh()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


# Defines the conv discriminator with the specified arguments.
class D_NLayers(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_spectral_norm=False,
                 norm_layer=None, nl_layer=None, use_sigmoid=False):
        super(D_NLayers, self).__init__()

        kw, padw, use_bias = 4, 1, True
        # st()
        if use_spectral_norm:
            sequence = [SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw,
                                     stride=2, padding=padw, bias=use_bias))]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                                  stride=2, padding=padw, bias=use_bias)]
        sequence += [nl_layer()]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if use_spectral_norm:
                sequence += [SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                          kernel_size=kw, stride=2, padding=padw, bias=use_bias))]
            else:
                sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                       kernel_size=kw, stride=2, padding=padw, bias=use_bias)]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias)]

        if norm_layer is not None:
            sequence += [norm_layer(ndf * nf_mult)]
        sequence += [nl_layer()]

        if use_spectral_norm:
            sequence += [SpectralNorm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw,
                                      stride=1, padding=0, bias=use_bias))]
        else:
            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw,
                                   stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        output = self.model(input)
        return output


class R_NLayers(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_spectral_norm=False,
                 norm_layer=None, nl_layer=None, use_sigmoid=False):
        super(R_NLayers, self).__init__()

        kw, padw, use_bias = 3, 1, True
        # st()
        if use_spectral_norm:
            sequence = [SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw,
                                     stride=2, padding=padw, bias=use_bias))]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                                  stride=2, padding=padw, bias=use_bias)]
        sequence += [nl_layer()]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if use_spectral_norm:
                sequence += [SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                          kernel_size=kw, stride=2, padding=padw, bias=use_bias))]
            else:
                sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                       kernel_size=kw, stride=2, padding=padw, bias=use_bias)]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias)]
        if norm_layer is not None:
            sequence += [norm_layer(ndf * nf_mult)]
        sequence += [nl_layer()]
        if use_spectral_norm:
            sequence += [SpectralNorm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw,
                                      stride=1, padding=0, bias=use_bias))]
        else:
            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw,
                                   stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        output = self.model(input)
        return output


class E_NLayers(nn.Module):
    def __init__(self, input_nc, output_nc=1, nef=64, n_layers=3,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_NLayers, self).__init__()
        self.vaeLike = vaeLike

        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, nef, kernel_size=kw,
                              stride=2, padding=padw), nl_layer()]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 4)
            sequence += [
                nn.Conv2d(nef * nf_mult_prev, nef * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None:
                sequence += [norm_layer(nef * nf_mult)]
            sequence += [nl_layer()]
        sequence += [nn.AvgPool2d(8)]
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[nn.Linear(nef * nf_mult, output_nc)])
        if vaeLike:
            self.fcVar = nn.Sequential(*[nn.Linear(nef * nf_mult, output_nc)])

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output


class E_ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, nef=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_ResNet, self).__init__()
        self.vaeLike = vaeLike
        max_nef = 4
        conv_layers = [
            nn.Conv2d(input_nc, nef, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_nef = nef * min(max_nef, n)
            output_nef = nef * min(max_nef, n + 1)
            conv_layers += [BasicBlock(input_nef,
                                       output_nef, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(output_nef, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_nef, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_nef, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output
        return output


##############################################################################
# Classes
##############################################################################
class CXLoss(nn.Module):
    def __init__(self, sigma=0.1, b=1.0, similarity="consine"):
        super(CXLoss, self).__init__()
        self.similarity = similarity
        self.sigma = sigma
        self.b = b

    def center_by_T(self, featureI, featureT):
        # Calculate mean channel vector for feature map.
        meanT = featureT.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        # Normalize on channel dimension (axis=1)
        norms = features.norm(p=2, dim=1, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, features):
        N, C, H, W = features.shape
        assert N == 1
        P = H * W
        # NCHW --> 1x1xCXHW --> HWxCx1x1
        patches = features.view(1, 1, C, P).permute((3, 2, 0, 1))
        return patches

    def calc_relative_distances(self, raw_dist, axis=1):
        epsilon = 1e-5
        # [0] means get the value, torch min will return the index as well
        div = torch.min(raw_dist, dim=axis, keepdim=True)[0]
        relative_dist = raw_dist / (div + epsilon)
        return relative_dist

    def calc_CX(self, dist, axis=1):
        W = torch.exp((self.b - dist) / self.sigma)
        W_sum = W.sum(dim=axis, keepdim=True)
        return W.div(W_sum)

    def forward(self, featureT, featureI):
        '''
        :param featureT: target
        :param featureI: inference
        :return:
        '''

        # print("featureT target size:", featureT.shape)
        # print("featureI inference size:", featureI.shape)

        featureI, featureT = self.center_by_T(featureI, featureT)

        featureI = self.l2_normalize_channelwise(featureI)
        featureT = self.l2_normalize_channelwise(featureT)

        dist = []
        N = featureT.size()[0]
        for i in range(N):
            # NCHW
            featureT_i = featureT[i, :, :, :].unsqueeze(0)
            # NCHW
            featureI_i = featureI[i, :, :, :].unsqueeze(0)
            featureT_patch = self.patch_decomposition(featureT_i)
            # Calculate cosine similarity
            dist_i = F.conv2d(featureI_i, featureT_patch)
            dist.append(dist_i)

        # NCHW
        dist = torch.cat(dist, dim=0)

        raw_dist = (1. - dist) / 2.

        relative_dist = self.calc_relative_distances(raw_dist)

        CX = self.calc_CX(relative_dist)

        CX = CX.max(dim=3)[0].max(dim=2)[0]
        CX = CX.mean(1)
        CX_B = -torch.log(CX)
        CX = torch.mean(CX_B)
        return CX, CX_B


class RecLoss(nn.Module):
    def __init__(self, use_L2=True):
        super(RecLoss, self).__init__()
        self.use_L2 = use_L2

    def __call__(self, input, target, batch_mean=True):
        if self.use_L2:
            diff = (input - target) ** 2
        else:
            diff = torch.abs(input - target)
        if batch_mean:
            return torch.mean(diff)
        else:
            return torch.mean(torch.mean(torch.mean(diff, dim=1), dim=2), dim=3)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, mse_loss=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss() if mse_loss else nn.BCELoss

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, inputs, target_is_real):
        # if input is a list
        all_losses = []
        for input in inputs:
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss_input = self.loss(input, target_tensor)
            all_losses.append(loss_input)
        loss = sum(all_losses)
        return loss, all_losses


def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero', use_spectral_norm=False):
    # padding_type = 'zero'
    if upsample == 'basic':
        if use_spectral_norm:
            upconv = [SpectralNorm(nn.ConvTranspose2d(
                      inplanes, outplanes, kernel_size=4, stride=2, padding=1))]
        else:
            upconv = [nn.ConvTranspose2d(
                      inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1)]
        if use_spectral_norm:
            upconv += [SpectralNorm(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0))]
        else:
            upconv += [nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)


# two usage cases, depend on kw and padw
def upsampleConv(inplanes, outplanes, kw, padw):
    sequence = []
    sequence += [nn.Upsample(scale_factor=2, mode='nearest')]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=kw,
                           stride=1, padding=padw, bias=True)]
    return nn.Sequential(*sequence)


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


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
            v.data = l2normalize(
                torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
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


# Self Attention module from self-attention gan
class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=None):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        # print('attention size', x.size())
        m_batchsize, C, width, height = x.size()
        # print('query_conv size', self.query_conv(x).size())
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width * height)  # B X C X (W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B X (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        return out


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, use_spectral_norm=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, use_attention=False,
                 upsample='basic', padding_type='zero'):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=4, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc) if norm_layer is not None else None
        uprelu = nl_layer()
        upnorm = norm_layer(outer_nc) if norm_layer is not None else None
        if use_attention:
            attn_layer = get_self_attention_layer(outer_nc)

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type,
                use_spectral_norm=use_spectral_norm)

            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type,
                use_spectral_norm=use_spectral_norm)

            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            model = down + up
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type,
                use_spectral_norm=use_spectral_norm)

            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv

            if use_attention:
                up += [attn_layer]

            if upnorm is not None:
                up += [upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


class UnetBlock_with_z(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc, nz=0,
                 submodule=None, outermost=False, innermost=False, use_spectral_norm=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, use_attention=False,
                 upsample='basic', padding_type='zero'):
        super(UnetBlock_with_z, self).__init__()
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        input_nc = input_nc + nz
        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=4, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nl_layer()
        if use_attention:
            attn_layer = get_self_attention_layer(outer_nc)

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type,
                use_spectral_norm=use_spectral_norm)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
        elif innermost:
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type,
                use_spectral_norm=use_spectral_norm)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type,
                use_spectral_norm=use_spectral_norm)
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(inner_nc)]
            up = [uprelu] + upconv

            if use_attention:
                up += [attn_layer]

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]

            if use_dropout:
                up += [nn.Dropout(0.5)]
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3))
            x_and_z = torch.cat([x, z_img], 1)
        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)


class AGISNetBlock(nn.Module):
    def __init__(self, input_cont, input_style, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, use_spectral_norm=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, use_attention=False,
                 upsample='basic', padding_type='zero', wo_skip=False):
        super(AGISNetBlock, self).__init__()
        self.wo_skip = wo_skip
        p = 0
        downconv1 = []
        downconv2 = []
        if padding_type == 'reflect':
            downconv1 += [nn.ReflectionPad2d(1)]
            downconv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv1 += [nn.ReplicationPad2d(1)]
            downconv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        downconv1 += [nn.Conv2d(input_cont, inner_nc, kernel_size=3, stride=2, padding=p)]
        downconv2 += [nn.Conv2d(input_style, inner_nc, kernel_size=3, stride=2, padding=p)]

        # downsample is different from upsample
        downrelu1 = nn.LeakyReLU(0.2, True)
        downrelu2 = nn.LeakyReLU(0.2, True)
        uprelu = nl_layer()
        uprelu2 = nl_layer()

        attn_layer = None
        if use_attention:
            attn_layer = get_self_attention_layer(outer_nc)

        if outermost:
            if self.wo_skip:
                upconv = upsampleLayer(
                    inner_nc, inner_nc, upsample=upsample, padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm)
                upconv_B = upsampleLayer(
                    inner_nc, outer_nc, upsample=upsample, padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm)
            else:
                upconv = upsampleLayer(
                    inner_nc * 4, inner_nc, upsample=upsample, padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm)
                upconv_B = upsampleLayer(
                    inner_nc * 3, outer_nc, upsample=upsample, padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm)

            upconv_out = [nn.Conv2d(inner_nc + outer_nc, outer_nc, kernel_size=3, stride=1, padding=p)]

            down1 = downconv1
            down2 = downconv2
            up = [uprelu] + upconv
            up_B = [uprelu2] + upconv_B + [nn.Tanh()]

            uprelu3 = nl_layer()
            up_out = [uprelu3] + upconv_out + [nn.Tanh()]
            self.up_out = nn.Sequential(*up_out)

            if use_attention:
                up += [attn_layer]

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]

            if use_dropout:
                up += [nn.Dropout(0.5)]

        elif innermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type,
                use_spectral_norm=use_spectral_norm)
            upconv_B = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type,
                use_spectral_norm=use_spectral_norm)
            down1 = [downrelu1] + downconv1
            down2 = [downrelu2] + downconv2
            up = [uprelu] + upconv
            up_B = [uprelu2] + upconv_B
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
                up_B += [norm_layer(outer_nc)]
        else:
            if self.wo_skip:  # without skip-connection
                upconv = upsampleLayer(
                    inner_nc, outer_nc, upsample=upsample, padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm)
                upconv_B = upsampleLayer(
                    inner_nc, outer_nc, upsample=upsample, padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm)
            else:
                upconv = upsampleLayer(
                    inner_nc * 4, outer_nc, upsample=upsample, padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm)
                upconv_B = upsampleLayer(
                    inner_nc * 3, outer_nc, upsample=upsample, padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm)
            down1 = [downrelu1] + downconv1
            down2 = [downrelu2] + downconv2
            if norm_layer is not None:
                down1 += [norm_layer(inner_nc)]
                down2 += [norm_layer(inner_nc)]
            up = [uprelu] + upconv
            up_B = [uprelu2] + upconv_B

            if use_attention:
                up += [attn_layer]
                attn_layer2 = get_self_attention_layer(outer_nc)
                up_B += [attn_layer2]

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
                up_B += [norm_layer(outer_nc)]

            if use_dropout:
                up += [nn.Dropout(0.5)]
                up_B += [nn.Dropout(0.5)]

        self.down1 = nn.Sequential(*down1)
        self.down2 = nn.Sequential(*down2)
        self.submodule = submodule
        self.up = nn.Sequential(*up)
        self.up_B = nn.Sequential(*up_B)

    def forward(self, content, style):

        x1 = self.down1(content)
        x2 = self.down2(style)
        if self.outermost:
            mid_C, mid_B = self.submodule(x1, x2)
            fake_B = self.up_B(mid_B)
            mid_C2 = self.up(mid_C)
            fake_C = self.up_out(torch.cat([mid_C2, fake_B], 1))
            return fake_C, fake_B
        elif self.innermost:
            mid_C = torch.cat([x1, x2], 1)
            mid_B = torch.cat([x1, x2], 1)
            fake_C = self.up(mid_C)
            fake_B = self.up_B(mid_B)
            tmp1 = torch.cat([content, style], 1)
            if self.wo_skip:
                return fake_C, fake_B
            else:
                return torch.cat([torch.cat([fake_C, fake_B], 1), tmp1], 1), torch.cat([fake_B, tmp1], 1)
        else:
            mid, mid_B = self.submodule(x1, x2)
            fake_C = self.up(mid)
            fake_B = self.up_B(mid_B)
            tmp1 = torch.cat([content, style], 1)
            if self.wo_skip:
                return fake_C, fake_B
            else:
                return torch.cat([torch.cat([fake_C, fake_B], 1), tmp1], 1), torch.cat([fake_B, tmp1], 1)


class BasicBlockUp(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlockUp, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [upsampleConv(inplanes, outplanes, kw=3, padw=1)]
        if norm_layer is not None:
            layers += [norm_layer(outplanes)]
        layers += [conv3x3(outplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = upsampleConv(inplanes, outplanes, kw=1, padw=0)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class G_Unet_add_input(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False,
                 use_attention=False, use_spectral_norm=False, upsample='basic'):
        super(G_Unet_add_input, self).__init__()
        self.nz = nz
        max_nchn = 8  # max channel factor
        # construct unet structure
        unet_block = UnetBlock(ngf*max_nchn, ngf*max_nchn, ngf*max_nchn, use_spectral_norm=use_spectral_norm,
                               innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(ngf*max_nchn, ngf*max_nchn, ngf*max_nchn, unet_block,
                                   norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout,
                                   use_spectral_norm=use_spectral_norm, upsample=upsample)
        unet_block = UnetBlock(ngf*4, ngf*4, ngf*max_nchn, unet_block, use_attention=use_attention,
                               use_spectral_norm=use_spectral_norm, norm_layer=norm_layer,
                               nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf*2, ngf*2, ngf*4, unet_block, use_attention=use_attention,
                               use_spectral_norm=use_spectral_norm, norm_layer=norm_layer,
                               nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf, ngf, ngf*2, unet_block, use_attention=use_attention,
                               use_spectral_norm=use_spectral_norm, norm_layer=norm_layer,
                               nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(input_nc + nz, output_nc, ngf, unet_block,
                               use_spectral_norm=use_spectral_norm, outermost=True, norm_layer=norm_layer,
                               nl_layer=nl_layer, upsample=upsample)

        self.model = unet_block

    def forward(self, x, z=None):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3))
            x_with_z = torch.cat([x, z_img], 1)
        else:
            x_with_z = x  # no z

        return self.model(x_with_z)


# AGISNet Module
class AGISNet(nn.Module):

    def __init__(self, input_content, input_style, output_nc, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False,
                 use_attention=False, use_spectral_norm=False, upsample='basic', wo_skip=False):
        super(AGISNet, self).__init__()
        max_nchn = 8  # max channel factor
        # construct unet structure
        dual_block = AGISNetBlock(ngf*max_nchn, ngf*max_nchn, ngf*max_nchn, ngf*max_nchn,
                                  use_spectral_norm=use_spectral_norm, innermost=True,
                                  norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample, wo_skip=wo_skip)
        for i in range(num_downs - 5):
            dual_block = AGISNetBlock(ngf*max_nchn, ngf*max_nchn, ngf*max_nchn, ngf*max_nchn, dual_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout,
                                      use_spectral_norm=use_spectral_norm, upsample=upsample, wo_skip=wo_skip)
        dual_block = AGISNetBlock(ngf*4, ngf*4, ngf*4, ngf*max_nchn, dual_block, use_attention=use_attention,
                                  use_spectral_norm=use_spectral_norm, norm_layer=norm_layer,
                                  nl_layer=nl_layer, upsample=upsample, wo_skip=wo_skip)
        dual_block = AGISNetBlock(ngf*2, ngf*2, ngf*2, ngf*4, dual_block, use_attention=use_attention,
                                  use_spectral_norm=use_spectral_norm, norm_layer=norm_layer,
                                  nl_layer=nl_layer, upsample=upsample, wo_skip=wo_skip)
        dual_block = AGISNetBlock(ngf, ngf, ngf, ngf*2, dual_block, use_attention=use_attention,
                                  use_spectral_norm=use_spectral_norm, norm_layer=norm_layer,
                                  nl_layer=nl_layer, upsample=upsample, wo_skip=wo_skip)
        dual_block = AGISNetBlock(input_content, input_style, output_nc, ngf, dual_block,
                                  use_spectral_norm=use_spectral_norm, outermost=True, norm_layer=norm_layer,
                                  nl_layer=nl_layer, upsample=upsample, wo_skip=wo_skip)

        self.model = dual_block

    def forward(self, content, style):
        return self.model(content, style)


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class G_Unet_add_all(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False,
                 use_attention=False, use_spectral_norm=False, upsample='basic'):
        super(G_Unet_add_all, self).__init__()
        self.nz = nz
        # construct unet structure
        unet_block = UnetBlock_with_z(ngf*8, ngf*8, ngf*8, nz, None, innermost=True,
                                      use_spectral_norm=use_spectral_norm, norm_layer=norm_layer,
                                      nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf*8, ngf*8, ngf*8, nz, unet_block,
                                      use_spectral_norm=use_spectral_norm, norm_layer=norm_layer,
                                      nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        for i in range(num_downs - 6):
            unet_block = UnetBlock_with_z(ngf*8, ngf*8, ngf*8, nz, unet_block, use_spectral_norm=use_spectral_norm,
                                          norm_layer=norm_layer, nl_layer=nl_layer,
                                          use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf*4, ngf*4, ngf*8, nz, unet_block, use_attention=use_attention,
                                      use_spectral_norm=use_spectral_norm, norm_layer=norm_layer,
                                      nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf*2, ngf*2, ngf*4, nz, unet_block, use_attention=use_attention,
                                      use_spectral_norm=use_spectral_norm,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf, ngf, ngf*2, nz, unet_block, use_attention=use_attention,
                                      use_spectral_norm=use_spectral_norm,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(input_nc, output_nc, ngf, nz, unet_block, use_spectral_norm=use_spectral_norm,
                                      outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        self.model = unet_block

    def forward(self, x, z):
        return self.model(x, z)


class PatchLoss(nn.Module):
    def __init__(self, device, opt):
        super(PatchLoss, self).__init__()
        self.device = device
        self.vgg19 = VGG19().to(device)
        self.vgg19.load_model(opt.vgg)
        self.vgg19.eval()
        self.vgg_layer = 'conv2_2'
        self.loss = torch.nn.L1Loss()

    def l2_normalize_patch(self, features):
        # Normalize on patch dimension (axis=1,4,5)
        N, C, H, W, k1, k2 = features.shape
        features = features.permute(0, 1, 4, 5, 2, 3).contiguous().view(N, C*k1*k2, H, W)
        norms = torch.norm(features, p=2, dim=1, keepdim=True)
        features = features.div(norms)
        features = features.view(N, C, k1, k2, H, W).permute(0, 1, 4, 5, 2, 3)
        return features

    def forward(self, output, reference, shape_ref, color_ref):
        '''
        output: G output
        reference: output reference
        shape_ref: color reference
        color_ref: colors, style input
        '''

        patch_size = 3

        output_feat = self.vgg19(output)[self.vgg_layer]  # N * C * 8 * 8
        ref_feat = self.vgg19(reference)[self.vgg_layer]
        color_feat = self.vgg19(color_ref)[self.vgg_layer]  # N * C * 16 * 16
        shape_feat = self.vgg19(shape_ref)[self.vgg_layer]

        N, C, H, W = output_feat.shape
        unfolder1 = torch.nn.Unfold(kernel_size=(H-patch_size+1, W-patch_size+1))
        unfolder2 = torch.nn.Unfold(kernel_size=(H*2-patch_size+1, W*2-patch_size+1))
        output_pat = unfolder1(output_feat)  # N * (C*H-1*W-1) * (2*2)
        output_pat = output_pat.view((N, C, H-patch_size+1, W-patch_size+1, patch_size, patch_size))

        shape_pat = unfolder2(shape_feat)
        shape_pat = shape_pat.view((N, C, (H*2-patch_size+1)*(W*2-patch_size+1), patch_size, patch_size))
        shape_pat = torch.transpose(shape_pat, 1, 2)

        color_pat = unfolder2(color_feat)
        color_pat = color_pat.view((N, C, (H*2-patch_size+1)*(W*2-patch_size+1), patch_size, patch_size))

        dist = list()
        for i in range(N):
            ref_i = ref_feat[i].view(1, ref_feat[i].shape[0], ref_feat[i].shape[1], ref_feat[i].shape[2])
            shape_i = shape_pat[i]

            conv1 = nn.Conv2d(C, (H*2-patch_size+1)*(W*2-patch_size+1), kernel_size=2, stride=1, bias=False)
            conv1.weight = torch.nn.Parameter(shape_i)
            net = nn.Sequential(conv1)
            similarity = net(ref_i)
            argmax = torch.argmax(similarity, 1)

            matched = torch.zeros(output_pat[i].shape).to(self.device)
            for k in range(H-patch_size+1):
                for j in range(W-patch_size+1):
                    row = k
                    col = j
                    ind = argmax[0, row, col]
                    matched[:, row, col, ...] = color_pat[i, :, ind, ...]

            matched = torch.unsqueeze(matched, 0)
            dist.append(matched)
        dist = torch.cat(dist, dim=0)

        # output_pat = self.l2_normalize_patch(output_pat)
        # dist = self.l2_normalize_patch(dist)

        diff = (output_pat-dist).abs()
        l1_patch = diff.sum(dim=1).sum(dim=-2).sum(dim=-1)
        l1_clip = torch.clamp(l1_patch, 0, 1000)
        loss = l1_clip.sum()
        return loss


class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()
        # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


# base style loss
class Base_StyleLoss(nn.Module):
    def __init__(self):
        super(Base_StyleLoss, self).__init__()
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def __call__(self, input, target):
        input_gram = self.gram(input)
        target_gram = self.gram(target)
        loss = self.criterion(input_gram, target_gram)
        return loss


# define the style loss
class StyleLoss(nn.Module):
    def __init__(self, device, opt):
        super(StyleLoss, self).__init__()
        self.device = device
        self.vgg19 = VGG19().to(device)
        self.vgg19.load_model(opt.vgg_font)
        self.vgg19.eval()
        self.vgg_layers = ['conv2_2', 'conv3_2']
        self.criterion = Base_StyleLoss()

    def __call__(self, input, target):
        loss = 0.0
        for layer in self.vgg_layers:
            inp_feat = self.vgg19(input)[layer]
            tar_feat = self.vgg19(target)[layer]
            loss += self.criterion(inp_feat, tar_feat)
        return loss
