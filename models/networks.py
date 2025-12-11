# SEP 8 (refined) — networks.py
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler
from util.util import noisy


###############################################################################
# Helper Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x


def conv(dimension):
    if dimension == 2:
        return nn.Conv2d
    elif dimension == 3:
        return nn.Conv3d
    raise ValueError('Invalid image dimension (expected 2 or 3).')


def maxpool(dimension):
    if dimension == 2:
        return nn.MaxPool2d
    elif dimension == 3:
        return nn.MaxPool3d
    raise ValueError('Invalid image dimension (expected 2 or 3).')


def convtranspose(dimension):
    if dimension == 2:
        return nn.ConvTranspose2d
    elif dimension == 3:
        return nn.ConvTranspose3d
    raise ValueError('Invalid image dimension (expected 2 or 3).')


def batch_norm(dimension):
    if dimension == 2:
        return nn.BatchNorm2d
    elif dimension == 3:
        return nn.BatchNorm3d
    raise ValueError('Invalid image dimension (expected 2 or 3).')


def instance_norm(dimension):
    if dimension == 2:
        return nn.InstanceNorm2d
    elif dimension == 3:
        return nn.InstanceNorm3d
    raise ValueError('Invalid image dimension (expected 2 or 3).')


def get_norm_layer(norm_type='instance', dimension=3):
    """Return a normalization layer factory."""
    if norm_type == 'batch':
        return functools.partial(batch_norm(dimension), affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        return functools.partial(instance_norm(dimension), affine=False, track_running_stats=False)
    elif norm_type in ('spectral', 'none'):
        # returns a callable that ignores channel arg and yields Identity()
        return lambda *_args, **_kwargs: Identity()
    else:
        raise NotImplementedError(f'Normalization layer [{norm_type}] is not found')


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler given options in `opt`."""
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    if opt.lr_policy == 'constant':
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    if opt.lr_policy == 'step':
        return lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)

    if opt.lr_policy == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)

    if opt.lr_policy == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)

    raise NotImplementedError(f'learning rate policy [{opt.lr_policy}] is not implemented')


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights."""
    def init_func(m):
        classname = m.__class__.__name__
        has_w = hasattr(m, 'weight') and (('Conv' in classname) or ('Linear' in classname))
        if has_w:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'init method [{init_type}] not implemented')
            if getattr(m, 'bias', None) is not None:
                init.constant_(m.bias.data, 0.0)
        # Cover both 2D/3D BatchNorm variants
        elif 'BatchNorm' in classname:
            if getattr(m, 'weight', None) is not None:
                init.normal_(m.weight.data, 1.0, init_gain)
            if getattr(m, 'bias', None) is not None:
                init.constant_(m.bias.data, 0.0)

    print(f'initialize network with {init_type}')
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=None):
    gpu_ids = gpu_ids or []
    device = torch.device(f'cuda:{gpu_ids[0]}') if gpu_ids else torch.device('cpu')
    net.to(device)
    if len(gpu_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


###############################################################################
# Factories
###############################################################################

def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, gpu_ids=None, kernel_size=9, given_psf=None, noise_setting=None,
             dimension=3):
    """Create a generator."""
    gpu_ids = gpu_ids or []
    net = None
    norm_layer = get_norm_layer(norm_type=norm, dimension=dimension)

    if netG == 'unet_twoouts':
        net = UnetTwoOuts(input_nc, output_nc)
    elif netG == 'unet_deconv':
        net = Unet_deconv(input_nc, output_nc, norm_layer=norm_layer, dimension=dimension)
    elif netG == 'unet_deconv_aniso':
        net = Unet_deconv_aniso(input_nc, output_nc, norm_layer=norm_layer)  # fixed 3D anisotropic path
    elif netG == 'unet_vanilla':
        net = Unet_vanilla(input_nc, output_nc, norm_layer=norm_layer, dimension=dimension)
    elif netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                              use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                              use_dropout=use_dropout, n_blocks=6)
    elif netG == 'VGG':
        net = VGG_net(input_nc, num_classes=2, VGG_type='VGG16')
    elif netG == 'linearkernel':
        net = LinearKernel(input_nc, output_nc, kernel_size, dimension=dimension)
    elif netG == 'linearkernel_double':
        net = LinearKernel_double(input_nc, output_nc, kernel_size, dimension=dimension)
    elif netG == 'linearkernel_LK31':
        net = LinearKernel(input_nc, output_nc, 31, dimension=dimension)
    elif netG == 'linearkernel_NC':
        net = LinearKernel_NC(input_nc, output_nc, kernel_size, dimension=dimension)
    elif netG == 'fixed_kernel':
        net = FixedLinearKernel(given_psf, noise_setting)
    elif netG == 'deep_linear_gen':
        net = DeepLinearGenerator(input_nc, output_nc)
    else:
        raise NotImplementedError(f'Generator model name [{netG}] is not recognized')

    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02,
             use_sigmoid=False, gpu_ids=None, dimension=3):
    """Create a discriminator."""
    gpu_ids = gpu_ids or []
    net = None
    norm_layer = get_norm_layer(norm_type=norm, dimension=dimension)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, use_sigmoid=use_sigmoid,
                                  norm_layer=norm_layer, dimension=dimension)
    elif netD == 'basic_SN':
        net = NLayerDiscriminatorSN(input_nc, ndf, n_layers=3, use_sigmoid=use_sigmoid,
                                    norm_layer=norm_layer, dimension=dimension)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid,
                                  norm_layer=norm_layer, dimension=dimension)
    elif netD == 'n_layers_SN':
        net = NLayerDiscriminatorSN(input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid,
                                    norm_layer=norm_layer, dimension=dimension)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, dimension=dimension)
    elif netD == 'kernelGAN':
        net = KernelPatchDiscriminator(input_nc, ndf, n_layers=5, norm_layer=norm_layer, dimension=dimension)
    else:
        raise NotImplementedError(f'Discriminator model name [{netD}] is not recognized')

    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# Losses
###############################################################################

class GANLoss(nn.Module):
    """GAN objective wrapper supporting vanilla, LSGAN, and WGAN(-GP)."""
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif 'wgan' in gan_mode:
            self.loss = None
        else:
            raise NotImplementedError(f'gan mode {gan_mode} not implemented')

    def get_target_tensor(self, prediction, target_is_real: bool):
        tgt = self.real_label if target_is_real else self.fake_label
        return tgt.expand_as(prediction)

    def forward(self, prediction, target_is_real: bool):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target_tensor)
        # WGAN family
        return -prediction.mean() if target_is_real else prediction.mean()


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Gradient penalty for WGAN-GP."""
    if lambda_gp <= 0.0:
        return 0.0, None

    if type == 'real':
        interpolatesv = real_data
    elif type == 'fake':
        interpolatesv = fake_data
    elif type == 'mixed':
        alpha = torch.rand(real_data.shape[0], 1, device=device)
        alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
        interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
    else:
        raise NotImplementedError(f'{type} not implemented')

    interpolatesv.requires_grad_(True)
    disc_interpolates = netD(interpolatesv)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolatesv,
        grad_outputs=torch.ones_like(disc_interpolates, device=device),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(real_data.size(0), -1)
    gradients_norm = (gradients + 1e-16).norm(2, dim=1)
    gradient_penalty = ((gradients_norm - constant) ** 2).mean() * lambda_gp
    return gradient_penalty, gradients


###############################################################################
# U-Net family (3D by default)
###############################################################################

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm_layer=None, dimension=3):
        super().__init__()
        _conv = conv(dimension)
        self.convolution = nn.Sequential(
            _conv(in_channels, out_channels, kernel_size, stride, padding),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            _conv(out_channels, out_channels, kernel_size, stride, padding),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convolution(x)


class last_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm_layer=None, dimension=3):
        super().__init__()
        _conv = conv(dimension)
        self.convolution = nn.Sequential(
            _conv(in_channels, out_channels, kernel_size, stride, padding),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convolution(x)


class triple_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm_layer=None, dimension=3):
        super().__init__()
        _conv = conv(dimension)
        self.convolution = nn.Sequential(
            _conv(in_channels, out_channels, kernel_size, stride, padding),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            _conv(out_channels, out_channels, kernel_size, stride, padding),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            _conv(out_channels, out_channels, kernel_size, stride, padding),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convolution(x)


class Unet_deconv(nn.Module):
    """3D U-Net with symmetric pooling/upsampling (2x2x2)."""
    def __init__(self, input_nc, output_nc, norm_layer=None, dimension=3):
        super().__init__()
        assert dimension == 3, "Unet_deconv is 3D; for 2D use a 2D variant."
        _maxpool = maxpool(dimension)
        _conv = conv(dimension)
        _t = convtranspose(dimension)

        start_nc = input_nc * 64

        # Down
        self.double_conv1 = double_conv(input_nc, start_nc, 3, 1, 1, norm_layer, dimension)
        self.maxpool1 = _maxpool(2)

        self.double_conv2 = double_conv(start_nc, start_nc * 2, 3, 1, 1, norm_layer, dimension)
        self.maxpool2 = _maxpool(2)

        # Bottom
        self.bottom_layer = triple_conv(start_nc * 2, start_nc * 4, 3, 1, 1, norm_layer, dimension)

        # Up
        self.t_conv2 = _t(start_nc * 4, start_nc * 2, 2, 2)
        self.ex_double_conv2 = double_conv(start_nc * 4, start_nc * 2, 3, 1, 1, norm_layer, dimension)

        self.t_conv1 = _t(start_nc * 2, start_nc, 2, 2)
        self.ex_conv1_1 = last_conv(start_nc * 2, start_nc, 3, 1, 1, norm_layer, dimension)

        # Final
        self.one_by_one = _conv(start_nc, output_nc, 1, 1, 0)
        self.one_by_one_2 = _conv(output_nc, output_nc, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

        self.debug = False  # set True to print shapes

    def forward(self, x):
        # Down
        c1 = self.double_conv1(x)
        if self.debug: print("conv1:", c1.shape)
        p1 = self.maxpool1(c1)

        c2 = self.double_conv2(p1)
        if self.debug: print("conv2:", c2.shape)
        p2 = self.maxpool2(c2)

        # Bottom
        cb = self.bottom_layer(p2)
        if self.debug: print("conv_bottom:", cb.shape)

        # Up 2
        u2 = self.t_conv2(cb)
        if self.debug: print("u2 (pre-align):", u2.shape)
        if u2.shape[2:] != c2.shape[2:]:
            u2 = F.interpolate(u2, size=c2.shape[2:], mode='trilinear', align_corners=False)
            if self.debug: print("u2 (aligned):", u2.shape)
        e2 = self.ex_double_conv2(torch.cat([c2, u2], dim=1))

        # Up 1
        u1 = self.t_conv1(e2)
        if self.debug: print("u1 (pre-align):", u1.shape)
        if u1.shape[2:] != c1.shape[2:]:
            u1 = F.interpolate(u1, size=c1.shape[2:], mode='trilinear', align_corners=False)
            if self.debug: print("u1 (aligned):", u1.shape)
        e1 = self.ex_conv1_1(torch.cat([c1, u1], dim=1))

        y = self.one_by_one(e1)
        y2 = self.one_by_one_2(y)
        out = self.sigmoid(y2)
        return out


class Unet_deconv_aniso(nn.Module):
    """
    3D U-Net variant that *does not* downsample along Z.
    Pool/upsample with (1,2,2) to respect anisotropy (thin Z, large XY).
    """
    def __init__(self, input_nc, output_nc, norm_layer=None):
        super().__init__()
        _conv = conv(3)
        _t = convtranspose(3)

        start_nc = input_nc * 64

        # Down
        self.double_conv1 = double_conv(input_nc, start_nc, 3, 1, 1, norm_layer, 3)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.double_conv2 = double_conv(start_nc, start_nc * 2, 3, 1, 1, norm_layer, 3)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Bottom
        self.bottom_layer = triple_conv(start_nc * 2, start_nc * 4, 3, 1, 1, norm_layer, 3)

        # Up (mirror XY only)
        self.t_conv2 = _t(start_nc * 4, start_nc * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.ex_double_conv2 = double_conv(start_nc * 4, start_nc * 2, 3, 1, 1, norm_layer, 3)

        self.t_conv1 = _t(start_nc * 2, start_nc, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.ex_conv1_1 = last_conv(start_nc * 2, start_nc, 3, 1, 1, norm_layer, 3)

        # Final
        self.one_by_one = _conv(start_nc, output_nc, 1, 1, 0)
        self.one_by_one_2 = _conv(output_nc, output_nc, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1 = self.double_conv1(x); p1 = self.maxpool1(c1)
        c2 = self.double_conv2(p1); p2 = self.maxpool2(c2)
        cb = self.bottom_layer(p2)

        u2 = self.t_conv2(cb)
        e2 = self.ex_double_conv2(torch.cat([c2, u2], dim=1))

        u1 = self.t_conv1(e2)
        e1 = self.ex_conv1_1(torch.cat([c1, u1], dim=1))

        y = self.one_by_one(e1)
        y2 = self.one_by_one_2(y)
        return self.sigmoid(y2)


class Unet_vanilla(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=None, dimension=3):
        super().__init__()
        _maxpool = maxpool(dimension)
        _conv = conv(dimension)
        _t = convtranspose(dimension)

        start_nc = input_nc * 64

        # Down
        self.double_conv1 = double_conv(input_nc, start_nc, 3, 1, 1, norm_layer, dimension)
        self.maxpool1 = _maxpool(2)

        self.double_conv2 = double_conv(start_nc, start_nc * 2, 3, 1, 1, norm_layer, dimension)
        self.maxpool2 = _maxpool(2)

        self.double_conv3 = double_conv(start_nc * 2, start_nc * 4, 3, 1, 1, norm_layer, dimension)
        self.maxpool3 = _maxpool(2)

        # Bottom
        self.bottom_layer = double_conv(start_nc * 4, start_nc * 8, 3, 1, 1, norm_layer, dimension)

        # Up
        self.t_conv3 = _t(start_nc * 8, start_nc * 4, 2, 2)
        self.ex_double_conv3 = double_conv(start_nc * 8, start_nc * 4, 3, 1, 1, norm_layer, dimension)

        self.t_conv2 = _t(start_nc * 4, start_nc * 2, 2, 2)
        self.ex_double_conv2 = double_conv(start_nc * 4, start_nc * 2, 3, 1, 1, norm_layer, dimension)

        self.t_conv1 = _t(start_nc * 2, start_nc, 2, 2)
        self.ex_conv1_1 = double_conv(start_nc * 2, start_nc, 3, 1, 1, norm_layer, dimension)

        # Final
        self.one_by_one = _conv(start_nc, output_nc, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1 = self.double_conv1(x); p1 = self.maxpool1(c1)
        c2 = self.double_conv2(p1); p2 = self.maxpool2(c2)
        c3 = self.double_conv3(p2); p3 = self.maxpool3(c3)

        cb = self.bottom_layer(p3)

        u3 = self.t_conv3(cb)
        e3 = self.ex_double_conv3(torch.cat([c3, u3], 1))

        u2 = self.t_conv2(e3)
        e2 = self.ex_double_conv2(torch.cat([c2, u2], 1))

        u1 = self.t_conv1(e2)
        e1 = self.ex_conv1_1(torch.cat([c1, u1], 1))

        out = self.sigmoid(self.one_by_one(e1))
        return out


class Unet_vanilla_shallow(nn.Module):
    """Fixed shallow U-Net (2 downs / 2 ups)."""
    def __init__(self, input_nc, output_nc, norm_layer=None, dimension=3):
        super().__init__()
        _maxpool = maxpool(dimension)
        _conv = conv(dimension)
        _t = convtranspose(dimension)

        start_nc = input_nc * 64

        self.double_conv1 = double_conv(input_nc, start_nc, 3, 1, 1, norm_layer, dimension)
        self.maxpool1 = _maxpool(2)

        self.double_conv2 = double_conv(start_nc, start_nc * 2, 3, 1, 1, norm_layer, dimension)
        self.maxpool2 = _maxpool(2)

        self.bottom_layer = double_conv(start_nc * 2, start_nc * 4, 3, 1, 1, norm_layer, dimension)

        self.t_conv2 = _t(start_nc * 4, start_nc * 2, 2, 2)
        self.ex_double_conv2 = double_conv(start_nc * 4, start_nc * 2, 3, 1, 1, norm_layer, dimension)

        self.t_conv1 = _t(start_nc * 2, start_nc, 2, 2)
        self.ex_conv1_1 = double_conv(start_nc * 2, start_nc, 3, 1, 1, norm_layer, dimension)

        self.one_by_one = _conv(start_nc, output_nc, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1 = self.double_conv1(x); p1 = self.maxpool1(c1)
        c2 = self.double_conv2(p1); p2 = self.maxpool2(c2)
        cb = self.bottom_layer(p2)

        u2 = self.t_conv2(cb)
        e2 = self.ex_double_conv2(torch.cat([c2, u2], 1))

        u1 = self.t_conv1(e2)
        e1 = self.ex_conv1_1(torch.cat([c1, u1], 1))

        return self.sigmoid(self.one_by_one(e1))


###############################################################################
# VGG (2D) for perceptual loss (unchanged)
###############################################################################

VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_net(nn.Module):
    """Simple VGG head for perceptual features / classification (2D)."""
    def __init__(self, input_nc, num_classes, VGG_type):
        super().__init__()
        self.in_channels = input_nc
        self.conv_layers = self.create_conv_layers(VGG_types[VGG_type])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fcs(x)

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if isinstance(x, int):
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(True)]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)


###############################################################################
# ResNet generator (2D) — unchanged from your code
###############################################################################

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert n_blocks >= 0
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type, norm_layer, use_dropout, use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Sigmoid()]  # your choice

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'padding [{padding_type}] is not implemented')

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'padding [{padding_type}] is not implemented')

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


###############################################################################
# Linear kernels (3D default)
###############################################################################

class LinearKernel(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, dimension=3):
        super().__init__()
        _conv = conv(dimension)
        p = int(np.round((kernel_size - 1) / 2))
        padding_size = (p, p) if dimension == 2 else (p, p, p)
        self.convlayer = _conv(input_nc, output_nc, kernel_size=kernel_size, stride=1, padding=padding_size, bias=False)

    def forward(self, x):
        return self.convlayer(x)


class LinearKernel_double(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, dimension=3):
        super().__init__()
        _conv = conv(dimension)
        p = int(np.round((kernel_size - 1) / 2))
        padding_size = (p, p) if dimension == 2 else (p, p, p)
        self.convlayer = _conv(input_nc, output_nc, kernel_size=kernel_size, stride=1, padding=padding_size, bias=False)

    def forward(self, x):
        h = self.convlayer(x)
        return self.convlayer(h)


class LinearKernel_NC(nn.Module):
    """Linear kernel + learnable noise branch."""
    def __init__(self, input_nc, output_nc, kernel_size, dimension=3):
        super().__init__()
        _conv = conv(dimension)
        p = int(np.round((kernel_size - 1) / 2))
        padding_size = (p, p) if dimension == 2 else (p, p, p)
        self.blur_convlayer = _conv(input_nc, output_nc, kernel_size=kernel_size, stride=1, padding=padding_size, bias=False)
        self.noise_convlayer = _conv(input_nc, output_nc, kernel_size=kernel_size, stride=1, padding=padding_size, bias=False)

    def forward(self, x):
        return self.blur_convlayer(x) + self.noise_convlayer(x)


class DeepLinearGenerator(nn.Module):
    """3D deep linear generator (no downsampling at the end)."""
    def __init__(self, input_nc, output_nc):
        super().__init__()
        narrowing_kernels = [7, 5, 3]
        unit_kernels = [1, 1]
        self.first_layer = nn.Conv3d(in_channels=input_nc, out_channels=input_nc * 64,
                                     kernel_size=narrowing_kernels[0], padding=3, bias=False)

        fb = []
        fb += [nn.Conv3d(input_nc * 64, input_nc * 64, kernel_size=narrowing_kernels[1], padding=2, bias=False)]
        fb += [nn.Conv3d(input_nc * 64, input_nc * 64, kernel_size=narrowing_kernels[2], padding=1, bias=False)]
        for layer in range(len(unit_kernels)):
            in_c = input_nc * int(64 * ((1 / 2) ** layer))
            out_c = input_nc * int(64 * ((1 / 2) ** (layer + 1)))
            fb += [nn.Conv3d(in_c, out_c, kernel_size=unit_kernels[layer], padding=0, bias=False)]
        self.feature_block = nn.Sequential(*fb)

        last_out_c = input_nc * int(64 * ((1 / 2) ** (len(unit_kernels))))
        self.final_layer = nn.Conv3d(in_channels=last_out_c, out_channels=output_nc, kernel_size=unit_kernels[-1], padding=0, bias=False)

    def forward(self, x):
        h = self.first_layer(x)
        h = self.feature_block(h)
        return self.final_layer(h)


class FixedLinearKernel(nn.Module):
    """3D fixed PSF convolution + noise injection."""
    def __init__(self, psf, noise_setting):
        super().__init__()
        self.psf = nn.Parameter(psf, requires_grad=False)
        self.kernel_size = np.asarray(self.psf.shape[2:])
        self.gau_sigma, self.poisson_peak = noise_setting

    def forward(self, x):
        padding_size = tuple(int(np.round((k - 1) / 2)) for k in self.kernel_size)
        y = F.conv3d(x, self.psf, stride=1, padding=padding_size)
        if self.kernel_size[-1] % 2 == 0:
            y = y[:, :, 1:, 1:, 1:]
        y = noisy('gauss', y, sigma=self.gau_sigma, is_tensor=True)
        y = noisy('poisson', y, peak=self.poisson_peak, is_tensor=True)
        return y


###############################################################################
# Discriminators
###############################################################################

class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator (2D/3D)."""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=None, use_sigmoid=False, dimension=3):
        super().__init__()
        _conv = conv(dimension)

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == instance_norm(dimension)
        else:
            use_bias = norm_layer == instance_norm(dimension)

        kw, padw = 4, 1
        seq = [
            _conv(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            seq += [
                _conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        seq += [
            _conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        seq += [_conv(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            print("Using sigmoid in the last D layer (LSGAN-friendly).")
            seq += [nn.Sigmoid()]

        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminatorSN(nn.Module):
    """Spectral-norm PatchGAN."""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=None, use_sigmoid=False, dimension=3):
        super().__init__()
        _conv = conv(dimension)
        use_bias = False

        kw, padw = 4, 1
        seq = [
            nn.utils.spectral_norm(_conv(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            seq += [
                nn.utils.spectral_norm(_conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        seq += [
            nn.utils.spectral_norm(_conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True)
        ]

        seq += [nn.utils.spectral_norm(_conv(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]
        if use_sigmoid:
            seq += [nn.Sigmoid()]

        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)


class KernelPatchDiscriminator(nn.Module):
    """KernelGAN-style patch discriminator (1x1 stack after a 7x receptive layer)."""
    def __init__(self, input_nc, ndf=64, n_layers=5, norm_layer=None, dimension=3):
        super().__init__()
        _conv = conv(dimension)

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == instance_norm(dimension)
        else:
            use_bias = norm_layer == instance_norm(dimension)

        self.first_layer = _conv(in_channels=input_nc, out_channels=ndf, kernel_size=7, bias=use_bias)

        fb = []
        for _ in range(1, n_layers - 1):
            fb += [
                _conv(in_channels=ndf, out_channels=ndf, kernel_size=1, bias=use_bias),
                norm_layer(ndf),
                nn.ReLU(True)
            ]
        self.feature_block = nn.Sequential(*fb)
        self.final_layer = _conv(in_channels=ndf, out_channels=1, kernel_size=1, bias=use_bias)

    def forward(self, x):
        r = self.first_layer(x)
        f = self.feature_block(r)
        return self.final_layer(f)


class PixelDiscriminator(nn.Module):
    """1x1 PatchGAN (pixel-wise)."""
    def __init__(self, input_nc, ndf=64, norm_layer=batch_norm, dimension=3):
        super().__init__()
        _conv = conv(dimension)

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == instance_norm(dimension)
        else:
            use_bias = norm_layer == instance_norm(dimension)

        self.net = nn.Sequential(
            _conv(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            _conv(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            _conv(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
        )

    def forward(self, x):
        return self.net(x)


###############################################################################
# Legacy U-Net (two-output) kept for completeness
###############################################################################

class UnetTwoOuts(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        self.double_conv1 = double_conv(1, input_nc, 3, 1, 1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.double_conv2 = double_conv(input_nc, input_nc * 2, 3, 1, 1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.double_conv3 = double_conv(input_nc * 2, input_nc * 4, 3, 1, 1)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        self.double_conv5 = double_conv(input_nc * 4, input_nc * 8, 3, 1, 1)

        self.t_conv3 = nn.ConvTranspose3d(input_nc * 8, input_nc * 4, 2, 2)
        self.ex_double_conv3 = double_conv(input_nc * 8, input_nc * 4, 3, 1, 1)

        self.t_conv2 = nn.ConvTranspose3d(input_nc * 4, input_nc * 2, 2, 2)
        self.ex_double_conv2 = double_conv(input_nc * 4, input_nc * 2, 3, 1, 1)

        self.t_conv1 = nn.ConvTranspose3d(input_nc * 2, input_nc, 2, 2)
        self.ex_double_conv1 = double_conv(input_nc * 2, input_nc, 3, 1, 1)

        self.one_by_one = nn.Conv3d(input_nc, output_nc, 1, 1, 0)
        self.one_by_one_2 = double_conv(input_nc, 1, 1, 1, 0)

    def forward(self, x):
        c1 = self.double_conv1(x); p1 = self.maxpool1(c1)
        c2 = self.double_conv2(p1); p2 = self.maxpool2(c2)
        c3 = self.double_conv3(p2); p3 = self.maxpool3(c3)

        c5 = self.double_conv5(p3)

        u3 = self.t_conv3(c5)
        e3 = self.ex_double_conv3(torch.cat([c3, u3], 1))

        u2 = self.t_conv2(e3)
        e2 = self.ex_double_conv2(torch.cat([c2, u2], 1))

        u1 = self.t_conv1(e2)
        e1 = self.ex_double_conv1(torch.cat([c1, u1], 1))

        y1 = self.one_by_one(e1)
        y2 = self.one_by_one_2(e1)
        return y1, y2
