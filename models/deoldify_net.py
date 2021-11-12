"""
*******************************************************************************************
*******************************************************************************************
**                                                                                       **
**                                      GENRATOR                                         **
**                                                                                       **
*******************************************************************************************
*******************************************************************************************
"""

#Code from https://dthiagarajan.github.io/technical_blog/draft/pytorch/hooks/2020/03/18/Dynamic-UNet-and-PyTorch-Hooks.html

 
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import tifffile as tiff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.transforms.functional as tf
from tqdm.notebook import tqdm
import torchvision


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNetEncoder(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNetEncoder, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = nn.Sequential(self.conv1, self.bn1, self.relu)
        self.layer1 = nn.Sequential(self.maxpool, self._make_layer(block, 64, layers[0]))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.out_dim = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.cat([x,x,x],1)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEncoder(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEncoder(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEncoder(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print("load Resnet50 Pretrain")
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEncoder(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model

def defineBackbone(name,pretrained=False):
    if name == "resnet18":
        return resnet18(pretrained)
    elif name == "resnet34":
        return resnet34(pretrained)
    elif name == "resnet50":
        return resnet50(pretrained)
    elif name == "resnet101":
        return resnet101(pretrained)
    else:
        print("No defind",name)
        raise 
    

def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEncoder(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model

class ConvLayer(nn.Module):
    def __init__(self, num_inputs, num_filters, bn=True, kernel_size=3, stride=1,
                 padding=None, transpose=False, dilation=1):
        super(ConvLayer, self).__init__()
        if padding is None:
            padding = (kernel_size-1)//2 if transpose is not None else 0
        if transpose:
            self.layer = nn.ConvTranspose2d(num_inputs, num_filters, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation)
        else:
            self.layer = nn.Conv2d(num_inputs, num_filters, kernel_size=kernel_size,
                                   stride=stride, padding=padding)
        nn.init.kaiming_uniform_(self.layer.weight, a=np.sqrt(5))
        self.bn_layer = nn.BatchNorm2d(num_filters) if bn else None

    def forward(self, x):
        out = self.layer(x)
        out = F.relu(out)
        return out if self.bn_layer is None else self.bn_layer(out)
    
class ConcatLayer(nn.Module):
    def forward(self, x, dim=1):
        return torch.cat(list(x.values()), dim=dim)
    
class LambdaLayer(nn.Module):
    def __init__(self, f):
        super(LambdaLayer, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)

def upconv2x2(inplanes, outplanes, size=None, stride=1):
    if size is not None:
        return [
            ConvLayer(inplanes, outplanes, kernel_size=2, dilation=2, stride=stride),
            nn.Upsample(size=size, mode='bilinear', align_corners=True)
        ] 
    else:
        return [
            ConvLayer(inplanes, outplanes, kernel_size=2, dilation=2, stride=stride),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        ]

class DecoderConnect(nn.Module):
    def __init__(self, inplanes, output_size):
        super(DecoderConnect, self).__init__()
        self.bottom_process = nn.Sequential(
            ConvLayer(inplanes, inplanes * 2, kernel_size=3),
            ConvLayer(inplanes * 2, inplanes * 2, kernel_size=3),
            *upconv2x2(inplanes * 2, inplanes, size=output_size)
        )
        self.concat_process = nn.Sequential(
            ConcatLayer(),
            ConvLayer(inplanes * 2, inplanes * 2, kernel_size=1),
            ConvLayer(inplanes * 2, inplanes, kernel_size=3),
            ConvLayer(inplanes, inplanes, kernel_size=3)
        )
        
    def forward(self, x):
        decoder_input = self.bottom_process(x)
        return self.concat_process({0: x, 1: decoder_input})

class DynamicUnet(nn.Module):
    def __init__(self, encoder, input_size=(224, 224),num_in_channels = 1, num_output_channels=None, verbose=0):
        super(DynamicUnet, self).__init__()
        self.encoder = encoder
        self.verbose = verbose
        self.input_size = input_size
        self.num_input_channels = num_in_channels  # This must be 3 because we're using a ResNet encoder
        self.num_output_channels = num_output_channels
        
        self.decoder = self.setup_decoder()
        
    def forward(self, x):
        encoder_outputs = []
        def encoder_output_hook(self, input, output):
            encoder_outputs.append(output)

        handles = [
            child.register_forward_hook(encoder_output_hook) for name, child in self.encoder.named_children()
            if name.startswith('layer')
        ]

        try:
            self.encoder(x)
        finally:
            if self.verbose >= 1:
                print("Removing all forward handles")
            for handle in handles:
                handle.remove()
        prev_output = None
        for reo, rdl in zip(reversed(encoder_outputs), self.decoder):
            if prev_output is not None:
                prev_output = rdl({0: reo, 1: prev_output})
            else:
                prev_output = rdl(reo)
        return prev_output
                
    def setup_decoder(self):
        input_sizes = []
        output_sizes = []
        def shape_hook(self, input, output):
            input_sizes.append(input[0].shape)
            output_sizes.append(output.shape)

        handles = [
            child.register_forward_hook(shape_hook) for name, child in self.encoder.named_children()
            if name.startswith('layer')
        ]    

        self.encoder.eval()
        test_input = torch.randn(1, self.num_input_channels, *self.input_size)
        try:
            self.encoder(test_input)
        finally:
            if self.verbose >= 1:
                print("Removing all shape hook handles")
            for handle in handles:
                handle.remove()
        decoder = self.construct_decoder(input_sizes, output_sizes, num_output_channels=self.num_output_channels)
        return decoder
        
    def construct_decoder(self, input_sizes, output_sizes, num_output_channels=None):
        decoder_layers = []
        for layer_index, (input_size, output_size) in enumerate(zip(input_sizes, output_sizes)):
            upsampling_size_factor = int(input_size[-1] / output_size[-1])
            upsampling_channel_factor = input_size[-3] / output_size[-3]
            next_layer = []
            bs, c, h, w = input_size
            ops = []
            if layer_index == len(input_sizes) - 1:
                last_layer_ops = DecoderConnect(output_size[-3], output_size[2:])
                last_layer_ops_input = torch.randn(*output_size)
                last_layer_concat_ops_output = last_layer_ops(last_layer_ops_input)
                next_layer.extend([last_layer_ops])
                if upsampling_size_factor > 1 or upsampling_channel_factor != 1:
                    last_layer_concat_upconv_op = upconv2x2(output_size[-3], input_size[-3], size=input_size[2:])
                    last_layer_concat_upconv_op_output = nn.Sequential(*last_layer_concat_upconv_op)(
                        last_layer_concat_ops_output
                    )
                    next_layer.extend(last_layer_concat_upconv_op)
            elif layer_index == 0:
                first_layer_concat_ops = [
                    ConcatLayer(),
                    ConvLayer(output_size[-3] * 2, output_size[-3] * 2, kernel_size=1),
                    *upconv2x2(
                        output_size[-3] * 2,
                        output_size[-3],
                        size=[dim * upsampling_size_factor for dim in output_size[2:]]
                    ),
                    ConvLayer(output_size[-3], output_size[-3], kernel_size=3),
                    ConvLayer(
                        output_size[-3],
                        input_size[-3] if self.num_output_channels is None else self.num_output_channels,
                        kernel_size=1
                    ),
                ]
                first_layer_concat_ops_output = nn.Sequential(*first_layer_concat_ops)(
                    {0: torch.randn(*output_size), 1: torch.randn(*output_size)}
                )
                next_layer.extend(first_layer_concat_ops)
            else:
                middle_layer_concat_ops = [
                    ConcatLayer(),
                    ConvLayer(output_size[-3] * 2, output_size[-3] * 2, kernel_size=1),
                    ConvLayer(output_size[-3] * 2, output_size[-3], kernel_size=3),
                    ConvLayer(output_size[-3], output_size[-3], kernel_size=3)
                ]
                middle_layer_concat_ops_output = nn.Sequential(*middle_layer_concat_ops)(
                    {0: torch.randn(*output_size), 1: torch.randn(*output_size)}
                )
                next_layer.extend(middle_layer_concat_ops)
                if upsampling_size_factor > 1 or upsampling_channel_factor != 1:
                    middle_layer_concat_upconv_op = upconv2x2(output_size[-3], input_size[-3], size=input_size[2:])
                    middle_layer_concat_upconv_op_output = nn.Sequential(*middle_layer_concat_upconv_op)(
                        middle_layer_concat_ops_output
                    )
                    next_layer.extend(middle_layer_concat_upconv_op)
            decoder_layers.append(nn.Sequential(*next_layer))
        return nn.ModuleList(reversed(decoder_layers))

"""
*******************************************************************************************
*******************************************************************************************
**                                                                                       **
**                                      DISCRIMINATOR                                    **
**                                                                                       **
*******************************************************************************************
*******************************************************************************************
"""





class  NLayerDiscriminator(nn.Module):
    """
    NLayerDiscriminator 
    Parameters:
        in_channels (int)       -- the number of channels in input images
    """
    def __init__(self,in_channels,n_layers=3):
        super(NLayerDiscriminator,self).__init__()
        self.network = []
        self.network.append(nn.Conv2d(in_channels,64,
                            kernel_size = 4,stride = 2,padding=1))
        self.network.append(nn.LeakyReLU(0.2,True))
        
        use_bias = 0
        nf_mult = 1
        nf_mult_prev = 1
        ndf = 64
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.network.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                                kernel_size= 4, stride= 2, padding= 1,bias=use_bias))
            self.network.append(nn.BatchNorm2d(ndf * nf_mult))
            self.network.append(nn.LeakyReLU(0.2,True))
        

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.network.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,kernel_size= 4,
                            stride=1,padding=1,bias=use_bias))
        self.network.append(nn.BatchNorm2d(ndf * nf_mult))
        self.network.append(nn.LeakyReLU(0.2,True))

        self.network.append(nn.Conv2d(ndf * nf_mult, 1,kernel_size= 4,
                            stride=1,padding=1))
        self.network = nn.Sequential(*self.network)
    def forward(self,input):
        return self.network(input)

# def Defind_Dis(in_channels, init_type='normal', init_gain=0.02, gpu_ids=[]):
#     """
#     Parameters:
#         in_channels (int)     -- the number of channels in input images
#         init_type (str)    -- the name of the initialization method.
#         init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
#         gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
#     Returns a discriminator
#     """

#     net = NLayerDiscriminator(in_channels)
#     return init_net(net, init_type, init_gain, gpu_ids)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss