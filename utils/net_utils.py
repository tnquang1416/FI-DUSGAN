'''
Created on Jul 27, 2021

@author: Quang Tran
'''

import torch
from torch import nn
from nets import flow_nets

        
def flownets(path=None, input_channels=6, is_batchnorm=True):
    """
    FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)
    :param path:where to load pretrained network. will create a new one if not set
    """
    model = flow_nets.FlowNetS(input_channels=input_channels, batchNorm=is_batchnorm)
    if path is not None:
        data = torch.load(path, map_location=lambda storage, loc: storage)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
    
    return model

# end flowNetS define/load


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    '''
    @author: https://github.com/baowenbo/MEMC-Net/blob/82b6e25f6446496b7fecc47704f4422d1d6ac9a2/networks/FlowNetS/FlowNetS.py#L10
    '''
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
# end conv
        

def deconv(in_planes, out_planes):
    '''
    @author: https://github.com/baowenbo/MEMC-Net/blob/82b6e25f6446496b7fecc47704f4422d1d6ac9a2/networks/FlowNetS/FlowNetS.py#L26
    '''
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )
    
# end deconv
    
    
def predict_flow(in_planes):
    '''
    @author: https://github.com/baowenbo/MEMC-Net/blob/82b6e25f6446496b7fecc47704f4422d1d6ac9a2/networks/FlowNetS/FlowNetS.py#L23
    '''
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False)

# end predict_flow


def conv_relu_conv(input_filter, output_filter, kernel_size, padding):

    layers = nn.Sequential(
        nn.Conv2d(input_filter, input_filter, kernel_size, 1, padding),
        nn.ReLU(inplace=False),
        nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
    )
    return layers


def conv_relu(input_filter, output_filter, kernel_size, padding):
    layers = nn.Sequential(*[
        nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),

        nn.ReLU(inplace=False)
    ])
    return layers


def conv_relu_maxpool(input_filter, output_filter, kernel_size, padding, kernel_size_pooling):

    layers = nn.Sequential(*[
        nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),

        nn.ReLU(inplace=False),

        nn.BatchNorm2d(output_filter),

        nn.MaxPool2d(kernel_size_pooling)
    ])
    return layers


def conv_relu_unpool(input_filter, output_filter, kernel_size, padding, unpooling_factor):

    layers = nn.Sequential(*[
        nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),

        nn.ReLU(inplace=False),

        nn.BatchNorm2d(output_filter),

        nn.Upsample(scale_factor=unpooling_factor, mode='bilinear')
    ])
    return layers


def get_MonoNet5(channel_in, channel_out, name):

    '''
    @author: https://github.com/baowenbo/MEMC-Net/blob/82b6e25f6446496b7fecc47704f4422d1d6ac9a2/networks/MEMC_Net.py#L180
    Generally, the MonoNet is aimed to provide a basic module for generating either offset, or filter, or occlusion.
    :param channel_in: number of channels that composed of multiple useful information like reference frame, previous coarser-scale result
    :param channel_out: number of output the offset or filter or occlusion
    :param name: to distinguish between offset, filter and occlusion, since they should use different activations in the last network layer
    :return: output the network model
    '''
    model = []

    # block1
    model += conv_relu(channel_in * 2, 32, (3, 3), (1, 1))
    model += conv_relu(32, 32, (3, 3), (1, 1))
    model += conv_relu_maxpool(32, 32, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.5
    # block2
    model += conv_relu(32, 64, (3, 3), (1, 1))
    model += conv_relu_maxpool(64, 64, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.4
    # block3
    model += conv_relu(64, 128, (3, 3), (1, 1))
    model += conv_relu_maxpool(128, 128, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.3
    # block4
    model += conv_relu(128, 256, (3, 3), (1, 1))
    model += conv_relu_maxpool(256, 256, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.2
    # block5
    model += conv_relu(256, 512, (3, 3), (1, 1))
    model += conv_relu_maxpool(512, 512, (3, 3), (1, 1), (2, 2))

    # intermediate block5_5
    model += conv_relu(512, 512, (3, 3), (1, 1))
    model += conv_relu(512, 512, (3, 3), (1, 1))

    # block 6
    model += conv_relu_unpool(512, 512, (3, 3), (1, 1), 2)  # THE OUTPUT No.1 UP
    model += conv_relu(512, 256, (3, 3), (1, 1))
    # block 7
    model += conv_relu_unpool(256, 256, (3, 3), (1, 1), 2)  # THE OUTPUT No.2 UP
    model += conv_relu(256, 128, (3, 3), (1, 1))
    # block 8
    model += conv_relu_unpool(128, 128, (3, 3), (1, 1), 2)  # THE OUTPUT No.3 UP
    model += conv_relu(128, 64, (3, 3), (1, 1))

    # block 9
    model += conv_relu_unpool(64, 64, (3, 3), (1, 1), 2)  # THE OUTPUT No.4 UP
    model += conv_relu(64, 32, (3, 3), (1, 1))

    # block 10
    model += conv_relu_unpool(32, 32, (3, 3), (1, 1), 2)  # THE OUTPUT No.5 UP
    model += conv_relu(32, 16, (3, 3), (1, 1))

    # output our final purpose
    branch1 = []
    branch2 = []
    branch1 += conv_relu_conv(16, channel_out, (3, 3), (1, 1))
    branch2 += conv_relu_conv(16, channel_out, (3, 3), (1, 1))

    return  (nn.ModuleList(model), nn.ModuleList(branch1), nn.ModuleList(branch2))

# end get_MonoNet5
