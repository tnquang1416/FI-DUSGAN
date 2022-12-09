'''
Created on Dec 14, 2020

@author: Quang Tran
'''

import torch
import torch.nn as nn

from utils import CONSTANT, utils


def Conv_Block(in_channels, out_channels=None, kernel_size=4, stride=2, padding=1):
    out_channels = in_channels * 2 if out_channels is None else out_channels
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.02, inplace=True)
        )
        
# end get_extraction_block

                
def TransConv_Block(in_channels, out_channels=None, kernel_size=4, stride=2, padding=1):
    out_channels = in_channels // 2 if out_channels is None else out_channels
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),  # size*2
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )
    
# end get_extraction_block 


class FrameInterpolationGenerator(nn.Module):
    '''
    The generator network of the proposed framework, in which follows the baseline and handle 128x128 input frames.
    '''

    def __init__(self, nfg=32):
        '''
        Constructor of FrameInterpolationGenerator
        
        :param nfg: feature map size
        '''
        super(FrameInterpolationGenerator, self).__init__()
        self.nfg = nfg  # the size of feature map
        self.c = 3  # output channel
        self.version = "down-up_generator"

        self._initialize_blocks()
        self._initialize_pre_gen_blocks()
    # end constructor
    
    def _initialize_blocks(self):
        # (9, 128, 128) --> (nfg*2, 128, 128)
        self.input_processing = nn.Sequential(
            nn.Conv2d(self.c * 3, self.nfg * 2, kernel_size=3, stride=1, padding=1, bias=False),  # size
            nn.BatchNorm2d(self.nfg * 2),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(self.nfg * 2, self.nfg * 2, kernel_size=3, stride=1, padding=1, bias=False),  # size
            nn.BatchNorm2d(self.nfg * 2),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(self.nfg * 2, self.nfg * 2, kernel_size=3, stride=1, padding=1, bias=False),  # size
            nn.BatchNorm2d(self.nfg * 2),
            nn.LeakyReLU(0.02, inplace=True),
            )
        
        self.ftr_extractor_1 = Conv_Block(self.nfg * 2, self.nfg * 2)
        self.synthesizer_0 = TransConv_Block(in_channels=self.nfg * (2 ** 1), out_channels=self.nfg * (2 ** 0))
        self.conv_0 = Conv_Block(in_channels=self.nfg + self.nfg * (2 ** 1), out_channels=self.nfg * (2 ** 0), kernel_size=3, stride=1, padding=1)
        
        self.ftr_extractor_2 = Conv_Block(in_channels=self.nfg * 2, out_channels=self.nfg * (2 ** 2))
        self.synthesizer_1 = TransConv_Block(in_channels=self.nfg * (2 ** 2), out_channels=self.nfg * (2 ** 1))
        self.conv_1 = Conv_Block(in_channels=2 * self.nfg * (2 ** 1), out_channels=self.nfg * (2 ** 1), kernel_size=3, stride=1, padding=1)
        
        self.ftr_extractor_3 = Conv_Block(in_channels=self.nfg * (2 ** 2), out_channels=self.nfg * (2 ** 3))        
        self.synthesizer_2 = TransConv_Block(in_channels=self.nfg * (2 ** 3), out_channels=self.nfg * (2 ** 2))
        self.conv_2 = Conv_Block(in_channels=2 * self.nfg * (2 ** 2), out_channels=self.nfg * (2 ** 2), kernel_size=3, stride=1, padding=1)
        
        self.ftr_processing = nn.Sequential(
            nn.ConvTranspose2d(self.nfg, self.c, kernel_size=3, stride=1, padding=1, bias=True),  # size
            nn.Tanh()
            )
        
    # end _initialize_blocks
    
    def _initialize_pre_gen_blocks(self):
        # (6, 64, 64) --> (nfg*2, 64, 64)
        self.pre_gen_input_processing0 = nn.Sequential(
            nn.Conv2d(self.c * 2, self.nfg * 2, kernel_size=3, stride=1, padding=1, bias=False),  # size
            nn.BatchNorm2d(self.nfg * 2),
            nn.LeakyReLU(0.02, inplace=True)
            )
        self.pre_gen_input_processing1 = nn.Sequential(
            nn.Conv2d(self.nfg * 2, self.nfg * 2, kernel_size=3, stride=1, padding=1, bias=False),  # size
            nn.BatchNorm2d(self.nfg * 2),
            nn.LeakyReLU(0.02, inplace=True)
            )
        self.pre_gen_input_processing2 = nn.Sequential(
            nn.Conv2d(self.nfg * 2, self.nfg * 2, kernel_size=3, stride=1, padding=1, bias=False),  # size
            nn.BatchNorm2d(self.nfg * 2),
            nn.LeakyReLU(0.02, inplace=True),
            )
        
        self.pre_gen_ftr_extractor_1 = Conv_Block(self.nfg * 2, self.nfg * 2)
        self.pre_gen_synthesizer_0 = TransConv_Block(in_channels=self.nfg * (2 ** 1), out_channels=self.nfg * (2 ** 0))
        self.pre_gen_conv_0 = Conv_Block(in_channels=self.nfg + self.nfg * (2 ** 1), out_channels=self.nfg * (2 ** 0), kernel_size=3, stride=1, padding=1)
        
        self.pre_gen_ftr_extractor_2 = Conv_Block(in_channels=self.nfg * 2, out_channels=self.nfg * (2 ** 2))
        self.pre_gen_synthesizer_1 = TransConv_Block(in_channels=self.nfg * (2 ** 2), out_channels=self.nfg * (2 ** 1))
        self.pre_gen_conv_1 = Conv_Block(in_channels=2 * self.nfg * (2 ** 1), out_channels=self.nfg * (2 ** 1), kernel_size=3, stride=1, padding=1)
        
        self.pre_gen_ftr_extractor_3 = Conv_Block(in_channels=self.nfg * (2 ** 2), out_channels=self.nfg * (2 ** 3))        
        self.pre_gen_synthesizer_2 = TransConv_Block(in_channels=self.nfg * (2 ** 3), out_channels=self.nfg * (2 ** 2))
        self.pre_gen_conv_2 = Conv_Block(in_channels=2 * self.nfg * (2 ** 2), out_channels=self.nfg * (2 ** 2), kernel_size=3, stride=1, padding=1)
        
        self.pre_gen_ftr_processing = nn.Sequential(
            nn.ConvTranspose2d(self.nfg, self.c, kernel_size=3, stride=1, padding=1, bias=True),  # size
            nn.Tanh()
            )
        
    # end _initialize_blocks

    def forward(self, frame1, frame2):
        '''
        forward the given input frames to the networks
        :param frame1: 4d tensor
        :param frame2: 4d tensor
        :param new_input: the difference between two input frames
        '''
        temp1 = nn.functional.interpolate(frame1, scale_factor=1 / 2, mode="bilinear")
        temp2 = nn.functional.interpolate(frame2, scale_factor=1 / 2, mode="bilinear")
        output = self._gen_pre_result(temp1, temp2)
        new_input = nn.functional.interpolate(output, scale_factor=2.0, mode="bilinear")
        
        input_padded = torch.cat((frame1, frame2, new_input), 1)
        ftr0 = self.input_processing(input_padded)
        
        ftr1 = self.ftr_extractor_1(ftr0)
        ftr2 = self.ftr_extractor_2(ftr1)
        ftr3 = self.ftr_extractor_3(ftr2)
        
        syn2 = self.synthesizer_2(ftr3)
        
        ftr_syn_2 = torch.cat((ftr2, syn2), 1)
        conv2 = self.conv_2(ftr_syn_2)
        syn1 = self.synthesizer_1(conv2)
        ftr_syn_1 = torch.cat((ftr1, syn1), 1)
        conv1 = self.conv_1(ftr_syn_1)
        syn0 = self.synthesizer_0(conv1)
        
        ftr_syn_0 = torch.cat((ftr0, syn0), 1)
        conv0 = self.conv_0(ftr_syn_0)            
        output_final = self.ftr_processing(conv0)

        return new_input, output_final
    
    def _gen_pre_result(self, frame1, frame2):
        '''
        forward the given input frames to the networks
        :param frame1: 4d tensor
        :param frame2: 4d tensor
        :param new_input: the difference between two input frames
        '''
#         print("Go down 0")
        input_cat = torch.cat((frame1, frame2), 1)
        temp_ftr0 = self.pre_gen_input_processing0(input_cat)
        temp_ftr1 = self.pre_gen_input_processing1(temp_ftr0)
        ftr0 = self.pre_gen_input_processing2(temp_ftr1)
#         print("Go down 1")
        ftr1 = self.pre_gen_ftr_extractor_1(ftr0)
        ftr2 = self.pre_gen_ftr_extractor_2(ftr1)
        ftr3 = self.pre_gen_ftr_extractor_3(ftr2)
#         print("Go down 2")
        syn2 = self.pre_gen_synthesizer_2(ftr3)
#         print("Go up 2")
        ftr_syn_2 = torch.cat((ftr2, syn2), 1)
        conv2 = self.pre_gen_conv_2(ftr_syn_2)
        syn1 = self.pre_gen_synthesizer_1(conv2)
#         print("Go up 1")
        ftr_syn_1 = torch.cat((ftr1, syn1), 1)
        conv1 = self.pre_gen_conv_1(ftr_syn_1)
        syn0 = self.pre_gen_synthesizer_0(conv1)
#         print("Go up 0")
        ftr_syn_0 = torch.cat((ftr0, syn0), 1)
        conv0 = self.pre_gen_conv_0(ftr_syn_0)            
        output = self.pre_gen_ftr_processing(conv0)

        return output
    
    # end forward
    
    def __str__(self):
        '''
        To string methods that ignore printing skip connections
        '''
        str_net = "%s First generator %s\n" % (self.__class__.__name__, self.version)
        
        str_net += "%s: %s\n" % ("ftr0", str(self.pre_gen_input_processing0))
        str_net += "%s: %s\n" % ("ftr0", str(self.pre_gen_input_processing1))
        str_net += "%s: %s\n" % ("ftr0", str(self.pre_gen_input_processing2))
        str_net += "%s: %s\n" % ("ftr1", str(self.pre_gen_ftr_extractor_1))
        str_net += "%s: %s\n" % ("ftr2", str(self.pre_gen_ftr_extractor_2))
        str_net += "%s: %s\n" % ("ftr3", str(self.pre_gen_ftr_extractor_3))
        str_net += "%s: %s\n" % ("syn2", str(self.pre_gen_synthesizer_2))
        str_net += "%s: %s\n" % ("syn1", str(self.pre_gen_synthesizer_1))
        str_net += "%s: %s\n" % ("syn0", str(self.pre_gen_synthesizer_0))
        str_net += "%s: %s\n" % ("syn_output", str(self.pre_gen_ftr_processing))
        
        str_net += "%s Second generator %s\n" % (self.__class__.__name__, self.version)
        
        str_net += "%s: %s\n" % ("ftr0", str(self.input_processing))
        str_net += "%s: %s\n" % ("ftr1", str(self.ftr_extractor_1))
        str_net += "%s: %s\n" % ("ftr2", str(self.ftr_extractor_2))
        str_net += "%s: %s\n" % ("ftr3", str(self.ftr_extractor_3))
        str_net += "%s: %s\n" % ("syn2", str(self.synthesizer_2))
        str_net += "%s: %s\n" % ("syn1", str(self.synthesizer_1))
        str_net += "%s: %s\n" % ("syn0", str(self.synthesizer_0))
        str_net += "%s: %s\n" % ("syn_output", str(self.ftr_processing))
        
        return str_net;
    
    # end toString

# end FrameInterpolationGenerator


class FrameInterpolationDiscriminator(nn.Module):
    
    def __init__(self, nfg):
        '''
        Constructor of FrameInterpolationDiscriminator
        
        :param nfg: feature map size
        '''
        super(FrameInterpolationDiscriminator, self).__init__()
        self.nfg = nfg  # the size of feature map
        self.c = 3  # output channel
        
        self._initialize_blocks()

    # end instructor
    
    def _initialize_blocks(self):
        self.models = nn.Sequential(
            nn.Conv2d(self.nfg, self.nfg * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.nfg * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(self.nfg * 2, self.nfg * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.nfg * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(self.nfg * 4, self.nfg * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.nfg * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(self.nfg * 8, self.nfg * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.nfg * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(self.nfg * 16, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            )
        
        self.input_processing = nn.Sequential(
            nn.Conv2d(self.c * 3, self.nfg, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Dropout2d(0.25),
            )
        
    # end _initialize_blocks
    
    def forward(self, frame1, frame2, frame3):
        input_padded = torch.cat((frame1, frame2, frame3), 1)
        input_processed = self.input_processing(input_padded)
        output_final = self.models(input_processed)
                
        return output_final;
    
    # end forward
