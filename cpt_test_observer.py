'''
Created on Feb 9, 2021

@author: lab1323pc
'''
import numpy as np
import time
import torch
from torchvision.utils import save_image
import math

from utils import CONSTANT, utils, frame_utils


class CheckPointTest_Observer(object):
    '''
    classdocs
    '''

    def __init__(self, path, testing_dataloader):
        '''
        Constructor
        '''
        self.dataloader = testing_dataloader
        self.path = path
        
    def notify(self, net_gen, epoch, opt, do_partition=True, is_saving_output=False):
        return self._run_test(net_gen, epoch, do_partition, is_saving_output, opt)
        
    def _run_test(self, net_gen, epoch, do_partition, is_saving_output, opt):
        '''
        Run test on whole loaded dataset
        :param net_gen: generator instance
        :param epoch: #epoch
        '''
        psnr_list = []
        ssim_list = []
        time_list = []
        log = "Sample\tPSNR\tSSIM\n" if opt.is_testing else "";
        net_gen.eval()
        
        for i, imgs in enumerate(self.dataloader):
            input1 = imgs[0].to('cuda')
            input2 = imgs[2].to('cuda')
            gt = imgs[1].to('cuda')

            t1 = time.time()
            gen_frame = self._generate_full_frame(net_gen, input1, input2, do_partition)
            time_list.append(time.time() - t1)
            psnr, ssim = self._cal_metrics(gt, gen_frame)
            if opt.is_testing: log += "%d\t%1.2f\t%1.3f\n" % (i * opt.test_batch_size, np.average(np.array(psnr)), np.average(np.array(ssim)))

            psnr_list.extend(psnr)
            ssim_list.extend(ssim)
            
            if is_saving_output:
                samples = torch.cat((abs(input1 - input2), gt, gen_frame), 1)
                samples = samples.view(samples.shape[0], -1, opt.channels, samples.shape[2], samples.shape[3])
                save_image(samples[0], "%s/%05d.png" % (opt.path, i), padding=10, nrow=samples.shape[1])
        # end run test

        avgPSNR = np.average(np.array(psnr_list))
        avgSSIM = np.average(np.array(ssim_list))
        avgTime = np.average(np.array(time_list))
        
        mess = "%d (%d patches):\t%1.2f\t%1.3f\t%1.2f ms" % (epoch, len(self.dataloader.dataset), avgPSNR, avgSSIM, avgTime * 1000)
        print("==> Testing: %s" % mess)
        log += "%s" % mess
        utils.logging(self.path, log, is_exist=not opt.is_testing)
            
        return avgPSNR, avgSSIM

    # end run_test
    
    def _cal_metrics(self, gt_tensors, gen_tensors):
        if (gen_tensors.data[0].shape[1] != gt_tensors.data[0].shape[1]):
            print(CONSTANT.MESS_CONFLICT_DATA_SIZES)
            
        list_psnr = []
        list_ssim = []
            
        for i in range(gen_tensors.data.shape[0]):
            list_psnr.append(utils.cal_psnr_img_from_tensor(gen_tensors.data[i], gt_tensors.data[i]))
            list_ssim.append(utils.cal_ssim_tensor(gen_tensors.data[i].view(1, gen_tensors.shape[1], gen_tensors.shape[2], gen_tensors.shape[3]),
                                                    gt_tensors.data[i].view(1, gt_tensors.shape[1], gt_tensors.shape[2], gt_tensors.shape[3])))
                
        return list_psnr, list_ssim
    
    # end _cal_metrics

    def _generate_full_frame(self, net_gen, input1, input2, do_partition):
        return self._generate_full_frame_without_partition(net_gen, input1, input2)
    
    # end _generate_full_frame
    
    def _generate_full_frame_without_partition(self, net_gen, input1, input2, division=32):
        
        '''
        Generate a full frame (larger than training input 128x128) from given frames.
        :param input1: previous frame
        :param input2: latter frame
        :param division: division for model 's required input size calculation
        '''
        h = math.ceil(input1.shape[2]/division) * division
        w = math.ceil(input1.shape[3]/division) * division
        size = max(h, w)
        img1, pad_h, pad_w = frame_utils.pre_processing(input1, size)
        img2, _, _ = frame_utils.pre_processing(input2, size)
        
        _, gen = net_gen(img1, img2)
        gen = frame_utils.post_processing(gen, pad_h, pad_w)
        
        return gen
    
    # end _generate_full_frame_without_partition