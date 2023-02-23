'''
Created on Jul 27, 2021

@author: Quang Tran
'''

import argparse
import os
import time
import torch

from torch import cuda

from nets import net
from utils import CONSTANT, model_utils, utils
from observers import cpt_test_observer

parser = argparse.ArgumentParser()
parser.add_argument("--test_batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--patch_size", type=int, default=3, help="size of the patch of sample")
parser.add_argument("--input_size", type=int, default=128, help="size of the input of network")
parser.add_argument("--nfg", type=int, default=32, help="feature map size of networks")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--path_gen", type=str, default="pre-trained/net_gen.pt", help="loaded generator for training")
parser.add_argument("--path", type=str, default="output_test", help="output folder")
parser.add_argument("--is_testing", action='store_false')

opt = parser.parse_args()

assert opt.is_testing

# networks
net_gen = net.FrameInterpolationGenerator(nfg=opt.nfg)
# net_gen.to('cuda')

def load_generator(path):
    assert os.path.isfile(path), "Expected a file path."
    
    model_state_dict, optim_state_dict, epoch, loss = model_utils.load_model(path)
    net_gen.load_state_dict(model_state_dict)
    
    return optim_state_dict, epoch, loss

# end load_generator

def main():    
    if not cuda.is_available():
        print(CONSTANT.MESS_NO_CUDA)
        return
		
    test_dir = "data/tri_testlist.txt"
    path = opt.path

    print(opt)
        
    os.makedirs(opt.path, exist_ok=True);
    print("==============<Prepare models/>=============================")
    # init training parameters and information
    gen_optim_state, cur_gen_epoch, gen_loss = load_generator(opt.path_gen)
    
    print("==============<Prepare dataset/>=============================")
    t1 = time.time()
    test_dataloader = utils.load_dataset_from_path_file(batch_size=1, txt_path=test_dir, is_testing=True, patch_size=opt.patch_size)
    print("==> Takes total %1.4fs" % ((time.time() - t1)))
    
    print("==============<Testing.../>====================================")
    cpt_testing_observer = cpt_test_observer.CheckPointTest_Observer(opt.path, test_dataloader)
    psnr, ssim = cpt_testing_observer.notify(net_gen, cur_gen_epoch, opt, False, False)
	
    print("Done.")
    print("--> Avg PSNR = %1.2f" % psnr)
    print("--> Avg SSIM = %1.3f" % ssim)
    
main()
