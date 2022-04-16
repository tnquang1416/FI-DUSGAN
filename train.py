'''
Created on Jul 27, 2021

@author: Quang Tran
Train version 3.x: down-up scale generator with optical flow-based pre-processing.
'''

import argparse
import os
import time
import torch

from torch import nn, optim, cuda
from torchvision.utils import save_image
from torch.optim import lr_scheduler

from nets import net
from loss import GDL, MS_SSIM
from utils import CONSTANT, utils, model_utils
from observers import checkpoint_observer, cpt_test_observer

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--patch_size", type=int, default=3, help="size of the patch of sample")
parser.add_argument("--input_size", type=int, default=128, help="size of the input of network")
parser.add_argument("--nfg", type=int, default=32, help="feature map size of networks")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lr_decay", type=int, default=20, help="adam: learning rate decay step")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="number of batches between image sampling")
parser.add_argument("--lambda_adv", type=float, default=0.0001, help="the default weight of adv Loss")
parser.add_argument("--lambda_px_loss", type=float, default=1.0, help="the default weight of L2 Loss")
parser.add_argument("--lambda_gdl", type=float, default=1.0, help="the default weight of GDL Loss")
parser.add_argument("--lambda_ms_ssim", type=float, default=1.0, help="the default weight of MS-SSIM Loss")
parser.add_argument("--path_gen", type=str, default=None, help="loaded generator for training")
parser.add_argument("--path_dis", type=str, default=None, help="loaded discriminator for training")
parser.add_argument("--path", type=str, default=None, help="training folder")

opt = parser.parse_args()
opt.debug_mode = (opt.debug_mode == "True" or opt.debug_mode == True)

# networks
net_gen = net.FrameInterpolationGenerator(nfg=opt.nfg)
net_dis = net.FFrameInterpolationDiscriminator(nfg=opt.nfg)

optimizer_G = optim.Adam(net_gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = optim.Adam(net_dis.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# loss functions
adversarial_loss = nn.BCEWithLogitsLoss().cuda()
l2_loss = nn.MSELoss().cuda()
gd_loss = GDL.GDL(cudaUsed=True).cuda()
ms_ssim = MS_SSIM.MS_SSIM_Loss().cuda()

# optimizers
optimizer_G = optim.Adam(net_gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = optim.Adam(net_dis.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

def load_generator(path=None):
    if path is None:
        net_gen.apply(model_utils._weights_init_normal)
        return None, 0, None, -1;
    
    model_state_dict, optim_state_dict, epoch, version, loss = model_utils.load_model(path)
    net_gen.load_state_dict(model_state_dict)
    
    return optim_state_dict, epoch, version, loss

# end load_generator


def load_discriminator(path=None):
    if path is None:
        net_dis.apply(model_utils._weights_init_normal)
        return None, 0, CONSTANT.MESS_NO_VERSION, -1;
    
    model_state_dict, optim_state_dict, epoch, version, loss = model_utils.load_model(path)
    net_dis.load_state_dict(model_state_dict)
    
    return optim_state_dict, epoch, version, loss

# end load_discriminator


def _train_interval(in_pres, in_lats, gt):
    # Adversarial ground truths
    valid = Tensor(gt.shape[0], 1, 1, 1).fill_(0.95)
    fake = Tensor(gt.shape[0], 1, 1, 1).fill_(0.1)
    valid.requires_grad_(False)
    fake.requires_grad_(False)

    _, gen_imgs = net_gen(in_pres, in_lats)  # generate output images
                
    # ---------------------
    #  Train Discriminator
    # ---------------------
    for param in net_dis.parameters():
        param.grad = None
                        
    # Calculate gradient for D
    gt_distingue = net_dis(in_pres, gt, in_lats)
    fake_distingue = net_dis(in_pres, gen_imgs.detach(), in_lats)
    real_loss = adversarial_loss(gt_distingue, valid)
    fake_loss = adversarial_loss(fake_distingue, fake)
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward()
                        
    optimizer_D.step()  # update D's weights

    # -----------------
    #  Train Generator
    # -----------------
    for param in net_gen.parameters():
        param.grad = None
    
    # Calculate gradient for G
    # Loss measures generator's ability to fool the discriminator and generate similar image to ground truth
    adv_loss = lambda_adv * adversarial_loss(net_dis(in_pres, gen_imgs, in_lats), valid)
    rec_loss = lambda_px_loss * l2_loss(gen_imgs, gt) + lambda_gdl * gd_loss(gen_imgs, gt) + lambda_ms_ssim * ms_ssim(gen_imgs, gt)
    g_loss = adv_loss + rec_loss
    
    g_loss.backward()
                        
    optimizer_G.step()  # update G's weights
    
    return gen_imgs, real_loss.item(), fake_loss.item(), g_loss.item(), adv_loss.item()

# end _train_interval


def _show_progress(gen_imgs, groundtruth, epoch, batch_index, total_batch, d_real_loss, d_fake_loss, g_loss, g_adv_loss):
    psnr = cal_psnr_tensor(gen_imgs.data[0].cpu(), groundtruth.data[0].cpu())
    log = ("%s: [Epoch %d] [Batch %d/%d] [D loss (real/fake): (%1.5f, %1.5f)] [G loss (adv_loss): %1.5f (%1.5f)] [PSNR: %1.2f]" 
                % (version, epoch, batch_index, total_batch, d_real_loss, d_fake_loss, g_loss, g_adv_loss, psnr))
             
    # Display result (input and output) after every sample_intervals
    batches_done = epoch * total_batch + batch_index
        
    if batches_done % (sample_interval * (epoch // 10 + 1)) == 0:
        save_image(gen_imgs.data[:16], path + "/train_%d.png" % (batches_done), nrow=4, normalize=True)
        print("Saved train_%d.png" % batches_done)
        print(log)
    elif batches_done % 1000 == 0:
        print(log)
    
    return "%d\t%d/%d\t%f\t%f\t%f" % (epoch, batch_index, total_batch, (d_real_loss + d_fake_loss) / 2, g_loss, psnr)

# end _show_progress


def train(epoch, dataloader, test_dataloader):
    cpt_observer = checkpoint_observer.CheckPointObserver(opt.path)
	cpt_testing_observer = cpt_test_observer.CheckPointTest_Observer(opt.path + "/cpt/", test_dataloader)
    scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=max(opt.lr_decay, opt.n_epochs // 10), gamma=0.1)
    scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=max(opt.lr_decay, opt.n_epochs // 10), gamma=0.1)
                    
    # loss functions
    g_loss = 0
    d_loss = 0
    
    utils.logging(opt.path, "[Epoch]\t[Batch]\t[Total_batch]\t[D loss]\t[G loss]\t[PSNR]")
    
    while (epoch < opt.n_epochs):
        # enable training mode
        net_gen.train()
        net_dis.train()
        temp_log = ""
        for i, imgs in enumerate(dataloader):
            in_pres = imgs[0].to('cuda')
            in_lats = imgs[2].to('cuda')
            gt = imgs[1].to('cuda')
            
            gen_imgs, real_loss, fake_loss, g_loss, adv_loss = _train_interval(in_pres, in_lats, gt, 1)
            d_loss = (real_loss + fake_loss) / 2
            
            # Show progress
            if i > 0: temp_log += "\n"
            temp_log = _show_progress(gen_imgs, gt, epoch, i, dataloader.__len__(), real_loss, fake_loss, g_loss, adv_loss)
            utils.logging(opt.path, temp_log)
        
        cpt_observer.notify(net_gen.state_dict(), net_dis.state_dict(), optimizer_G.state_dict(), optimizer_D.state_dict(), epoch + 1, g_loss, d_loss)
        epoch += 1
        if scheduler_G.get_lr()[0] > 10 ** -7:
            scheduler_G.step()
            scheduler_D.step()
		# end if
		
		cpt_testing_observer.notify(net_gen, epoch)
			
	# end while
    
    return epoch, g_loss, d_loss
# end train


def main():    
    if not cuda.is_available():
        print(CONSTANT.MESS_NO_CUDA);
        return;
		
	data_dir = "data/tri_trainlist.txt"
	val_dir = "data/tri_vallist.txt"
	path = opt.path

    print(opt)
        
    os.makedirs(opt.path, exist_ok=True);

	print("==============<Prepare models/>=============================")
	# init training parameters and information
	gen_optim_state, cur_gen_epoch, gen_loss = load_generator(path_gen)
	dis_optim_state, cur_dis_epoch, dis_loss = load_discriminator(path_dis)
	net_gen.to('cuda')
	net_dis.to('cuda')

	if gen_optim_state is not None:
		optimizer_G.load_state_dict(gen_optim_state)
		# copy tensor into GPU manually
		for state in optimizer_G.state.values():
			for k, v in state.items():
				if torch.is_tensor(v):
					state[k] = v.cuda()

	if dis_optim_state is not None:
		optimizer_D.load_state_dict(dis_optim_state)        
		# copy tensor into GPU manually
		for state in optimizer_D.state.values():
			for k, v in state.items():                
				if torch.is_tensor(v):
					state[k] = v.cuda()

	if cur_gen_epoch != cur_dis_epoch and not is_testing:
			print("%s (%d-%d)" % (MESS_CONFLICT_PROGRESS, cur_gen_epoch, cur_dis_epoch))

	print("==> Models are ready at %d epoch." % cur_gen_epoch)
    
    print("==============<Prepare dataset/>=============================")
    t1 = time.time()
    dataloader = general_utils.load_dataset_from_path_file(batch_size=opt.batch_size, txt_path=data_dir, is_testing=False, patch_size=opt.patch_size)
	test_dataloader = general_utils.load_dataset_from_path_file(batch_size=1, txt_path=val_dir, is_testing=True, patch_size=opt.patch_size)
    print("==> Takes total %1.4fs" % ((time.time() - t1)))
    
	print("==============<Training.../>====================================")
	os.makedirs(path, exist_ok=True);
	log = "Opts[n_epochs:%d, batch_size:%d, input_size:%d, nfg: %d lr:%f, path_gen:%s, path_dis:%s, data_dir: %s, val_dir:%s, test_dir:%s]" % (n_epochs, batch_size, input_size, nfg, lr, path_gen, path_dis, data_dir, val_dir, test_dir)
    
	logging(path, log, is_exist=False)
	t1 = time.time()

	epoch, g_loss, d_loss = train(cur_gen_epoch, dataloader)
	print("==> Takes total %1.2fmins" % ((time.time() - t1) / (60)))
	logging(path, "Training takes %1.2fmins" % ((time.time() - t1) / (60)), is_exist=True)
      
	model_utils.save_models(net_gen.state_dict(), net_dis.state_dict(), optimizer_G.state_dict(), optimizer_D.state_dict(), epoch, path, True, g_loss, d_loss)

    
main()

if __name__ == '__main__':
    pass
