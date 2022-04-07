'''
Created on Nov 23, 2020

@author: Quang Tran
'''

import torch
from torch import nn
import os
from utils import CONSTANT


def _weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        #TODO: m.weight is None in depth-net
        if m.weight is None:
            print(classname)
        else:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

# end _weights_init_normal


def load_model(path, is_testing=False):
    '''
    Load network's state data (respecting to saved information)
    :param path: file path
        '''
    checkpoint = torch.load(path)
    start_epoch = checkpoint[CONSTANT.LABEL_EPOCH]
    model_state_dict = checkpoint[CONSTANT.LABEL_MODEL_STATE_DICT]
    try: 
        version = checkpoint[CONSTANT.LABEL_VERSION]
    except:
        version = CONSTANT.MESS_NO_VERSION
    if is_testing:
        return model_state_dict, start_epoch
    
    optim_state_dict = checkpoint[CONSTANT.LABEL_OPTIMIZER_STATE_DICT]
    loss = checkpoint[CONSTANT.LABEL_LOSS]
    
    return model_state_dict, optim_state_dict, start_epoch, loss

# end _load_model


def save_models(gen_state_dict, dis_state_dict, gen_opt_state_dict, dis_opt_state_dict, epoch, outpath, is_rewrite=True, loss_gen=-1, loss_dis=-1):
    '''
    Save model into file which contains state's information
    :param epoch: last epochth train
    :param outpath: saved directory path
    '''
    save_generator(gen_state_dict, gen_opt_state_dict, epoch, outpath, loss_gen, is_rewrite)
    save_discriminator(dis_state_dict, dis_opt_state_dict, epoch, outpath, loss_dis, is_rewrite)

# end save_models


def save_generator(state_dict, opt_state_dict, epoch, outpath, loss=-1, is_rewrite=True):
    os.makedirs(outpath, exist_ok=True)
    os.makedirs("%s/cpt" % outpath, exist_ok=True)
    path = "%s/net_gen.pt" % (outpath) if is_rewrite else "%s/cpt/net_gen_%d.pt" % (outpath, epoch)
    
    _save_model(state_dict, opt_state_dict, epoch, path, loss)
    
# end save_generator


def save_discriminator(state_dict, opt_state_dict, epoch, outpath, loss=-1, is_rewrite=True):
    os.makedirs(outpath, exist_ok=True)
    os.makedirs("%s/cpt" % outpath, exist_ok=True)
    path = "%s/net_dis.pt" % (outpath) if is_rewrite else "%s/cpt/net_dis_%d.pt" % (outpath, epoch)
    
    _save_model(state_dict, opt_state_dict, epoch, path, loss)
    
# end save_discriminator


def _save_model(net_state_dict, opt_state_dict, epoch, outpath, loss):
    '''
    Save model into file which contains state's information
    :param net_state_dict:
    :param opt_state_dict:
    :param epoch: last epochth train
    :param outpath: saved path
    :param loss:
    '''
        
    checkpoint = {CONSTANT.LABEL_EPOCH: epoch,
                 CONSTANT.LABEL_MODEL_STATE_DICT: net_state_dict,
                 CONSTANT.LABEL_OPTIMIZER_STATE_DICT: opt_state_dict,
                 CONSTANT.LABEL_LOSS: loss,
                 }
        
    torch.save(checkpoint, outpath);

# end save_models
