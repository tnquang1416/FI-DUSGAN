'''
Created on Nov 27, 2020

@author: Quang Tran
'''

import threading

from utils import model_utils


class CheckPointObserver(object):
    '''
    The observer for saving checkpoint of networks
    '''

    def __init__(self, path, version=None):
        '''
        Constructor
        '''
        self.path = path
        self.version = version if version is not None else ""
        
    def notify(self, net_gen, net_dis, optim_gen, optim_dis, epoch, loss_gen, loss_dis):
        thread = CheckPointSaverThread(net_gen, net_dis, optim_gen, optim_dis, epoch, self.path, self.version, loss_gen, loss_dis)
        thread.start()
        
    # end notify

        
class CheckPointSaverThread(threading.Thread):
    
    def __init__(self, net_gen, net_dis, optim_gen, optim_dis, epoch, path, version, loss_gen, loss_dis):
        super(CheckPointSaverThread, self).__init__()
        self.net_gen = net_gen
        self.net_dis = net_dis
        self.optim_gen = optim_gen
        self.optim_dis = optim_dis
        self.epoch = epoch
        self.path = path
        self.version = version
        self.loss_gen = loss_gen
        self.loss_dis = loss_dis
        
    # end construction
    
    def run(self):
        model_utils.save_models(self.net_gen, self.net_dis, self.optim_gen, self.optim_dis, self.epoch, self.path, False, self.loss_gen, self.loss_dis)

    # end run
