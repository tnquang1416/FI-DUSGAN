'''
Created on Nov 20, 2020

@author: Quang Tran
'''
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from os import listdir
from os.path import join, isdir
from PIL import Image

import numpy as np
import os
import glob
import random as r
from warnings import warn

import utils.utils as utils


class DBreader_frame_interpolation(Dataset):
    """
    DBreader reads all triplet set of frames in a directory or from input tensor list
    """

    def __init__(self, db_dir=None, resize=None, tensor_list=None, path_list=None, path_file=None, patch_size=3):
        self.patch_size = patch_size
        if db_dir is not None:
            self._load_from_dir(db_dir, resize)
            self.mode = 1
        elif tensor_list is not None:
            self._load_from_tensor_list(tensor_list)
            self.mode = 2
        elif path_list is not None:
            self._load_from_dir(path_list, resize)
            self.mode = 3
        elif path_file is not None:
            self._load_from_file_path(path_file, resize)
            
    # end __init__
            
    def _load_from_tensor_list(self, tensor_list):
        '''
        Load from numpy tensor list (no_triplets, triplets_index, c, w, h)
        :param tensor_list:
        '''
        self.imgs_list = tensor_list
        
    def _load_from_dir(self, db_dir, resize=None):
        '''
        DBreader reads all triplet set of frames in a directory.
        '''
        files_list = np.array([(db_dir + '/' + f) for f in listdir(db_dir) if isdir(join(db_dir, f))])

        self._load_from_list_path(files_list, resize)
    
    # end _load_from_dir
    
    def _load_from_list_path(self, path_list, resize=None):
        if resize is not None:
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        self.files_list = path_list
        self.imgs_list = []
        flag = False
        try:
            Image.open("%s/frame%d.png" % (self.files_list[0], 0))
        except:
            flag = True

        for path in self.files_list:
            data = []
            if flag:
                for i in range(self.patch_size): data.append(self.transform(Image.open("%s/im%d.png" % (path, i + 1))))
            else:
                for i in range(self.patch_size): data.append(self.transform(Image.open("%s/frame%d.png" % (path, i))))
            self.imgs_list.append(data)
    
    # end _load_from_list_path
    
    def _load_from_file_path(self, txt_file_path, resize=None):
        lines = open(txt_file_path, "r")
        dir_path = txt_file_path[:txt_file_path.find("/t")]
        file_list = []
        for line in lines:
            if len(line.rstrip()) < 1: continue
            file_list.append("%s/%s/%s" % (dir_path, "sequences", line.strip()))
            
        return self._load_from_list_path(file_list, resize)

    # end _load_from_file_path
    
    def __getitem__(self, index):
        frame0 = self.imgs_list[index][0]
        frame1 = self.imgs_list[index][1]
        frame2 = self.imgs_list[index][2]
        
        if self.patch_size == 3:
            return frame0, frame1, frame2
        
        frame3 = self.imgs_list[index][3]
        frame4 = self.imgs_list[index][4]

        return frame0, frame1, frame2, frame3, frame4

    def __len__(self):
        return len(self.files_list)


# Custom crop image transformation
class CropTransform:

    def __init__(self, h, w, size):
        self.h_coor = h;
        self.w_coor = w;
        self.size = size;

    def __call__(self, img):
        return TF.crop(img, self.h_coor, self.w_coor, self.size, self.size);


# Custom image loader including load, transform, crop and group
class ImageDatasetLoader:
    
    def __init__(self, input_size):
        self.input_size = input_size;
        self.dataset = np.array([]);
    
    def loadImageDataset(self, path, is_testing=False, ratio=100, patch_size=3):
        '''
        Load all images from dataset
        :param path:
        '''
        if is_testing:
            self.input_size = -1
        if (not os.path.isdir(path)):
            raise FileNotFoundError("ImageDatasetLoader: Cannot locate %s" % (path))

        dataset = self._load_three_frame_patch(path, ratio) if patch_size == 3 else self._load_five_frame_patch(path, ratio)
                
        if len(dataset) == 0:
            warn("No data loaded.")
            return dataset
        
        self.dataset = np.asarray(dataset)
        
        return self.dataset;
    
    # end loadImageDataset
    
    def _load_three_frame_patch(self, path, ratio):
        imgs = []
        dataset = [];
        files = glob.glob(path + '/*.jpg')
        files.sort()
        current_progress = 0
        
        pre_file_name_1 = ""
        pre_file_name_2 = ""

        for file_name in files:
            progress = 100.0 * len(imgs) / len(files)
            if (int(progress) - current_progress >= 10) :
                current_progress = int(progress)
                print(str(int(current_progress)) + "%")
            
            img = Image.open(file_name)
            imgs.append(img.copy())
            
            if progress > ratio:
                img.load()
                break
        
            img_size = min(img.size)
            temp_name = utils.get_sub_name_from_path(file_name)
            
            if pre_file_name_1 == "":
                pre_file_name_1 = temp_name
            elif pre_file_name_2 == "":
                pre_file_name_2 = temp_name
            else:
                pre_file_name_0 = pre_file_name_1
                pre_file_name_1 = pre_file_name_2
                pre_file_name_2 = temp_name
                
                if (pre_file_name_1 != pre_file_name_2 or pre_file_name_0 != pre_file_name_2):
                    continue;
                h = r.randint(0, max(0, img_size - self.input_size - 1))
                w = r.randint(0, max(0, img_size - self.input_size - 1))
                data = []
                data.append(self._transformImageForNoNoiseGenerator(imgs[len(imgs) - 3], h, w));
                data.append(self._transformImageForNoNoiseGenerator(imgs[len(imgs) - 2], h, w));
                data.append(self._transformImageForNoNoiseGenerator(imgs[len(imgs) - 1], h, w));
                dataset.append(np.asarray(data))
                
            img.load()
        
        return dataset
    
    # end _load_three_frame_patch
    
    def _load_five_frame_patch(self, path, ratio, is_augmented=False):
        imgs = []
        dataset = [];
        files = glob.glob(path + '/*.jpg')
        files.sort()
        current_progress = 0
        buffer = []

        for file_name in files:
            progress = 100.0 * len(imgs) / len(files)
            if (int(progress) - current_progress >= 10) :
                current_progress = int(progress)
                print(str(int(current_progress)) + "%")
            
            img = Image.open(file_name)
            imgs.append(img.copy())
            
            if progress > ratio:
                img.load()
                break
        
            img_size = min(img.size)
            temp_name = utils.get_sub_name_from_path(file_name)
            
            if buffer.count(temp_name) == 0:
                buffer.clear()
                buffer.append(temp_name)
            elif len(buffer) < 5:
                buffer.append(temp_name)
            else:
                h = r.randint(0, max(0, img_size - self.input_size - 1))
                w = r.randint(0, max(0, img_size - self.input_size - 1))
                data = []
                is_vflip = len(imgs) % 3 == 0
                is_hflip = len(imgs) % 5 == 0
                if is_augmented:
                    data.append(self._augment_transform_training_sample(imgs[len(imgs) - 5], h, w, self.crop_size, is_vflip, is_hflip));
                    data.append(self._augment_transform_training_sample(imgs[len(imgs) - 4], h, w, self.crop_size, is_vflip, is_hflip));
                    data.append(self._augment_transform_training_sample(imgs[len(imgs) - 3], h, w, self.crop_size, is_vflip, is_hflip));
                    data.append(self._augment_transform_training_sample(imgs[len(imgs) - 2], h, w, self.crop_size, is_vflip, is_hflip));
                    data.append(self._augment_transform_training_sample(imgs[len(imgs) - 1], h, w, self.crop_size, is_vflip, is_hflip));
                else:
                    data.append(self._transformImageForNoNoiseGenerator(imgs[len(imgs) - 5], h, w));
                    data.append(self._transformImageForNoNoiseGenerator(imgs[len(imgs) - 4], h, w));
                    data.append(self._transformImageForNoNoiseGenerator(imgs[len(imgs) - 3], h, w));
                    data.append(self._transformImageForNoNoiseGenerator(imgs[len(imgs) - 2], h, w));
                    data.append(self._transformImageForNoNoiseGenerator(imgs[len(imgs) - 1], h, w));
                dataset.append(np.asarray(data))
                
            img.load()
        
        return dataset
    
    # end _load_five_frame_patch
    
    def _augment_transform_training_sample(self, img, crop_h, crop_w, size, v_flip=False, h_flip=False, affine_angle=0):
        trans_tensor = transforms.ToTensor()
        trans_crop = CropTransform(crop_h, crop_w, size)
        
        if v_flip and h_flip:
            return trans_tensor(TF.vflip(TF.hflip(trans_crop(TF.affine(img, angle=affine_angle, translate=(0, 0), scale=1.0, shear=0)))))
        if v_flip:
            return trans_tensor(TF.vflip(trans_crop(TF.affine(img, angle=affine_angle, translate=(0, 0), scale=1.0, shear=0))))
        if h_flip:
            return trans_tensor(TF.hflip(trans_crop(TF.affine(img, angle=affine_angle, translate=(0, 0), scale=1.0, shear=0))))
        
        return trans_tensor(trans_crop(TF.affine(img, angle=affine_angle, translate=(0, 0), scale=1.0, shear=0)))
    
    # end _augment_transform_training_sample
    
    def _transformImageForNoNoiseGenerator(self, img, h, w):
        size = min(img.size)
        transTensor = transforms.ToTensor();        
#         transResize = transforms.Resize(size)
        
        if (self.input_size == -1) :
            return transTensor(img).numpy();
        else:
            transCrop = CropTransform(h, w, min(self.input_size, size));
        
        return transTensor(transCrop(img)).numpy();
