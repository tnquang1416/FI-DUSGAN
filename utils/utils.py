'''
Created on Nov 18, 2020

@author: Quang Tran
'''

import numpy as np

from torch.utils.data.dataloader import DataLoader

from utils import calculator, dataset_handler, CONSTANT


def calculate_padding_conv(w_in, w_out, kernel_size, stride):
    '''
    Calculate padding parameter of convolutional layer
    w_out = (w_in-F+2P) / S + 1
    w_out: width of output
    w_in: width of input 
    '''
    return ((w_out - 1) * stride - w_in + kernel_size) // 2;
	

def cal_psnr_img_from_tensor(tensor1, tensor2):
    '''
    Calculate PSNR from two tensors of images
    :param tensor1: tensor
    :param tensor2: tensor
    '''
    img1 = transform_tensor_to_img(tensor1)
    img2 = transform_tensor_to_img(tensor2)
    diff = (np.array(img1) - np.array(img2))
    diff = diff ** 2

    if diff.sum().item() == 0:
        return float('inf')

    rmse = diff.mean().item()
    psnr = 20 * np.log10(255) - 10 * np.log10(rmse)

    return psnr

# end cal_psnr_tensor


def cal_psnr_tensor(img1, img2):
    '''
    Calculate PSNR from two tensors of images
    :param img1: tensor
    :param img2: tensor
    '''
    diff = (img1 - img2)
    diff = diff ** 2
        
    if diff.sum().item() == 0:
        return float('inf')

    rmse = diff.sum().item() / (img1.shape[0] * img1.shape[1] * img1.shape[2])
    psnr = 20 * np.log10(1) - 10 * np.log10(rmse)
        
    return psnr


def cal_ssim_tensor(X, Y,):
    return calculator.cal_ssim_tensor(X, Y, data_range=1.0).item()


def get_sub_name_from_path(file_name):
    '''
    Target: D:/gan_testing/data/cuhk/train\16.avi_244.jpg
    :param file_name: file path
    '''
    start = "\\"
    end = ".avi_"
    
    if (file_name.find(end) == -1):
        return "";
    
    temp = file_name[file_name.find(start) + 1:]
    
    if temp == -1:
        return "";
    
    return temp[:temp.find(end)]


def write_to_text_file(path, content, is_exist=True):
    '''
    create new file then write
    :param path:
    :param content:
    '''
    out = open(path, 'a') if is_exist else open(path, 'w')
    out.write(content)
    out.close();

    
def logging(path, content, is_exist=True):
    write_to_text_file("%s/%s" % (path, CONSTANT.LOG_FILE), content + "\n", is_exist)
    
# end logging

    
def load_dataset(input_size, batch_size, data_path, is_testing, dataset_ratio=100, patch_size=3):
    print("Loading dataset: %s" % data_path)
    dataset = dataset_handler.ImageDatasetLoader(input_size).loadImageDataset(data_path, is_testing=is_testing, ratio=dataset_ratio, patch_size=patch_size);
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=not is_testing, num_workers=0)
    print("Done loading dataset with %d patches." % (dataset.__len__()))
    
    return data_loader

# end load_dataset


def load_dataset_from_dir(batch_size, data_path, is_testing, patch_size):
    print("Loading dataset: %s" % data_path)
    dataset = dataset_handler.DBreader_frame_interpolation(db_dir=data_path, patch_size=patch_size)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=not is_testing, num_workers=0)
    print("Done loading dataset with %d patches." % (dataset.__len__()))
    
    return data_loader

# end load_dataset_from_dir


def load_dataset_from_path_file(batch_size, txt_path, is_testing, patch_size):
    print("Loading dataset: %s" % txt_path)
    dataset = dataset_handler.DBreader_frame_interpolation(path_file=txt_path, patch_size=patch_size)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=not is_testing, num_workers=0)
    print("Done loading dataset with %d patches." % (dataset.__len__()))
    
    return data_loader

# end load_dataset_from_path_file


def cal_correlation_coefficient(tensor1, tensor2):
    '''
    Calculate correlation coefficient between two tensor
    :param tensor1:
    :param tensor2:
    '''
    if tensor1.shape != tensor2.shape or tensor1.shape.__len__() != 3:
        print("%s %s" % (CONSTANT.MESS_CONFLICT_SHAPE, str(tensor1.shape)))
        return 0
    
    hist1, _ = np.histogram(tensor1.cpu())
    hist2, _ = np.histogram(tensor2.cpu())
    
    return np.corrcoef([hist1, hist2])[1][0]

# end cal_correlation_coefficient
