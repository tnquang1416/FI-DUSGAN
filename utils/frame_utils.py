'''
Created on Jan 22, 2021

@author: Quang Tran
'''

import torch
import math

from torch.nn import functional as F


def get_no_row_blocks(v_frame, v_block, overlapped_pixels=0):
    return int(math.ceil((v_frame - v_block) / (v_block - overlapped_pixels))) + 1

# end get_no_row_blocks


def get_no_col_blocks(h_frame, h_block, overlapped_pixels=0):
    return int(math.ceil((h_frame - h_block) / (h_block - overlapped_pixels))) + 1

# end get_no_col_blocks


def concatenate_weight_overlapped_block(tensor1, tensor2, overlapped_pixels_range, is_vertical=True):
    if is_vertical:
        return _concat_vertical(tensor1, tensor2, overlapped_pixels_range)
    
    return _concat_horizontal(tensor1, tensor2, overlapped_pixels_range)
        
# end concatenate_weight_overlapped_block

        
def _concat_vertical(tensor1, tensor2, overlapped_pixels_range):
    if tensor1.shape[2] != tensor2.shape[2]:
        raise ValueError("The shape of input for concatenation is different.")
    
    if overlapped_pixels_range < 0:
        return _concat_vertical(tensor2, tensor1, overlapped_pixels_range * -1)
    
    rs = torch.ones((tensor1.shape[0], tensor1.shape[1], tensor1.shape[2], tensor1.shape[3] + tensor2.shape[3] - overlapped_pixels_range))
    
    rs[:, :, :, 0:tensor1.shape[3]] = tensor1
    rs[:, :, :, (tensor1.shape[3] - overlapped_pixels_range):] = tensor2
    
    for i in range(overlapped_pixels_range):
        w = i / overlapped_pixels_range
        rs[:, :, :, (tensor1.shape[3] - overlapped_pixels_range + i)] = (tensor1[:, :, :, (tensor1.shape[3] - overlapped_pixels_range + i)] * (1 - w) 
                                                                      +tensor2[:, :, :, i] * w)
    
    return rs

# end _concat_vertical


def _concat_horizontal(tensor1, tensor2, overlapped_pixels_range):
    if tensor1.shape[3] != tensor2.shape[3]:
        raise ValueError("The shape of input for concatenation is different.")
    
    if overlapped_pixels_range < 0:
        return _concat_horizontal(tensor2, tensor1, overlapped_pixels_range * -1)
    
    rs = torch.ones((tensor1.shape[0], tensor1.shape[1], tensor1.shape[2] + tensor2.shape[2] - overlapped_pixels_range, tensor1.shape[3]))
    
    rs[:, :, 0:tensor1.shape[2], :] = tensor1
    rs[:, :, (tensor1.shape[2] - overlapped_pixels_range):, :] = tensor2
    
    for i in range(overlapped_pixels_range):
        w = i / overlapped_pixels_range
        rs[:, :, (tensor1.shape[2] - overlapped_pixels_range + i), :] = (tensor1[:, :, (tensor1.shape[2] - overlapped_pixels_range + i), :] * (1 - w) 
                                                                      +tensor2[:, :, i, :] * w)
    
    return rs

# end _concat_horizontal


def concat_to_frame(tensor_list, no_cols, no_rows, overlapped_pixels_range):
    if (no_cols * no_rows != len(tensor_list)):
        raise ValueError("The quantity of tensors is not enough. Expected is %d but given %d." % (no_cols * no_rows, len(tensor_list)))
    
    if (len(tensor_list[0].shape) != 4):
        raise ValueError("Expected shape of 4 but get." % (len(tensor_list[0].shape)))
    
    row_tensors = []
    
    for i in range(no_rows):
        temp = tensor_list[i * no_cols]
        for j in range(no_cols - 1):
#             print("%d - %d" % (i * no_cols + j, i * no_cols + j + 1))
            temp = concatenate_weight_overlapped_block(temp.cpu(), tensor_list[i * no_cols + j + 1].cpu(), overlapped_pixels_range)
        
        row_tensors.append(temp)
    # end row concatenation
    
    rs = row_tensors[0]
    for i in range(1, no_rows):
        rs = concatenate_weight_overlapped_block(rs, row_tensors[i], overlapped_pixels_range, is_vertical=False)
        
    return rs

# end concat_to_frame


def pre_processing(frame_tensor, size):
    '''
    Apply padding on the input to make it fit the size of network
    :param frame_tensor:
    :param size:
    '''
    h = int(list(frame_tensor.size())[2])
    w = int(list(frame_tensor.size())[3])
        
    if (h < size):
        pad_h = size - h
        frame_tensor = F.pad(frame_tensor, (0, 0, 0, pad_h))
            
    if (w < size):
        pad_w = size - w
        frame_tensor = F.pad(frame_tensor, (0, pad_w, 0, 0))
    
    return frame_tensor, h, w

# end pre_processing


def post_processing(frame_tensor, h, w):
    '''
    Remove padded values if necessary
    :param frame_tensor:
    :param h:
    :param w:
    '''
    if h > 0:
        frame_tensor = frame_tensor[:, :, 0:h, :]
    if w > 0:
        frame_tensor = frame_tensor[:, :, :, 0:w]
            
    return frame_tensor

# end post_processing


def cal_additional_input_frame(frame1, frame2):
    rs = abs(frame1 - frame2)
    rs = (rs < 0.1)
    rs = rs * frame1
    
    return rs

# end cal_additional_input_frame
