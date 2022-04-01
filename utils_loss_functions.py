from __future__ import print_function

import torch
import torch.nn as nn
import math
EPS = 2.2204e-16

def get_sum(input):
    size_h, size_w = input.shape[2:]
    v_sum = torch.sum(input, (2,3), keepdim=True)
    return v_sum.repeat(1, 1, size_h, size_w)

def get_max(input):
    size_h, size_w = input.shape[2:]
    v_max = torch.max(torch.max(input, 2, keepdim=True)[0], 3, keepdim=True)[0]
    return v_max.repeat(1, 1, size_h, size_w)

def get_min(input):
    size_h, size_w = input.shape[2:]
    v_max = torch.min(torch.min(input, 2, keepdim=True)[0], 3, keepdim=True)[0]
    return v_max.repeat(1, 1, size_h, size_w)

def get_mean(input):
    size_h, size_w = input.shape[2:]
    v_mean = torch.mean(input, (2,3), keepdim=True)
    return v_mean.repeat(1, 1, size_h, size_w)

def get_std(input):
    size_h, size_w = input.shape[2:]
    v_mean = torch.mean(input, (2,3), keepdim=True)
    tmp = torch.sum((input-v_mean)**2,(2,3),keepdim=True) / (size_h*size_w-1)

    return torch.sqrt(tmp).repeat(1,1, size_h, size_w)


def loss_fu(y_pred,y_true):

    kl_value  = metric_kl(y_pred,y_true)
    cc_value  = metric_cc(y_pred, y_true)
    nss_value = metric_nss(y_pred, y_true)
    loss_value = 10 * kl_value - 2 * cc_value - nss_value

    return torch.mean(loss_value,0)

def loss_fu_dy(y_pred,y_true):
    B, D, C, H, W = y_pred.size()
    y_pred = torch.reshape(y_pred, (B * D, C, H, W))
    y_true = torch.reshape(y_true, (B * D, 2, H, W))

    kl_value  = metric_kl(y_pred,y_true)
    cc_value  = metric_cc(y_pred, y_true)
    nss_value = metric_nss(y_pred, y_true)
    loss_value = 10 * kl_value - 2 * cc_value - nss_value

    return torch.mean(loss_value,0)

def metric_kl(y_pred,y_true):
    y_true = y_true[:, 0:1, :, :]
    y_true /= (get_sum(y_true) + EPS)
    y_pred /= (get_sum(y_pred) + EPS)

    return torch.mean(torch.sum(y_true * torch.log((y_true / (y_pred + EPS)) + EPS), (2,3)),0)

def metric_cc(y_pred, y_true):
    y_true = y_true[:, 0:1, :, :]
    y_true = (y_true - get_mean(y_true)) / (get_std(y_true) + EPS)
    y_pred = (y_pred - get_mean(y_pred)) / (get_std(y_pred) + EPS)

    y_true = y_true - get_mean(y_true)
    y_pred = y_pred - get_mean(y_pred)
    r1 = torch.sum(y_true * y_pred,(2,3))
    r2 = torch.sqrt(torch.sum(y_pred*y_pred,(2,3))*torch.sum(y_true*y_true,(2,3)))
    return torch.mean(r1 / (r2 +EPS) ,0)

    # size_h, size_w = y_pred.shape[2:]
    # y_true /= (get_sum(y_true) + EPS)
    # y_pred /= (get_sum(y_pred) + EPS)
    #
    # N = size_h * size_w
    # sum_prod = torch.sum(y_true * y_pred, dim=(2,3))
    # sum_x = torch.sum(y_true, dim=(2,3))
    # sum_y = torch.sum(y_pred, dim=(2,3))
    # sum_x_square = torch.sum(y_true*y_true, dim=(2,3))
    # sum_y_square = torch.sum(y_pred*y_pred, dim=(2,3))
    #
    # num = sum_prod - ((sum_x * sum_y) / N)
    # den = torch.sqrt((sum_x_square - sum_x*sum_x / N) * (sum_y_square - sum_y*sum_y / N))
    # return torch.mean(num / den,0)

def metric_nss(y_pred, y_true):
    y_true = y_true[:, 1:2, :, :]
    y_pred = (y_pred - get_mean(y_pred)) / (get_std(y_pred)+ EPS)

    return torch.mean(torch.sum(y_true * y_pred, dim=(2,3)) / (torch.sum(y_true, dim=(2,3))+EPS),0)

def metric_sim(y_pred, y_true):
    y_true = y_true[:, 0:1, :, :]
    y_true = (y_true - get_min(y_true)) / (get_max(y_true) - get_min(y_true) + EPS)
    y_pred = (y_pred - get_min(y_pred)) / (get_max(y_pred) - get_min(y_pred) + EPS)

    y_true /= (get_sum(y_true) + EPS)
    y_pred /= (get_sum(y_pred) + EPS)

    diff = torch.min(y_true,y_pred)
    score = torch.sum(diff,dim=(2,3))

    return torch.mean(score,0)

def loss_ml(y_pred, y_true):
    y_true = y_true[:, 0:1, :, :]

    y_pred /= (get_max(y_pred) + EPS)
    return torch.mean((y_pred - y_true)*(y_pred - y_true) / (1 - y_true + 0.1))


###################################################################
# For PWC loss
###################################################################
def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

def loss_smooth(output, target):
    avg_target = nn.functional.avg_pool2d(target,kernel_size=3, stride=1, padding=1,count_include_pad=False)
    lossvalue = torch.abs(output - avg_target).mean()
    return lossvalue

class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
    def forward(self, output, target):
        avg_target = nn.functional.avg_pool2d(target,kernel_size=3, stride=1, padding=1,count_include_pad=False)
        lossvalue = torch.abs(output - avg_target).mean()
        return lossvalue

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class MultiScale(nn.Module):
    def __init__(self, args, startScale = 4, numScales = 5, l_weight= 0.32, norm= 'L1'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.args = args
        self.l_type = norm
        self.div_flow = 0.05
        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1':
            self.loss = L1()
        else:
            self.loss = L2()

        self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE'],

    def forward(self, output, target):
        lossvalue = 0
        epevalue = 0

        if type(output) is tuple:
            target = self.div_flow * target
            for i, output_ in enumerate(output):
                target_ = self.multiScales[i](target)
                epevalue += self.loss_weights[i]*EPE(output_, target_)
                lossvalue += self.loss_weights[i]*self.loss(output_, target_)
            return [lossvalue, epevalue]
        else:
            epevalue += EPE(output, target)
            lossvalue += self.loss(output, target)
            return  [lossvalue, epevalue]







# if __name__ == '__main__':
#
#
#     # shape_r = 480
#     # shape_c = 640
#     # shape_r_out = 480
#     shape_c_out = 640
#
#     import cv2, os, scipy.io
#     import numpy as np
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
#     def read_maps(paths):
#         ims = np.zeros((len(paths), 1, shape_r, shape_c), dtype=np.float)
#
#         for i, path in enumerate(paths):
#             image = cv2.imread(path, -1)
#             # image = cv2.resize(image, (shape_c, shape_r))
#             ims[i, 0, :, :] = image
#
#         return ims
#
#     def read_fixmaps(paths):
#         ims = np.zeros((len(paths), 1, shape_r, shape_c), dtype=np.float)
#
#         for i, path in enumerate(paths):
#             fix_map = scipy.io.loadmat(path)["I"]
#             # fix_map = cv2.resize(fix_map, (shape_c, shape_r))
#             ims[i, 0, :, :] = fix_map
#
#         return ims
#
#     data_dir = 'E:/IIP_Saliency_Dataset/DataSet/salicon-15/val-16/'
#     maps_train_path = data_dir + 'maps/'
#     fixs_train_path = data_dir + 'fixations/maps/'
#     sals_train_path = data_dir + 'salmaps/'
#
#     maps_path = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith('.png')]
#     fixs_path = [fixs_train_path + f for f in os.listdir(fixs_train_path) if f.endswith('.mat')]
#     sals_path = [sals_train_path + f for f in os.listdir(sals_train_path) if f.endswith('.png')]
#
#     maps_path.sort()
#     fixs_path.sort()
#     sals_path.sort()
#
#     maps = read_maps(maps_path)
#     sals = read_maps(sals_path)
#     fixs = read_fixmaps(fixs_path)
#
#     y_pred = torch.tensor(sals).float()
#     y_true =  torch.tensor(np.concatenate((maps,fixs),axis=1)).float()
#
#     kl_value  = metric_kl(y_pred,y_true)
#     cc_value  = metric_cc(y_pred, y_true)
#     nss_value = metric_nss(y_pred, y_true)
#     sim_value = metric_sim(y_pred, y_true)
#
#     loss_v1 = loss_fu(y_pred,y_true)
