import torch, os, cv2
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math, shutil, copy


from model import STRNN_final


def getModelSize(model):
    param_size = 0
    param_sum = 0
    param_trainable = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()

        if param.requires_grad:
            param_trainable += param.nelement() * param.element_size()

    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()

    param_mb = param_size / 1024 / 1024
    buffer_mb = buffer_size / 1024 / 1024
    all_mb = (param_size + buffer_size) / 1024 / 1024

    # train_p_mb = param_trainable / 1024 / 1024

    print('param  size：{:.2f} MB'.format(param_mb))
    print('buffer size：{:.2f} MB'.format(buffer_mb))
    print('total  size：{:.2f} MB'.format(all_mb))

    # print('trainable params size：{:.2f} MB'.format(train_p_mb))

    return param_mb#,buffer_mb,all_mb


if __name__ == '__main__':

    other_size = 0

    model = STRNN_final()
    print('\nmodel')
    model_size = getModelSize(model)

    print('\nSt-Net')
    st_size = getModelSize(model.feat_sm)

    print('\nOF-Net')
    of_size = getModelSize(model.feat_of)

    print('\nSR-Fu')
    srfu_size = getModelSize(model.att_channel)

    print('\nfeat_fu')
    other_size += getModelSize(model.feat_fu)

    print('\nAA-LSTM')
    aalstm_size = getModelSize(model.att_lstm)

    print('\nconv_out')
    other_size +=getModelSize(model.conv_out)

    print('\n\n-----Params Size-----')
    print('Total STRNN: %.2f MB' % model_size)
    print('St-Net: %.2f MB' % st_size)
    print('OF-Net: %.2f MB' % of_size)
    print('SR-Fu: %.2f MB' % srfu_size)
    print('AA-LSTM: %.2f MB' % aalstm_size)
    print('Other: %.2f MB' % other_size)
    print('diff: %.2f MB' % (model_size-st_size-of_size-srfu_size-aalstm_size-other_size))

    print('\nTotal STRNN (param+buffer)): 361.07 MB')
    print('Saved STRNN model file size: 361.26 MB')

    print('done')
