import torch,os, cv2
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from model import *
from dataset import *
from utils_data import *
from utils_score import *
from utils_vis import *
from utils_loss_functions import *

import math,shutil,copy


def test(input_path, output_path, method_name='STRNN', saveFrames=float('inf'), use_cb=False, nb_gaussian=8,
         iosize=[480, 640, 60, 80], batch_size=2, time_dims=7):

    model = STRNN_final(ratio=16, pool_type='att', fusion_type=['channel_sr', 'feat_sum'], cnn_type='vgg16',
                        use_dcn=True, use_cb=False, use_bn=False, nb_gaussian=8, cnn_stride=16, out_stride=8,
                        iosize=[480, 640, 60, 80], time_dims=7, cat_type=[0, 1, 0, 1], pre_model_path='')

    model = model.cuda()
    model_path = './weights/strnn-vgg16-diem_final.pth'
    if os.path.exists(model_path):
        print("Load STRNN weights")
        model.load_state_dict(torch.load(model_path).state_dict())
    else:
        raise ValueError

    output_path = output_path + method_name + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    shape_r, shape_c, shape_r_out, shape_c_out = iosize
    if use_cb:
        cb_st = get_guasspriors('st', batch_size * time_dims, shape_r_out, shape_c_out, nb_gaussian)
        cb_dy = get_guasspriors('dy', batch_size * time_dims, shape_r_out, shape_c_out, nb_gaussian)
        x_cb = np.concatenate((cb_st,cb_dy),axis=-1).transpose((0, 3, 1, 2))
        x_cb = torch.tensor(x_cb).float()
    else:
        x_cb = torch.tensor([]).float()

    file_names = [f for f in os.listdir(input_path) if (f.endswith('.avi') or f.endswith('.AVI') or f.endswith('.mp4'))]
    file_names.sort()
    nb_videos_test = len(file_names)

    model.eval()
    with torch.no_grad():
        for idx_video in range(nb_videos_test):
            print("%d/%d   " % (idx_video + 1, nb_videos_test) + file_names[idx_video])

            ovideo_path = output_path + (file_names[idx_video])[:-4] + '.mat'
            if os.path.exists(ovideo_path):
                continue

            ivideo_path = input_path + file_names[idx_video]
            vidimgs, nframes, height, width = preprocess_videos(ivideo_path, shape_r, shape_c, saveFrames + time_dims - 1, mode='RGB', normalize=False)

            count_bs = int((nframes-1) / time_dims)
            isaveframes = count_bs * time_dims
            vidimgs = vidimgs[0:isaveframes + 1].transpose((0, 3, 1, 2))

            count_input = batch_size * time_dims
            pred_mat = np.zeros((isaveframes, height, width, 1),dtype=np.uint8)
            bs_steps = math.ceil(count_bs / batch_size)
            x_state = None
            for idx_bs in range(bs_steps):
                x_imgs = vidimgs[idx_bs * count_input:(idx_bs + 1) * count_input + 1]
                x_imgs = torch.tensor(x_imgs).float() / 255

                if use_cb and x_imgs.shape[0] -1 != count_input:
                    t_cb_st = get_guasspriors('st', x_imgs.shape[0] - 1, shape_r_out, shape_c_out, nb_gaussian)
                    t_cb_dy = get_guasspriors('dy', x_imgs.shape[0] - 1, shape_r_out, shape_c_out, nb_gaussian)
                    t_x_cb = np.concatenate((t_cb_st, t_cb_dy), axis=-1).transpose((0, 3, 1, 2))
                    x_cb_input = torch.tensor(t_x_cb).float()
                else:
                    x_cb_input = x_cb

                bs_out, out_state = model(x_imgs.cuda(), x_cb_input.cuda(), x_state)
                x_state = [[out_state[0].detach(), out_state[1].detach()]]

                bs_out = bs_out.data.cpu().numpy()
                bs_frames = bs_out.shape[0]
                for idx_pre in range(bs_frames):
                    isalmap = postprocess_predictions(bs_out[idx_pre,0,:,:], height, width)
                    pred_mat[idx_bs * batch_size * time_dims + idx_pre, :, :, 0] = np2mat(isalmap)

            iSaveFrame = min(isaveframes, saveFrames)
            pred_mat = pred_mat[0:iSaveFrame, :, :, :].transpose((1, 2, 3, 0))
            h5io.savemat(ovideo_path, {'salmap':pred_mat})


if __name__ == '__main__':

    DataSet = 'DIEM20'
    method_name = 'STRNN'
    batch_size = 2
    iosize = [480, 640, 60, 80]

    if os.name == 'nt':
        dataDir = 'E:/DataSet/'
    else:
        dataDir = '/home/kao/kao-ssd/DataSet'

    test_dataDir = dataDir + '/' + DataSet + '/'
    test_input_path = test_dataDir + 'Videos/'
    test_result_path = test_dataDir + 'Results/Results_STRNN/'
    test_output_path = test_result_path + 'Saliency/'

    if DataSet == 'DIEM20':
        saveFrames = 300
    else:
        saveFrames = float('inf')

    test(test_input_path, test_output_path, method_name=method_name, saveFrames=saveFrames, iosize=iosize, batch_size=batch_size)

    evalscores_vid(test_dataDir, test_result_path, DataSet=DataSet, MethodNames=[method_name])

    visual_vid(test_dataDir, test_result_path, DataSet=DataSet, MethodNames=[method_name], with_color=1, with_fix=1)

