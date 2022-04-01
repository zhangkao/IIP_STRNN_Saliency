import torch,os, cv2
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math,shutil,copy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model import *
from dataset import *
from utils_data import *
from utils_score import *
from utils_vis import *
from utils_loss_functions import *


def train(method_name = 'strnn', batch_size=4, epochs=20, cnn_type='vgg16', use_dcn=True, use_cb=False, use_bn=False, nb_gaussian=8,
          cnn_stride=16, out_stride=8, iosize=[480, 640, 60, 80], time_dims=7, cat_type=[0, 1, 0, 1],
          pre_model_path=''):

    tmdir = saveModelDir + method_name
    save_model_path = tmdir + '/' + method_name + '_'
    if not os.path.exists(tmdir):
        os.makedirs(tmdir)

    shape_r, shape_c, shape_r_out, shape_c_out = iosize
    #################################################################
    # Build the model
    #################################################################
    print("Build STRNN Model: " + method_name)
    model = STRNN_final(ratio=16, pool_type='att', fusion_type=['channel_sr', 'feat_sum'], cnn_type=cnn_type,
                        use_dcn=use_dcn, use_cb=use_cb, use_bn=use_bn, nb_gaussian=nb_gaussian, cnn_stride=cnn_stride,
                        out_stride=out_stride, iosize=iosize, time_dims=time_dims, cat_type=cat_type,
                        pre_model_path=pre_model_path)
    model = model.cuda()

    # When fine-tuning the model, you can fix some parameters to improve the training speed
    # for p in model.feat_sm.parameters():
    #     p.requires_grad = False
    # for p in model.feat_of.parameters():
    #     p.requires_grad = False

    criterion = loss_fu
    # When fine-tuning the model, it is recommended to use a smaller learning rate, like lr=1e-5, weight_decay=0.00001
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad == True], lr=1e-4, betas=(0.9, 0.999), weight_decay=0.00005)

    #################################################################
    # Train the model
    #################################################################
    print("Training STRNN Model")
    min_val_loss = 10000
    num_patience = 0
    if IS_EARLY_STOP:
        max_patience = Max_patience
    else:
        max_patience = epochs + 1

    if use_cb:
        cb_st = get_guasspriors('st', batch_size * time_dims, shape_r_out, shape_c_out, nb_gaussian)
        cb_dy = get_guasspriors('dy', batch_size * time_dims, shape_r_out, shape_c_out, nb_gaussian)
        x_cb = np.concatenate((cb_st,cb_dy),axis=-1).transpose((0, 3, 1, 2))
        x_cb = torch.tensor(x_cb).float()
    else:
        x_cb = torch.tensor([]).float()

    for epoch in range(epochs):
        print("\nEpochs: %d / %d " % (epoch + 1, epochs))
        for phase in ['train', 'val']:
            num_step = 0
            run_loss = 0.0
            if phase == 'train':
                model.train()
                shuffle = True
                Max_TrainFrame = Train_Max_TrainFrame
            else:
                model.eval()
                shuffle = False
                Max_TrainFrame = Val_Max_TrainFrame

            videos_list, vidmaps_list, vidfixs_list = read_video_list(train_dataDir, phase, shuffle=shuffle)

            for idx_video in range(len(videos_list)):
                print("Videos: %d / %d, %s with data from: %s" % (idx_video + 1, len(videos_list), phase.upper(), videos_list[idx_video]))

                vidmaps = preprocess_vidmaps(vidmaps_list[idx_video], shape_r_out, shape_c_out, Max_TrainFrame)
                vidfixs = preprocess_vidfixs(vidfixs_list[idx_video], shape_r_out, shape_c_out, Max_TrainFrame)
                vidimgs, nframes, height, width = preprocess_videos(videos_list[idx_video], shape_r, shape_c, Max_TrainFrame, mode='RGB' ,normalize=False)
                nframes = min(min(vidfixs.shape[0], vidmaps.shape[0]), nframes) - 1

                count_bs = nframes // time_dims
                trainFrames = count_bs * time_dims
                vidimgs = vidimgs[0:trainFrames + 1].transpose((0, 3, 1, 2))
                vidgaze = np.concatenate((vidmaps[0:trainFrames], vidfixs[0:trainFrames]), axis=-1).transpose((0, 3, 1, 2))

                count_input = batch_size * time_dims
                bs_steps = math.ceil(count_bs / batch_size)
                video_loss = 0.0
                x_state = None
                for idx_bs in range(bs_steps):
                    x_imgs = vidimgs[idx_bs * count_input:(idx_bs + 1) * count_input + 1]
                    y_gaze = vidgaze[idx_bs * count_input:(idx_bs + 1) * count_input]

                    if not np.any(y_gaze, axis=(2, 3)).all():
                        continue

                    if use_cb and x_imgs.shape[0] - 1 != count_input:
                        # continue
                        t_cb_st = get_guasspriors('st', x_imgs.shape[0] - 1 , shape_r_out, shape_c_out, nb_gaussian)
                        t_cb_dy = get_guasspriors('dy', x_imgs.shape[0] - 1 , shape_r_out, shape_c_out, nb_gaussian)
                        t_x_cb = np.concatenate((t_cb_st, t_cb_dy), axis=-1).transpose((0, 3, 1, 2))
                        x_cb_input = torch.tensor(t_x_cb).float()
                    else:
                        x_cb_input = x_cb

                    x_imgs = torch.tensor(x_imgs).float() / 255
                    y_gaze = torch.tensor(y_gaze).float()

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, out_state = model(x_imgs.cuda(), x_cb_input.cuda(), x_state)
                        loss = criterion(outputs, y_gaze.cuda())
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        x_state = [[out_state[0].detach(),out_state[1].detach()]]

                    batch_loss = loss.data.item()
                    video_loss += batch_loss
                    run_loss += batch_loss
                    num_step += 1

                    print("    Batch: [%d / %d], %s loss : %.4f " % (idx_bs + 1, bs_steps, phase.upper(), batch_loss))

                print("    Mean %s loss: %.4f " % (phase.upper(), video_loss / bs_steps))

            mean_run_loss = run_loss / num_step
            print("Epoch: %d / %d, Mean %s loss: %.4f" % (epoch + 1, epochs, phase.upper(), mean_run_loss))


        if not IS_BEST_ONLY:
            output_modename = save_model_path + "%02d_%.4f.pkl" % (epoch, mean_run_loss)
            torch.save(model, output_modename)
        if mean_run_loss < min_val_loss:
            min_val_loss = mean_run_loss
            num_patience = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            num_patience += 1
            if num_patience >= max_patience:
                print('Early stop')
                break

    # Save the best model
    finalmode_name = save_model_path + "final.pkl"
    model.load_state_dict(best_model_wts)
    torch.save(model, finalmode_name)


def test(input_path, output_path, method_name, saveFrames=float('inf'), use_cb=False, nb_gaussian=8,
         iosize=[480, 640, 60, 80], batch_size=4, time_dims=7):

    model_path = saveModelDir + method_name + '/' + method_name + '_final.pkl'
    model = torch.load(model_path)
    model = model.cuda()

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

    file_names = [f for f in os.listdir(input_path) if (f.endswith('.avi') or f.endswith('.mp4'))]
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

            count_bs = int(nframes / time_dims)
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


################################################################
# MODEL PARAMETERS
################################################################
IS_EARLY_STOP = True
IS_BEST_ONLY = False
Max_patience = 5
Train_Max_TrainFrame = float('inf')      #float('inf') for DIEM; 300 for DHF1K
Val_Max_TrainFrame = 1000                # 1000 for DIEM; 300 for DHF1K

################################################################
# DATASET PARAMETERS
################################################################
if os.name == 'nt':
    dataDir = 'E:/DataSet/'
else:
    dataDir = '/home/kao/kao-ssd/DataSet/'

DataSet_Train = 'DIEM'
DataSet_Test = 'DIEM20'

train_dataDir = dataDir + '/' + DataSet_Train + '/'
test_dataDir = dataDir + '/' + DataSet_Test + '/'

test_input_path = test_dataDir + 'Videos/'
test_result_path = test_dataDir + 'Results/Results_STRNN/'
test_output_path = test_result_path + 'Saliency/'

saveModelDir = './weights/temp_weights/'
pre_model_path = 'weights/strnn-vgg16-diem_final.pth'

if DataSet_Test == 'DIEM20':
    saveFrames = 300
else:
    saveFrames = float('inf')

if __name__ == '__main__':

    method_name = 'STRNN'
    epochs = 20
    batch_size = 2

    train(method_name=method_name, batch_size=batch_size, epochs=epochs, cnn_type='vgg16', use_dcn=True, use_cb=False,
          use_bn=False, nb_gaussian=8, cnn_stride=16, out_stride=8, iosize=[480, 640, 60, 80], time_dims=7,
          cat_type=[0, 1, 0, 1], pre_model_path=pre_model_path)

    test(test_input_path, test_output_path, method_name=method_name, saveFrames=saveFrames, use_cb=False,
         nb_gaussian=8, iosize=[480, 640, 60, 80], batch_size=batch_size)

    evalscores_vid(test_dataDir, test_result_path, DataSet='DIEM20', MethodNames=[method_name])

    visual_vid(test_dataDir, test_result_path, DataSet='DIEM20', MethodNames=[method_name], with_color=1, with_fix=1)

