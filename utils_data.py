from __future__ import division
import math,random,os,scipy,cv2
import numpy as np
import scipy.io
import scipy.ndimage
import hdf5storage as h5io
EPS = 2.2204e-16

def resize_img(img, maxvalue=640, minvalue=320):
    o_rows, o_cols = img.shape[0:2]
    max_shape = max(o_rows, o_cols)
    min_shape = min(o_rows, o_cols)
    rate = max(maxvalue/max_shape,minvalue/min_shape)

    n_rows = int(o_rows*rate)
    n_cols = int(o_cols*rate)
    img = cv2.resize(img, (n_cols,n_rows))
    return img

def resize_pts(img, maxvalue=640, minvalue=320):
    o_rows, o_cols = img.shape[0:2]
    max_shape = max(o_rows, o_cols)
    min_shape = min(o_rows, o_cols)
    rate = max(maxvalue/max_shape,minvalue/min_shape)

    n_rows = int(o_rows*rate)
    n_cols = int(o_cols*rate)

    out = np.zeros((n_rows, n_cols),np.uint8)
    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0]*rate))
        c = int(np.round(coord[1]*rate))
        if r == n_rows:
            r -= 1
        if c == n_cols:
            c -= 1
        out[r, c] = 1

    return out


def normalize_data(data, mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225]):

    ims = data.astype(np.float32) / 255.0
    if len(ims.shape)==3:
        ims[0, :, :] = (ims[0, :, :] - mean[0]) / std[0]
        ims[1, :, :] = (ims[1, :, :] - mean[1]) / std[1]
        ims[2, :, :] = (ims[2, :, :] - mean[2]) / std[2]
    elif len(ims.shape)==4:
        ims[:, 0, :, :] = (ims[:, 0, :, :] - mean[0]) / std[0]
        ims[:, 1, :, :] = (ims[:, 1, :, :] - mean[1]) / std[1]
        ims[:, 2, :, :] = (ims[:, 2, :, :] - mean[2]) / std[2]
    elif len(ims.shape)==5:
        ims[:, :, 0, :, :] = (ims[:, :, 0, :, :] - mean[0]) / std[0]
        ims[:, :, 1, :, :] = (ims[:, :, 1, :, :] - mean[1]) / std[1]
        ims[:, :, 2, :, :] = (ims[:, :, 2, :, :] - mean[2]) / std[2]
    else:
        raise ValueError

    return ims

def imgs_submean(input_imgs, mb = 103.939, mg = 116.779, mr = 123.68):
    imgs = np.zeros(input_imgs.shape,dtype=np.float32)
    if imgs.shape[3] == 3:
        imgs[:, :, :, 0] = input_imgs[:, :, :, 0] - mb
        imgs[:, :, :, 1] = input_imgs[:, :, :, 1] - mg
        imgs[:, :, :, 2] = input_imgs[:, :, :, 2] - mr
    else:
        raise NotImplementedError
    return imgs

def im2uint8(img):
    if img.dtype == np.uint8:
        return img
    else:
        img[img < 0] = 0
        img[img > 255] = 255
        img = np.rint(img).astype(np.uint8)
        return img

def np2mat(img, dtype=np.uint8):

    if dtype == np.uint8:
        return im2uint8(img)
    else:
        return img.astype(dtype)

def saveVid(savename, data):

    h,w,c,nframes=data.shape
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    VideoWriter = cv2.VideoWriter(savename, fourcc, 30, (w,h), isColor=True)

    for idx_f in range(nframes):
        isalmap = data[:, :, :, idx_f]
        VideoWriter.write(im2uint8(isalmap))
    VideoWriter.release()

#########################################################################
# Videos TRAINING SETTINGS
#########################################################################
def shuffleData4Dir(data_path, ratio=0.8, shuffle=True, saveTxt=True):

    imgs_train_path = data_path + '/videos/'
    images = [f for f in os.listdir(imgs_train_path) if f.endswith(('.mp4', '.avi'))]

    if shuffle:
        random.shuffle(images)

    train_num = int(len(images) * ratio)
    train_images = images[:train_num]
    val_images = images[train_num:]

    train_images.sort()
    val_images.sort()

    if saveTxt:
        f = open(data_path+'/train.txt','w')
        lists = [str(line) + "\n" for line in train_images]
        f.writelines(lists)
        f.close()

        f = open(data_path+'/val.txt','w')
        lists = [str(line) + "\n" for line in val_images]
        f.writelines(lists)
        f.close()

    return train_images, val_images

def shuffleData4List(list_path, ratio=0.8, shuffle=True, saveTxt=True):

    data_path, _ = os.path.split(list_path)
    f = open(list_path)
    images = f.readlines()
    images = [f.strip('\n') for f in images]

    if shuffle:
        random.shuffle(images)

    train_num = int(len(images) * ratio)
    train_images = images[:train_num]
    val_images = images[train_num:]

    train_images.sort()
    val_images.sort()

    if saveTxt:
        f = open(data_path+'/train.txt','w')
        lists = [str(line) + "\n" for line in train_images]
        f.writelines(lists)
        f.close()

        f = open(data_path+'/val.txt','w')
        lists = [str(line) + "\n" for line in val_images]
        f.writelines(lists)
        f.close()

    return train_images, val_images


def read_video_list(datapath, phase_gen='train', shuffle=True):
    if phase_gen in ['train', 'val']:
        txt_path     = datapath + '/txt/' + phase_gen + '.txt'
        videos_path  = datapath + '/Videos/'
        vidmaps_path = datapath + '/maps/'
        vidfixs_path = datapath + '/fixations/maps/'
    else:
        raise NotImplementedError

    f = open(txt_path)
    lines = f.readlines()

    if shuffle:
        random.shuffle(lines)

    videos  = [videos_path + f.strip('\n') + '.mp4' for f in lines]
    vidmaps = [vidmaps_path + f.strip('\n') + '_fixMaps.mat' for f in lines]
    vidfixs = [vidfixs_path + f.strip('\n') + '_fixPts.mat' for f in lines]
    f.close()

    return videos, vidmaps, vidfixs

def get_video_list(datapath, phase_gen='train', shuffle=True):

    if phase_gen in ['train', 'val']:
        videos_path  = datapath + '/' + phase_gen + '/videos/'
        vidmaps_path = datapath + '/' + phase_gen + '/maps/'
        vidfixs_path = datapath + '/' + phase_gen + '/fixations/maps/'
    else:
        raise NotImplementedError

    videos  = [videos_path + f for f in os.listdir(videos_path) if (f.endswith('.avi') or f.endswith('.mp4'))]
    vidmaps = [vidmaps_path + f for f in os.listdir(vidmaps_path) if f.endswith('.mat')]
    vidfixs = [vidfixs_path + f for f in os.listdir(vidfixs_path) if f.endswith('.mat')]

    if shuffle:
        out = list(zip(videos, vidmaps, vidfixs))
        random.shuffle(out)
        videos, vidmaps, vidfixs = zip(*out)
    else:
        videos.sort()
        vidmaps.sort()
        vidfixs.sort()

    return videos, vidmaps, vidfixs

####################################################################
# Preprocess input and output video data
####################################################################
def preprocess_images(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 3),np.float32)

    for i, path in enumerate(paths):
        original_image = cv2.imread(path)
        padded_image = padding(original_image, shape_r, shape_c, 3)
        ims[i] = padded_image

    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68

    return ims

def preprocess_maps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1),np.float32)

    for i, path in enumerate(paths):
        original_map = cv2.imread(path, 0)
        padded_map = padding(original_map, shape_r, shape_c, 1)
        ims[i,:,:,0] = padded_map.astype(np.float32)
        ims[i,:,:,0] /= 255.0

    return ims

def preprocess_fixmaps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1),np.uint8)

    for i, path in enumerate(paths):
        fix_map = scipy.io.loadmat(path)["I"]
        ims[i,:,:,0] = padding_fixation(fix_map, shape_r=shape_r, shape_c=shape_c)

    return ims

def preprocess_vidmaps(path, shape_r, shape_c, frames=float('inf')):

    fixmaps = h5io.loadmat(path)["fixMap"]
    h,w,c,nframes = fixmaps.shape
    nframes = min(nframes, frames)

    ims = np.zeros((nframes, shape_r, shape_c, 1),np.uint8)
    for i in range(nframes):
        original_map = fixmaps[:,:,:,i]
        ims[i, :, :, 0] = padding(original_map, shape_r, shape_c, 1)

    return ims

def preprocess_vidfixs(path, shape_r, shape_c, frames=float('inf')):

    fixmaps = h5io.loadmat(path)["fixLoc"]
    h,w,c,nframes = fixmaps.shape
    nframes = min(nframes, frames)

    ims = np.zeros((nframes, shape_r, shape_c, 1),np.uint8)
    for i in range(nframes):
        original_map = fixmaps[:,:,0,i]
        ims[i, :, :, 0] = padding_fixation(original_map, shape_r, shape_c)

    return ims

def preprocess_videos(path, shape_r, shape_c, frames=float('inf'), mode='RGB' ,normalize=True):

    cap = cv2.VideoCapture(path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    nframes = min(nframes,frames)
    ims = np.zeros((nframes, shape_r, shape_c, 3),np.uint8)
    for idx_frame in range(nframes):
        ret, frame = cap.read()
        ims[idx_frame] = padding(frame, shape_r, shape_c, 3)

    # ims = ims.astype(np.float32) / 255.0
    if mode == 'RGB':
        ims = ims[:,:,:,[2,1,0]]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif mode == 'BGR':
        mean = [0.456, 0.406, 0.485]
        std = [0.224, 0.225, 0.229]
    else:
        raise ValueError

    if normalize:
        ims = ims.astype(np.float32) / 255.0
        ims[:, :, :, 0] = (ims[:, :, :, 0] - mean[0] ) / std[0]
        ims[:, :, :, 1] = (ims[:, :, :, 1] - mean[1] ) / std[1]
        ims[:, :, :, 2] = (ims[:, :, :, 2] - mean[2] ) / std[2]

    cap.release()

    return ims,nframes,height,width

def postprocess_predictions(pred, shape_r, shape_c):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    return img / np.max(img) * 255

def padding(img, shape_r=480, shape_c=640, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded

def resize_fixation(img, rows=480, cols=640):
    out = np.zeros((rows, cols),np.uint8)
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0]*factor_scale_r))
        c = int(np.round(coord[1]*factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 1

    return out

def padding_fixation(img, shape_r=480, shape_c=640):
    img_padded = np.zeros((shape_r, shape_c),np.uint8)

    original_shape = img.shape
    if original_shape[0] == shape_r and original_shape[1] == shape_c:
        return img

    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = resize_fixation(img, rows=shape_r, cols=new_cols)
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = resize_fixation(img, rows=new_rows, cols=shape_c)
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


#####################################################################
#Generate gaussmaps
#####################################################################
def st_get_gaussmaps(height,width,nb_gaussian):
    e = height / width
    e1 = (1 - e) / 2
    e2 = e1 + e

    mu_x = np.repeat(0.5,nb_gaussian,0)
    mu_y = np.repeat(0.5,nb_gaussian,0)

    sigma_x = e*np.array(np.arange(1,9))/16
    sigma_y = sigma_x

    x_t = np.dot(np.ones((height, 1)), np.reshape(np.linspace(0.0, 1.0, width), (1, width)))
    y_t = np.dot(np.reshape(np.linspace(e1, e2, height), (height, 1)), np.ones((1, width)))

    x_t = np.repeat(np.expand_dims(x_t, axis=-1), nb_gaussian, axis=2)
    y_t = np.repeat(np.expand_dims(y_t, axis=-1), nb_gaussian, axis=2)

    gaussian = 1 / (2 * np.pi * sigma_x * sigma_y + EPS) * \
               np.exp(-((x_t - mu_x) ** 2 / (2 * sigma_x ** 2 + EPS) +
                       (y_t - mu_y) ** 2 / (2 * sigma_y ** 2 + EPS)))

    return gaussian

def dy_get_gaussmaps(height,width,nb_gaussian):
    e = height / width
    e1 = (1 - e) / 2
    e2 = e1 + e

    mu_x = np.repeat(0.5,nb_gaussian,0)
    mu_y = np.repeat(0.5,nb_gaussian,0)


    sigma_x = np.array([1/4,1/4,1/4,1/4,
                        1/2,1/2,1/2,1/2])
    sigma_y = e*np.array([1 / 16, 1 / 8, 3 / 16, 1 / 4,
                          1 / 8, 1 / 4, 3 / 8, 1 / 2])
    # sigma_x = np.ones(nb_gaussian) / 2
    # sigma_y = e * np.array(np.arange(1, 9)) / 16

    # sigma_x = np.array([4 / height, 8 / height, 16 / height, 32 / height,
    #            4 / height, 8 / height, 16 / height, 32 / height,
    #            4 / height, 8 / height, 16 / height, 32 / height,
    #            4 / height, 8 / height, 16 / height, 32 / height])
    # sigma_x = e*np.array(np.arange(1,9))/16
    # sigma_y = sigma_x

    x_t = np.dot(np.ones((height, 1)), np.reshape(np.linspace(0.0, 1.0, width), (1, width)))
    y_t = np.dot(np.reshape(np.linspace(e1, e2, height), (height, 1)), np.ones((1, width)))

    x_t = np.repeat(np.expand_dims(x_t, axis=-1), nb_gaussian, axis=2)
    y_t = np.repeat(np.expand_dims(y_t, axis=-1), nb_gaussian, axis=2)

    gaussian = 1 / (2 * np.pi * sigma_x * sigma_y + EPS) * \
               np.exp(-((x_t - mu_x) ** 2 / (2 * sigma_x ** 2 + EPS) +
                       (y_t - mu_y) ** 2 / (2 * sigma_y ** 2 + EPS)))

    return gaussian

def get_guasspriors(type='st', b_s=2, shape_r=60, shape_c=80, channels = 8):

    if type == 'dy':
        ims = dy_get_gaussmaps(shape_r, shape_c, channels)
    else:
        ims = st_get_gaussmaps(shape_r, shape_c, channels)

    ims = np.expand_dims(ims, axis=0)
    ims = np.repeat(ims,b_s,axis=0)

    return ims

def get_guasspriors_3d(type = 'st', b_s = 2, time_dims=7, shape_r=60, shape_c=80, channels = 8):

    if type == 'dy':
        ims = dy_get_gaussmaps(shape_r, shape_c, channels)
    else:
        ims = st_get_gaussmaps(shape_r, shape_c, channels)

    ims = np.expand_dims(ims, axis=0)
    ims = np.repeat(ims, time_dims, axis=0)

    ims = np.expand_dims(ims, axis=0)
    ims = np.repeat(ims, b_s, axis=0)
    return ims



def get_gaussmaps(height,width,nb_gaussian):
    e = height / width
    e1 = (1 - e) / 2
    e2 = e1 + e

    mu_x = np.repeat(0.5,16,0)
    mu_y = np.repeat(0.5,16,0)

    # sigma_x = np.array([4 / width, 4 / width, 4 / width, 4 / width,
    #            8 / width, 8 / width, 8 / width, 8 / width,
    #            16 / width, 16 / width, 16 / width, 16 / width,
    #            32 / width, 32 / width, 32 / width, 32 / width])
    sigma_y = np.array([4 / height, 8 / height, 16 / height, 32 / height,
               4 / height, 8 / height, 16 / height, 32 / height,
               4 / height, 8 / height, 16 / height, 32 / height,
               4 / height, 8 / height, 16 / height, 32 / height])
    sigma_x = sigma_y.transpose()

    x_t = np.dot(np.ones((height, 1)), np.reshape(np.linspace(0.0, 1.0, width), (1, width)))
    y_t = np.dot(np.reshape(np.linspace(e1, e2, height), (height, 1)), np.ones((1, width)))

    x_t = np.repeat(np.expand_dims(x_t, axis=-1), nb_gaussian, axis=2)
    y_t = np.repeat(np.expand_dims(y_t, axis=-1), nb_gaussian, axis=2)

    gaussian = 1 / (2 * np.pi * sigma_x * sigma_y + EPS) * \
               np.exp(-((x_t - mu_x) ** 2 / (2 * sigma_x ** 2 + EPS) +
                       (y_t - mu_y) ** 2 / (2 * sigma_y ** 2 + EPS)))

    return gaussian

def preprocess_priors(b_s, shape_r, shape_c, channels = 16):

    ims = get_gaussmaps(shape_r, shape_c, channels)

    ims = np.expand_dims(ims, axis=0)
    ims = np.repeat(ims,b_s,axis=0)

    return ims

def preprocess_priors_3d(b_s, time_dims,shape_r, shape_c, channels = 16):

    ims = get_gaussmaps(shape_r, shape_c, channels)

    ims = np.expand_dims(ims, axis=0)
    ims = np.repeat(ims, time_dims, axis=0)

    ims = np.expand_dims(ims, axis=0)
    ims = np.repeat(ims, b_s, axis=0)
    return ims


