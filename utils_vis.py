import os, cv2
import numpy as np
import hdf5storage as h5io

EPS = 2.2204e-16

def im2uint8(img):
    if img.dtype == np.uint8:
        return img
    else:
        img[img < 0] = 0
        img[img > 255] = 255
        img = np.rint(img).astype(np.uint8)
        return img

def heatmap_overlay(image, heatmap):

	img = np.array(image,   copy=True)
	map = np.array(heatmap, copy=True)

	if img.shape[:2] != map.shape[:2]:
		map = cv2.resize(map, (img.shape[1],img.shape[0]))

	if len(map.shape) == 2:
		map = np.repeat(np.expand_dims(map, axis=2), 3, axis=2)

	if map.dtype == np.uint8:
		map_color = cv2.applyColorMap(map, cv2.COLORMAP_JET)
	else:
		tmap = im2uint8(map/np.max(map)*255)
		map_color = cv2.applyColorMap(tmap, cv2.COLORMAP_JET)

	img = img / (np.max(img) + EPS)
	map = map / (np.max(map) + EPS)
	map_color = map_color / np.max(map_color)

	o_map = 0.8 * (1 - map ** 0.8) * img + map * map_color
	return o_map


def visual_img(RootDir, salsDir, MethodNames, with_fix=0):

	imgsDir = RootDir + 'images/'
	fixsDir = RootDir + 'fixations/maps/'
	# salsDir = RootDir + 'Results/Saliency/'

	img_ext = '.jpg'
	sal_ext = '.png'

	for idx_m in range(len(MethodNames)):
		print("---" + str(idx_m + 1) + "/" + str(len(MethodNames)) + "---: " + MethodNames[idx_m])

		salmap_dir = salsDir + MethodNames[idx_m] + '/'
		out_path = salmap_dir + 'Visual_color/'
		if not os.path.exists(out_path):
			os.makedirs(out_path)

		sal_names = [f for f in os.listdir(salmap_dir) if f.endswith(sal_ext)]
		sal_names.sort()

		for idx_n in range(len(sal_names)):
			print(str(idx_n + 1) + "/" + str(len(sal_names)) + ": " + sal_names[idx_n])

			file_name = sal_names[idx_n][:-4]
			outname = out_path + file_name + sal_ext
			if os.path.exists(outname):
				continue

			img    = cv2.imread(imgsDir + file_name + img_ext,-1)
			salmap = cv2.imread(salmap_dir + file_name + sal_ext,-1)

			fixname = fixsDir + file_name + '.mat'
			if with_fix and os.path.exists(fixname):
				fixmap = h5io.loadmat(fixname)["I"]

			overmap = heatmap_overlay(img,salmap)
			if with_fix and os.path.exists(fixname):
				fixpts_dilate = cv2.dilate(fixmap, np.ones((5, 5), np.uint8))
				fixpts_dilate = np.repeat(np.expand_dims(fixpts_dilate, axis=2), 3, axis=2)
				overmap[fixpts_dilate > 0.5] = 1

			overmap = overmap / np.max(overmap) *255
			cv2.imwrite(out_path + file_name + sal_ext, im2uint8(overmap))

def visual_vid(RootDir, SalDir, DataSet, MethodNames, with_color=0, with_fix=0):

	vidsDir = RootDir + 'Videos/'
	fixsDir = RootDir + 'fixations/maps/'
	salsDir = SalDir + 'Saliency/'

	vid_ext = '.mp4'
	if DataSet.upper() in ['CITIUS', 'UAV2', 'UAV2-TE']:
		vid_ext = '.avi'
	elif DataSet.upper() in ['DHF1K-TE','DHF1K']:
		vid_ext = '.AVI'

	for idx_m in range(len(MethodNames)):
		print("---" + str(idx_m + 1) + "/" + str(len(MethodNames)) + "---: " + MethodNames[idx_m])

		salmap_dir = salsDir + MethodNames[idx_m] + '/'
		if with_color:
			out_path = salmap_dir + 'Visual_color/'
		else:
			out_path = salmap_dir + 'Visual_gray/'

		if not os.path.exists(out_path):
			os.makedirs(out_path)

		sal_names = [f for f in os.listdir(salmap_dir) if f.endswith('.mat')]
		sal_names.sort()

		for idx_n in range(len(sal_names)):
			print(str(idx_n + 1) + "/" + str(len(sal_names)) + ": " + sal_names[idx_n])

			file_name = sal_names[idx_n][:-4]
			outname = out_path + file_name + '.mp4'
			if os.path.exists(outname):
				continue

			VideoCap = cv2.VideoCapture(vidsDir + file_name + vid_ext)
			vidsize = (int(VideoCap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(VideoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
			vidframes = int(VideoCap.get(cv2.CAP_PROP_FRAME_COUNT))
			vidfps = VideoCap.get(cv2.CAP_PROP_FPS)

			salmap = np.rint(h5io.loadmat(salmap_dir + file_name + '.mat')["salmap"]).astype(np.uint8)

			nframes = min(vidframes, salmap.shape[3])
			fixname = fixsDir + file_name + '_fixPts.mat'
			if with_fix and os.path.exists(fixname):
				fixpts = h5io.loadmat(fixname)["fixLoc"]
				nframes = min(nframes, fixpts.shape[3])

			fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
			VideoWriter = cv2.VideoWriter(outname, fourcc, vidfps, vidsize, isColor=True)

			for idx_f in range(nframes):

				isalmap = salmap[:, :, 0, idx_f]

				if with_color:
					ret, img = VideoCap.read()
					iovermap = heatmap_overlay(img, isalmap)
				else:
					iovermap = np.repeat(np.expand_dims(isalmap, axis=2), 3, axis=2)/255

				if with_fix and os.path.exists(fixname):
					ifixpts = fixpts[:, :, 0, idx_f]
					ifixpts_dilate = cv2.dilate(ifixpts,np.ones((5,5), np.uint8))
					ifixpts_dilate = np.repeat(np.expand_dims(ifixpts_dilate, axis=2), 3, axis=2)
					iovermap[ifixpts_dilate>0.5] = 1

				iovermap = iovermap / np.max(iovermap) *255
				VideoWriter.write(im2uint8(iovermap))

			VideoCap.release()
			VideoWriter.release()


if __name__ == "__main__":


	# for video visualization
	DataSet = 'DIEM20'
	RootDir = 'E:/Dataset_Models/DataSet/' + DataSet + '/'
	ResDir = RootDir + 'Results/'
	MethodNames = [
		'strnn',
	]

	WITH_FIX = 1
	WITH_COLOT = 1
	visual_vid(RootDir, ResDir, DataSet, MethodNames, with_color=WITH_COLOT, with_fix=WITH_FIX)

	# # for images visualization
	# DataSet = 'salicon'
	# RootDir = 'E:/Dataset_Models/DataSet/salicon-15/val/'
	# salsDir = 'E:/Dataset_Models/DataSet/salicon-15/val/'
	# MethodNames = ['maps']
	#
	# WITH_FIX = 1
	# visual_color_img(RootDir, salsDir, MethodNames, with_fix=WITH_FIX)
