import os, cv2
import numpy as np
import hdf5storage as h5io

from utils_vis import *

def visual_vid_frame(RootDir, SalDir, DataSet, MethodNames, VID_NUM, Frame_NUM, with_color=1, with_fix=0):

	vidsDir = RootDir + 'Videos/'
	mapsDir = RootDir + 'maps/'
	fixsDir = RootDir + 'fixations/maps/'
	salsDir = SalDir

	out_path = SalDir + 'frame_out/'
	if not os.path.exists(out_path):
		os.makedirs(out_path)

	vid_ext = '.mp4'
	if DataSet.upper() == 'CITIUS':
		vid_ext = '.avi'
	elif DataSet.upper() in ['DHF1K-TE','DHF1K']:
		vid_ext = '.AVI'

	for idx_m in range(len(MethodNames)):
		print("---" + str(idx_m + 1) + "/" + str(len(MethodNames)) + "---: " + MethodNames[idx_m])

		vid_names = [f for f in os.listdir(vidsDir) if f.endswith(vid_ext)]
		vid_names.sort()

		for idx_n in VID_NUM:
			print(str(idx_n + 1) + "/" + str(len(vid_names)) + ": " + vid_names[idx_n])

			file_name = vid_names[idx_n][:-4]
			VideoCap = cv2.VideoCapture(vidsDir + file_name + vid_ext)
			vidframes = int(VideoCap.get(cv2.CAP_PROP_FRAME_COUNT))

			if MethodNames[idx_m] == 'GT':
				gt_path = salsDir + MethodNames[idx_m] + '/'
				if not os.path.exists(gt_path):
					os.makedirs(gt_path)
				gtname = gt_path + file_name + '.mat'
				if not os.path.exists(gtname):
					salmap = np.rint(h5io.loadmat(mapsDir + file_name + '_fixMaps.mat')["fixMap"]).astype(np.uint8)
					if DataSet.upper() == 'DIEM20':
						salmap = salmap[:,:,:,:300]
					h5io.savemat(gtname, {'salmap': salmap})
				else:
					salmap_dir = salsDir + MethodNames[idx_m] + '/'
					salmap = np.rint(h5io.loadmat(salmap_dir + file_name + '.mat')["salmap"]).astype(np.uint8)
			else:
				salmap_dir = salsDir + MethodNames[idx_m] + '/'
				salmap = np.rint(h5io.loadmat(salmap_dir + file_name + '.mat')["salmap"]).astype(np.uint8)

			nframes = min(vidframes, salmap.shape[3])
			fixname = fixsDir + file_name + '_fixPts.mat'
			if with_fix and os.path.exists(fixname):
				fixpts = h5io.loadmat(fixname)["fixLoc"]
				nframes = min(nframes, fixpts.shape[3])

			if Frame_NUM=='ALL':
				r_frames = range(0,nframes,5)
			else:
				r_frames = Frame_NUM

			for idx_f in r_frames:

				outname = out_path + file_name + '_' + str(idx_f) + '_' + MethodNames[idx_m] + '.png'
				# if os.path.exists(outname):
				# 	continue

				isalmap = salmap[:, :, 0, idx_f]

				VideoCap.set(cv2.CAP_PROP_POS_FRAMES, idx_f)
				if with_color:
					ret, img = VideoCap.read()
					iovermap = heatmap_overlay(img, isalmap)
				else:
					iovermap = np.repeat(np.expand_dims(isalmap, axis=2), 3, axis=2) / 255

				if with_fix and os.path.exists(fixname):
					ifixpts = fixpts[:, :, 0, idx_f]
					ifixpts_dilate = cv2.dilate(ifixpts, np.ones((5, 5), np.uint8))
					ifixpts_dilate = np.repeat(np.expand_dims(ifixpts_dilate, axis=2), 3, axis=2)
					iovermap[ifixpts_dilate > 0.5] = 1

				iovermap = iovermap / np.max(iovermap) * 255
				iovermap = im2uint8(iovermap)

				cv2.imwrite(outname,iovermap)
				imgname = out_path + file_name + '_' + str(idx_f) + '_frame.png'
				if not os.path.exists(imgname):
					cv2.imwrite(imgname,img)

			VideoCap.release()

if __name__ == "__main__":

	DataSet = 'DIEM20'

	if os.name == 'nt':
		RootDir = 'E:/DataSet/' + DataSet + '/'
	else:
		RootDir = '/home/kao/kao-ssd/DataSet/' + DataSet + '/'

	ResDir = RootDir + 'Results/Results_Oth/Saliency/'
	MethodNames = [
		'STRNN',
	]

	WITH_FIX = 1
	WITH_COLOT = 1

	VID_NUM = [0]
	Frame_NUM = 'ALL'

	visual_vid_frame(RootDir, ResDir, DataSet, MethodNames, VID_NUM, Frame_NUM, with_color=WITH_COLOT, with_fix=WITH_FIX)
