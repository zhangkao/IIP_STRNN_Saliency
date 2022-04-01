from functools import partial
import numpy as np
from numpy import random
from skimage import exposure
from skimage import img_as_float
from skimage.transform import resize
import matplotlib.pyplot as plt
import hdf5storage as h5io
import re, os, glob, cv2

EPSILON = np.finfo('float').eps

from utils_score import *

if __name__ == "__main__":

	DataSet = 'salicon15'

	if os.name == 'nt':
		RootDir = 'E:/DataSet/salicon-15/val/'
	else:
		RootDir = '/home/kao/kao-ssd/DataSet/salicon-15/val/'

	ResDir = RootDir + 'Results/Results_Oth/'
	keys_order = ['AUC_shuffled', 'NSS', 'AUC_Judd', 'AUC_Borji', 'KLD', 'SIM', 'CC']
	MethodNames = [
		'STRNN',
	]

	IS_EVAL_SCORES=1
	if IS_EVAL_SCORES:
		evalscores_img(RootDir, ResDir, DataSet, MethodNames, keys_order)

	IS_ALL_SCORES = 1
	if IS_ALL_SCORES:

		# for matlab implementation
		import matlab
		import matlab.engine
		eng = matlab.engine.start_matlab()
		eng.Img_MeanScore(ResDir, nargout = 0)
		