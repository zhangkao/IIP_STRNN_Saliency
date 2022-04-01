import os, cv2
import numpy as np
import hdf5storage as h5io

from utils_vis import *



if __name__ == "__main__":

	DataSet = 'DIEM20'

	if os.name == 'nt':
		RootDir = 'E:/DataSet/' + DataSet + '/'
	else:
		RootDir = '/home/kao/kao-ssd/DataSet/' + DataSet + '/'

	ResDir = RootDir + 'Results/Results_Oth/'
	MethodNames = [
		'STRNN',
	]

	WITH_FIX = 1
	WITH_COLOT = 1
	visual_vid(RootDir, ResDir, DataSet, MethodNames, with_color=WITH_COLOT, with_fix=WITH_FIX)
