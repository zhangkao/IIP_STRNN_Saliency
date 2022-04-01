from functools import partial
import numpy as np
from utils_score import *

if __name__ == "__main__":

	DataSet = 'DIEM20'

	if os.name == 'nt':
		RootDir = 'E:/DataSet/' + DataSet + '/'
	else:
		RootDir = '/home/kao/kao-ssd/DataSet/' + DataSet + '/'

	ResDir = RootDir + 'Results/Results_Oth/'

	keys_order = ['AUC_shuffled', 'NSS', 'AUC_Judd', 'AUC_Borji', 'KLD', 'SIM', 'CC']
	MethodNames = [
		'STRNN',
	]

	IS_EVAL_SCORES=1
	if IS_EVAL_SCORES:
		evalscores_vid(RootDir, ResDir, DataSet, MethodNames, keys_order)

	IS_ALL_SCORES = 1
	if IS_ALL_SCORES:
		MaxVideoNums = float('inf')
		if DataSet.upper() == 'CITIUS':
			MaxVideoNums = 45

		# for matlab implementation
		import matlab
		import matlab.engine
		eng = matlab.engine.start_matlab()
		eng.Vid_MeanScore(ResDir, nargout = 0)