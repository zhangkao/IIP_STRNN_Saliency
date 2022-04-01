#!/usr/bin/env python

import torch
import os, math
import correlation

assert(int(str('').join(torch.__version__.split('.')[0:3])) >= 41) # requires at least pytorch version 0.4.1

##########################################################
Backward_tensorGrid = {}
Backward_tensorPartial = {}
def Backward(tensorInput, tensorFlow):
	if str(tensorFlow.size()) not in Backward_tensorGrid:
		tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
		tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

		Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
	# end

	if str(tensorFlow.size()) not in Backward_tensorPartial:
		Backward_tensorPartial[str(tensorFlow.size())] = tensorFlow.new_ones([ tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3) ])
	# end

	tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)
	tensorInput = torch.cat([ tensorInput, Backward_tensorPartial[str(tensorFlow.size())] ], 1)

	tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')

	tensorMask = tensorOutput[:, -1:, :, :]; tensorMask[tensorMask > 0.999] = 1.0; tensorMask[tensorMask < 1.0] = 0.0

	return tensorOutput[:, :-1, :, :] * tensorMask
# end

##########################################################
class Extractor(torch.nn.Module):
	def __init__(self):
		super(Extractor, self).__init__()

		self.moduleOne = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
		)

		self.moduleTwo = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
		)

		self.moduleThr = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
		)

		self.moduleFou = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
		)

		self.moduleFiv = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
		)

		self.moduleSix = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
		)

	# end

	def forward(self, tensorInput):
		tensorOne = self.moduleOne(tensorInput)
		tensorTwo = self.moduleTwo(tensorOne)
		tensorThr = self.moduleThr(tensorTwo)
		tensorFou = self.moduleFou(tensorThr)
		tensorFiv = self.moduleFiv(tensorFou)
		tensorSix = self.moduleSix(tensorFiv)

		return [tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix]

class Decoder(torch.nn.Module):
	def __init__(self, intLevel):
		super(Decoder, self).__init__()

		intPrevious = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 1]
		intCurrent = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 0]

		if intLevel < 6: self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
		if intLevel < 6: self.moduleUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
		if intLevel < 6: self.dblBackward = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]

		self.moduleOne = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
		)

		self.moduleTwo = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
		)

		self.moduleThr = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
		)

		self.moduleFou = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
		)

		self.moduleFiv = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
		)

		self.moduleSix = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
		)

	# end

	def forward(self, tensorFirst, tensorSecond, objectPrevious):
		tensorFlow = None
		tensorFeat = None

		if objectPrevious is None:
			tensorFlow = None
			tensorFeat = None

			tensorVolume = torch.nn.functional.leaky_relu(
				input=correlation.FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=tensorSecond),
				negative_slope=0.1, inplace=False)

			tensorFeat = torch.cat([tensorVolume], 1)

		elif objectPrevious is not None:
			tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow'])
			tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])

			tensorVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=Backward(tensorInput=tensorSecond, tensorFlow=tensorFlow * self.dblBackward)), negative_slope=0.1, inplace=False)

			tensorFeat = torch.cat([tensorVolume, tensorFirst, tensorFlow, tensorFeat], 1)

		# end

		tensorFeat = torch.cat([self.moduleOne(tensorFeat), tensorFeat], 1)
		tensorFeat = torch.cat([self.moduleTwo(tensorFeat), tensorFeat], 1)
		tensorFeat = torch.cat([self.moduleThr(tensorFeat), tensorFeat], 1)
		tensorFeat = torch.cat([self.moduleFou(tensorFeat), tensorFeat], 1)
		tensorFeat = torch.cat([self.moduleFiv(tensorFeat), tensorFeat], 1)

		tensorFlow = self.moduleSix(tensorFeat)

		return {
			'tensorFlow': tensorFlow,
			'tensorFeat': tensorFeat
		}

class Refiner(torch.nn.Module):
	def __init__(self):
		super(Refiner, self).__init__()

		self.moduleMain = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
		)

	# end

	def forward(self, tensorInput):
		# return self.moduleMain(tensorInput)
		tensorFeat = self.moduleMain[0:12](tensorInput)
		tensorFlow = self.moduleMain[12:](tensorFeat)
		return {
			'tensorFlow': tensorFlow,
			'tensorFeat': tensorFeat
		}


pre_pwc_path = './weights/pwc-default.pytorch'
class PWC_Net(torch.nn.Module):
	def __init__(self,
	             pre_pwc_path=pre_pwc_path):
		super(PWC_Net, self).__init__()

		self.moduleExtractor = Extractor()

		self.moduleTwo = Decoder(2)
		self.moduleThr = Decoder(3)
		self.moduleFou = Decoder(4)
		self.moduleFiv = Decoder(5)
		self.moduleSix = Decoder(6)

		self.moduleRefiner = Refiner()

		if os.path.exists(pre_pwc_path):
			print("Load pre-trained PWC weights")
			self.load_state_dict(torch.load(pre_pwc_path))
	# end

	def forward(self, tensorFirst, tensorSecond):
		tensorFirst = self.moduleExtractor(tensorFirst)
		tensorSecond = self.moduleExtractor(tensorSecond)

		objectEstimate6 = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None)
		objectEstimate5 = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate6)
		objectEstimate4 = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate5)
		objectEstimate3 = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate4)
		objectEstimate2 = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate3)

		refinerOut = self.moduleRefiner(objectEstimate2['tensorFeat'])

		tensorFlow = objectEstimate2['tensorFlow'] + refinerOut['tensorFlow']
		tensorFeat = torch.cat([objectEstimate2['tensorFeat'], refinerOut['tensorFeat']], 1)

		return tensorFlow, tensorFeat

class PWC_Net_PRE(torch.nn.Module):
    def __init__(self,
                 pre_pwc_path=pre_pwc_path):
        super(PWC_Net_PRE, self).__init__()

        self.moduleExtractor = Extractor()

        self.moduleTwo = Decoder(2)
        self.moduleThr = Decoder(3)
        self.moduleFou = Decoder(4)
        self.moduleFiv = Decoder(5)
        self.moduleSix = Decoder(6)

        self.moduleRefiner = Refiner()

        if os.path.exists(pre_pwc_path):
            print("Load pre-trained PWC weights")
            self.load_state_dict(torch.load(pre_pwc_path))
            # self.moduleExtractor.load_state_dict(torch.load(pre_pwc_path), strict=False)

    # end

    def forward(self, tensorFirst, tensorSecond):
        tensorFirst = self.moduleExtractor(tensorFirst)
        tensorSecond = self.moduleExtractor(tensorSecond)

        objectEstimate6 = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None)
        flow6 = objectEstimate6['tensorFlow']
        objectEstimate5 = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate6)
        flow5 = objectEstimate5['tensorFlow']
        objectEstimate4 = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate5)
        flow4 = objectEstimate4['tensorFlow']
        objectEstimate3 = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate4)
        flow3 = objectEstimate3['tensorFlow']
        objectEstimate2 = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate3)

        refinerOut = self.moduleRefiner(objectEstimate2['tensorFeat'])

        flow2 = objectEstimate2['tensorFlow'] + refinerOut['tensorFlow']
        # tensorFeat = torch.cat([objectEstimate2['tensorFeat'], refinerOut['tensorFeat']], 1)

        return flow2, flow3, flow4, flow5, flow6

'''
##########################################################
class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		class Extractor(torch.nn.Module):
			def __init__(self):
				super(Extractor, self).__init__()

				self.moduleOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
			# end

			def forward(self, tensorInput):
				tensorOne = self.moduleOne(tensorInput)
				tensorTwo = self.moduleTwo(tensorOne)
				tensorThr = self.moduleThr(tensorTwo)
				tensorFou = self.moduleFou(tensorThr)
				tensorFiv = self.moduleFiv(tensorFou)
				tensorSix = self.moduleSix(tensorFiv)

				return [ tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix ]
			# end
		# end

		class Decoder(torch.nn.Module):
			def __init__(self, intLevel):
				super(Decoder, self).__init__()

				intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
				intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

				if intLevel < 6: self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.moduleUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.dblBackward = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

				self.moduleOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
				)
			# end

			def forward(self, tensorFirst, tensorSecond, objectPrevious):
				tensorFlow = None
				tensorFeat = None

				if objectPrevious is None:
					tensorFlow = None
					tensorFeat = None

					tensorVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=tensorSecond), negative_slope=0.1, inplace=False)

					tensorFeat = torch.cat([ tensorVolume ], 1)

				elif objectPrevious is not None:
					tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow'])
					tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])

					tensorVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=Backward(tensorInput=tensorSecond, tensorFlow=tensorFlow * self.dblBackward)), negative_slope=0.1, inplace=False)

					tensorFeat = torch.cat([ tensorVolume, tensorFirst, tensorFlow, tensorFeat ], 1)

				# end

				tensorFeat = torch.cat([ self.moduleOne(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleTwo(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleThr(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleFou(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleFiv(tensorFeat), tensorFeat ], 1)

				tensorFlow = self.moduleSix(tensorFeat)

				return {
					'tensorFlow': tensorFlow,
					'tensorFeat': tensorFeat
				}
			# end
		# end

		class Refiner(torch.nn.Module):
			def __init__(self):
				super(Refiner, self).__init__()

				self.moduleMain = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
				)
			# end

			def forward(self, tensorInput):
				return self.moduleMain(tensorInput)
			# end
		# end

		self.moduleExtractor = Extractor()

		self.moduleTwo = Decoder(2)
		self.moduleThr = Decoder(3)
		self.moduleFou = Decoder(4)
		self.moduleFiv = Decoder(5)
		self.moduleSix = Decoder(6)

		self.moduleRefiner = Refiner()

		self.load_state_dict(torch.load('pwc-default.pytorch'))
	# end

	def forward(self, tensorFirst, tensorSecond):
		tensorFirst = self.moduleExtractor(tensorFirst)
		tensorSecond = self.moduleExtractor(tensorSecond)

		objectEstimate6 = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None)
		objectEstimate5 = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate6)
		objectEstimate4 = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate5)
		objectEstimate3 = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate4)
		objectEstimate2 = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate3)

		tensorFlow = objectEstimate2['tensorFlow'] + self.moduleRefiner(objectEstimate2['tensorFeat'])

		return tensorFlow
	# end
# end

moduleNetwork = Network().cuda().eval()
def estimate(tensorFirst, tensorSecond):
	assert(tensorFirst.size(1) == tensorSecond.size(1))
	assert(tensorFirst.size(2) == tensorSecond.size(2))

	intWidth = tensorFirst.size(2)
	intHeight = tensorFirst.size(1)

	assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, intHeight, intWidth)
	tensorPreprocessedSecond = tensorSecond.cuda().view(1, 3, intHeight, intWidth)

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

	tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

	tensorFlow = 20.0 * torch.nn.functional.interpolate(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	return tensorFlow[0, :, :, :].cpu()
# end

##########################################################
import sys, getopt, PIL
import numpy as np
import PIL.Image
from flow_utils import writeFlow, flowToColor

arguments_strFirst = './images/first.png'
arguments_strSecond = './images/second.png'
arguments_strOut = './images/out_pwc.flo'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use
	if strOption == '--first' and strArgument != '': arguments_strFirst = strArgument # path to the first frame
	if strOption == '--second' and strArgument != '': arguments_strSecond = strArgument # path to the second frame
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

if __name__ == '__main__':
	torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance
	torch.cuda.device(1)  # change this if you have a multiple graphics cards and you want to utilize them
	torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

	tensorFirst = torch.FloatTensor(np.array(PIL.Image.open(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
	tensorSecond = torch.FloatTensor(np.array(PIL.Image.open(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))

	tensorOutput = estimate(tensorFirst, tensorSecond)
	writeFlow(arguments_strOut,tensorOutput.numpy().transpose(1, 2, 0))
	flowimg = flowToColor(tensorOutput.transpose(1, 2, 0))
# end

'''