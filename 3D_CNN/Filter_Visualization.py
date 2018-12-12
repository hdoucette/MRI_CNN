### PREAMBLE ##################################################################
import torch
import torch.nn as nn
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

# the path for the network to load
pathNets = './Model'
fileToLoad = 'model-Stripped.55'

### VISUALIZE FILTERS #########################################################
# load the network into the variable 'net'
net = torch.load(os.path.join(pathNets, fileToLoad), map_location='cpu')

## figuring out dimensions of filters, layers, etc.
for k, v in net.items():
	print(k)

# the filters are of size 3x4x4, and there are 64 of them
# I think this is for the first conv. layer?
#print(net["features.0.weight"].size())

# plot just one channel of every filter
for jj in range(16):
	# get jj-th filter, which is 3x4x4
	print(net["features.0.weight"][jj,0,0].size())
	temp = np.floor((net["features.0.weight"][jj,0])*255);
	# save the first (0th) channel in the variable 'img'
	img = torch.FloatTensor(temp)
	plt.figure()
	plt.imshow(img, cmap='gray')
	plt.show()