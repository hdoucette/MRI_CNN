import numpy as np
import scipy
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
import tflearn
import tensorflow as tf
from sys import platform
import os
import math
import torch.nn as nn
import torch.nn.functional as F
from scipy.misc import imread


class MRI_CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(MRI_CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1,16,kernel_size=5,stride=2,padding=0),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool3d(kernel_size=3,stride=2,padding=0),
            nn.Conv3d(16,32,kernel_size=2,stride=2,padding=0),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(.5))
        self.classifier=nn.Sequential(nn.Linear(72000,1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(.5),
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(.5),
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(.5),
            nn.Linear(128,3),
            nn.BatchNorm1d(3),
            nn.Softmax())

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 72000)
        x = self.classifier(x)
        return x

def CNN_Model():
    model = MRI_CNN()
    return model