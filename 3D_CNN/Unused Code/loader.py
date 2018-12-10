import os
from keras import backend as K
import numpy as np
from sys import platform

K.set_image_data_format('channels_last')

class DataLoader(object):

    @classmethod
    def load(cls, dataset='training',set='train'):
        if platform == 'win32':
            root = 'C:/Users/douce/Desktop/MIT Fall 2018/6.869 Machine Vision/Final Project/'
        else:
            root = '/home/ubuntu'
        raw = np.load(root + '/Model/datanp_{0}'.format(dataset), mmap_mode='r')
        return

    @classmethod
    def load_training(cls, dataset='train'):
        return DataLoader.load(dataset=dataset)

    @classmethod
    def load_testing(cls, dataset='datanp_testing.npz"', records=-1):
        return DataLoader.load(dataset=dataset)

    @classmethod
    def batch_data(cls, train_data, train_labels, batch_size):
        """ Simple sequential chunks of data """
        for batch in range(int(np.ceil(train_data.shape[0] / batch_size))):
            start = batch_size * batch
            end = start + batch_size
            if end > train_data.shape[0]:
                yield train_data[-batch_size:, ...], \
                        train_labels[-batch_size:, ...]
            else:
                yield train_data[start:end, ...], \
                        train_labels[start:end, ...]


DataLoader.load_training()