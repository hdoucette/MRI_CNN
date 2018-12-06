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



K.set_image_data_format('channels_last')

def get_CNN(shape=[100,176,256, 256],optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy'):
    if platform == 'win32':
        root = 'C:/Users/douce/Desktop/MIT Fall 2018/6.869 Machine Vision/Final Project/'
    else:
        root = '/home/ubuntu'
    tf.reset_default_graph()
    net = tflearn.input_data(shape)
    net = tflearn.conv_3d(net, 16, 5, strides=2, activation='leaky_relu', padding='VALID', weights_init='xavier',
                          regularizer='L2', weight_decay=0.01)
    net = tflearn.max_pool_3d(net, kernel_size=3, strides=2, padding='VALID')
    net = tflearn.conv_3d(net, 32, 3, strides=2, padding='VALID', weights_init='xavier', regularizer='L2',
                          weight_decay=0.01)
    net = tflearn.normalization.batch_normalization(net)
    net = tflearn.activations.leaky_relu(net)
    net = tflearn.max_pool_3d(net, kernel_size=2, strides=2, padding='VALID')
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, 1024, weights_init='xavier', regularizer='L2')
    net = tflearn.normalization.batch_normalization(net, gamma=1.1, beta=0.1)
    net = tflearn.activations.leaky_relu(net)
    net = tflearn.dropout(net, 0.6)
    net = tflearn.fully_connected(net, 512, weights_init='xavier', regularizer='L2')
    net = tflearn.normalization.batch_normalization(net, gamma=1.2, beta=0.2)
    net = tflearn.activations.leaky_relu(net)
    net = tflearn.dropout(net, 0.7)
    net = tflearn.fully_connected(net, 128, weights_init='xavier', regularizer='L2')
    net = tflearn.normalization.batch_normalization(net, gamma=1.4, beta=0.4)
    net = tflearn.activations.leaky_relu(net)
    net = tflearn.dropout(net, 0.7)
    net = tflearn.fully_connected(net, 3, weights_init='xavier', regularizer='L2')
    net = tflearn.normalization.batch_normalization(net, gamma=1.3, beta=0.3)
    net = tflearn.activations.softmax(net)
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
    model = tflearn.DNN(net, checkpoint_path=os.path.join(root, 'model.tfl.ckpt'), max_checkpoints=3)