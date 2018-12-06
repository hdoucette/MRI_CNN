import os
import tensorflow as tf
import tflearn
import numpy as np
import csv
from sys import platform
import skimage
from skimage import transform

def traindata(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data=list(reader)
        return data

if platform=='win32':
   root='C:/Users/douce/Desktop/MIT Fall 2018/6.869 Machine Vision/Final Project/'
else: root='/home/ubuntu'

#Get paths
PD_Path=os.path.join(root,'oasis-scripts')
label_path=os.path.join(PD_Path,'train_data.csv')
train_data=traindata(label_path)

datanp=[]                               #images
truenp=[]                               #labels

count=0
denom=len(train_data)
for row in train_data:
    try:
        data=np.load(row[0]+'.npz')
        data=data['data']
        img=data[0][0]
        if img.shape==(176, 256, 256):
            datanp.append(img)
            truenp.append(data[0][1])
        else:
            count=count+1
            print(count," of",denom," eliminated from set")
    except:
        print(row[0],"not loaded")
        count = count+1
        print('exception ',count, " of", denom, " eliminated from set")
datanp=np.array(datanp)
truenp=np.array(truenp)
sh=datanp.shape
#
tf.reset_default_graph()
net = tflearn.input_data(shape=[None,sh[0], sh[1], sh[2],sh[3]])
#when more training data is known, change 1 to 5
net = tflearn.conv_3d(net,16,5,strides=2,activation='leaky_relu', padding='VALID',weights_init='xavier',regularizer='L2',weight_decay=0.01)
#Change kernel size from 1 to 3
net = tflearn.max_pool_3d(net, kernel_size = 3, strides=2, padding='VALID')
net = tflearn.conv_3d(net, 32,3,strides=2, padding='VALID',weights_init='xavier',regularizer='L2',weight_decay=0.01)
net = tflearn.normalization.batch_normalization(net)
net = tflearn.activations.leaky_relu (net)
#kernel size from 1 to 2
net = tflearn.max_pool_3d(net, kernel_size = 2, strides=2, padding='VALID')

net = tflearn.dropout(net,0.5)
net = tflearn.fully_connected(net, 1024,weights_init='xavier',regularizer='L2')
net = tflearn.normalization.batch_normalization(net,gamma=1.1,beta=0.1)
net = tflearn.activations.leaky_relu (net)
net = tflearn.dropout(net,0.6)
net = tflearn.fully_connected(net, 512,weights_init='xavier',regularizer='L2')
net = tflearn.normalization.batch_normalization(net,gamma=1.2,beta=0.2)
net = tflearn.activations.leaky_relu (net)
net = tflearn.dropout(net,0.7)
net = tflearn.fully_connected(net, 128,weights_init='xavier',regularizer='L2')
net = tflearn.normalization.batch_normalization(net,gamma=1.4,beta=0.4)
net = tflearn.activations.leaky_relu (net)
net = tflearn.dropout(net,0.7)
net = tflearn.fully_connected(net, 3,weights_init='xavier',regularizer='L2')
net = tflearn.normalization.batch_normalization(net,gamma=1.3,beta=0.3)
net = tflearn.activations.softmax(net)
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
model = tflearn.DNN(net, checkpoint_path = os.path.join(root,'model.tfl.ckpt'),max_checkpoints=3)                      #model definition
#
ckpt=root
# model.load(ckpt) #loading checkpoints
#
# model.fit(datanp, truenp, batch_size = 8, show_metric=True)   #training with batch size of 8