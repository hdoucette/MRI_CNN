import os
import pandas as pd
import numpy as np
import skimage
from skimage import transform
import nibabel
from sys import platform
import csv
import matplotlib.pyplot as plt

# from skimage.viewer import ImageViewer

def getdata(path,file):
    file_path=os.path.join(path,file)
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data=list(reader)
        return data

if platform=='win32':
   root='C:/Users/douce/Desktop/MIT Fall 2018/6.869 Machine Vision/Final Project/'
else: root='/home/ubuntu'

#Read Patient Data
PD_Path=os.path.join(root,'3D-Convnet-for-Alzheimer-s-Detection/3D_CNN/data')
train_list=getdata(PD_Path,'train_data.csv')
test_list=getdata(PD_Path,'test_data.csv')
denom=len(train_list)+len(test_list)

label_path=os.path.join(PD_Path,'train_data.csv')
labels_df=pd.read_csv(label_path,names=['path','diagnosis'])

#labeling and object formation
num=0
for list in [train_list,test_list]:
    if list==train_list:
        folder='train'
    else: folder='test'
    for file in list:
        path=file[0]
        img_name=file[1]
        label=file[2]
        if int(float(label)) == 0:
            labelar = np.array([1, 0, 0])
        elif int(float(label)) >= 2:
            labelar = np.array([0, 0, 1])
        else:
            labelar = np.array([0, 0, 1])

        # will be used for numpy object
        netdata=[]
        try:
            #load image
            img = nibabel.load(path)
            img = img.get_data()
            num=num+1
            #append image and label to netdata
            netdata.append([img, labelar])
            #save as compressed numpy array with array name = 'data'
            img_path=PD_Path+'/{0}/{1}'.format(folder,img_name)
            #print(img_path)
            np.savez_compressed(img_path, data=netdata)
            print(num, 'of',denom," is npz saved")
        except:
            print(path,' could not be appended and saved as numpy array')

#normalization
num=0
#total number of pixels in the image
totalnum=[]
#mean of the pixels in the image
mean=[]
#maximum value of pixels in the image
nummax=[]
num=0

train_folder=os.path.join(PD_Path,'train')
test_folder=os.path.join(PD_Path,'test')
for type in ['train','test']:
    folder=os.path.join(PD_Path,type)
    dir= os.listdir(folder)
    for file in dir:
        file_name=os.path.join(folder,file)
        try:
            img = np.load(file_name)
            #obtain array out of compressed file
            img = img['data']
            average=np.mean(img[0][0])
            max=np.max(img[0][0])
            size=img[0][0].shape[0]*img[0][0].shape[1]*img[0][0].shape[2]
            mean.append(average)
            totalnum.append(size)
            nummax.append(max)
            num=num+1
            print(num," of",denom," appended to mean and max")
        except:
            print(file_name," could not be included in mean and max calculations")


nummean=np.vdot(mean,totalnum)/np.sum(totalnum)
nummax=np.max(nummax)
print('NUMMAX:',nummax,' NUMMEAN:',nummean)

num=0
for type in ['train','test']:
    folder=os.path.join(PD_Path,type)
    dir= os.listdir(folder)
    for file in dir:
        file_name=os.path.join(folder,file)
        try:
            #load compressed array
            img = np.load(file_name)
            img = img['data']
            #normalize by mean and max
            img[0][0]=(img[0][0]-nummean)/nummax #normalisation(x-mean/max value)
            num+=1
            print(num,' of',denom, ' is normalized')
            np.savez_compressed(file[0], data=img)
        except:
            print(file[0],' could not be normalized')
