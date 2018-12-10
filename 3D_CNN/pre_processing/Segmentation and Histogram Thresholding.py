import nibabel
import matplotlib.pyplot as plt
import os
import numpy as np
from sys import platform
import pandas as pd
import csv

def traindata(path,file):
    file_path=os.path.join(path,file)
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data=list(reader)
        return data

if platform=='win32':
   root='C:/Users/douce/Desktop/MIT Fall 2018/6.869 Machine Vision/Final Project/'
else: root='/home/ubuntu'

#Read Patient Data
PD_Path=os.path.join(root,'MRI_CNN/3D_CNN/data')
train_list=traindata(PD_Path,'train_data.csv')


def segmentation_Threshholding(img_dir):  #image directory
    img=nibabel.load(img_dir)  #loading the image
    img_data=img.get_data()

    hist,bins=np.histogram(img_data[img_data!=0],bins='auto',density=True)         #mapping the histogram of the image using probability density function(density=True), background black values are ignored.
    bins=0.5*(bins[1:]+bins[:-1])                                                  #taking midpoints of bins

    t1=0                                                                           #threshold1 index
    t2=0                                                                           #threshold2 index

    currvar=0                                                                      #we have to maximise this value
    u=np.zeros(3)                                                                  #mean of the three distributions
    w=np.zeros(3)                                                                  #weightages of the three distributions

    uT=np.vdot(bins,hist)/np.sum(hist)                                             #mean of the full histogram

    for i in range(1,int(len(hist)/2)):
        w[0]=np.sum(hist[:i])/np.sum(hist)
        u[0]=np.vdot(bins[:i],hist[:i])/np.sum(hist[:i])
        for j in range(i+1,len(hist)):
            w[1]=np.sum(hist[i:j])/np.sum(hist)
            u[1]=np.vdot(bins[i:j],hist[i:j])/np.sum(hist[i:j])
            w[2] = np.sum(hist[j:])/np.sum(hist)
            u[2] =np.vdot(bins[j:],hist[j:])/np.sum(hist[j:])
            maxvar=np.vdot(w,(np.power((u-uT),2)))                                  #according to formula
            if(maxvar>currvar):                                                     #maximimsing currvar
                currvar=maxvar
                print(currvar)
                t2 = i
                t1 = j

    plt.bar(bins,hist,width=1)
    plt.axvline(bins[t1],c='r')  #plotting histogram with the two thresholds,red vertical line is threshold1 and green vertical line is threshold2
    plt.axvline(bins[t2],c='g')
    plt.show()

    threshold1=bins[t1]
    threshold2=bins[t2]
    print('threshold1 = ',threshold1)
    print('threshold2 = ',threshold2)

segmentation_Threshholding(train_list[69][0])

num=0
# for i in train_list:
#     print(num,i[2])
#     num=num+1