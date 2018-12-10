import os
import tensorflow as tf
import tflearn
import numpy as np
import sys
from CNN_Model import *
import torch
import csv

if platform=='win32':
   root='C:/Users/douce/Desktop/MIT Fall 2018/6.869 Machine Vision/Final Project/'
else: root='/home/ubuntu'

data_root = './data/'
test_root = data_root + 'test'

datanp=[]                               #images
truenp=[]                               #labels

def load_categories():
    categories = list([])
    for line in [0,1,2]:
        categories.append(line)
    return categories

def main():
    # load classification categories
    categories = load_categories()

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model and set to evaluation mode
    model=MRI_CNN()
    model.to(device)
    model.eval()

    csv_path = os.path.join(root,'MRI_CNN/3D_CNN/Model/test_loss.csv')
    with open(csv_path, 'w', newline='') as writeFile:
        loss=[]

        # load the image
        for file in os.listdir(test_root):
            data = np.load(os.path.join(test_root, file))
            datanp.append(data['data'])
            #truenp.append(data['labels'])
            img=data['data'][0][0]
            label=data['data'][0][1]
            if img.shape==(176,256,256):
                image = torch.from_numpy(img)

                image = image.to(device)
                inputs = image.unsqueeze(0).to(device)
                inputs = inputs.unsqueeze(0).to(device)

                # run the forward process
                prediction = model(inputs)
                prediction = prediction.to(device)
                _, cls = torch.max(prediction, dim=1)

                print("The predicted category is ", cls.data.cpu().numpy()[0])
                print("The real category is", label)
                loss.append([cls.data.cpu().numpy()[0],label])
        writer = csv.writer(writeFile)
        writer.writerows(line for line in loss)

main()