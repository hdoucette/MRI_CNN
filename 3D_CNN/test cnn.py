import os
import tensorflow as tf
import tflearn
import numpy as np
from sys import platform
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

def load_model(model_name='MRI_CNN',epoch=4):
    """load the pre-trained model"""
    try:
        model = MRI_CNN()
        model_path = './Model/model.{0}'.format(epoch)
    except:
        raise NotImplementedError(model_name + ' is not implemented here')

    checkpoint = torch.load(model_path, map_location='cpu')
    #print(checkpoint)
    model.load_state_dict(checkpoint)

    return model


def main(epoch=30):
    # load classification categories
    categories = load_categories()

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model and set to evaluation mode
    model=load_model(epoch)
    model.to(device)
    model.eval()

    csv_path = os.path.join(root,'MRI_CNN/3D_CNN/Model/test_loss_epoch{0}.csv'.format(epoch))
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
                print(prediction)
                prediction = prediction.to(device)
                _, cls = torch.max(prediction, dim=1)

                print("The predicted category is ", cls.data.cpu().numpy()[0])
                print("The real category is", label)
                loss.append([cls.data.cpu().numpy()[0],label])
        writer = csv.writer(writeFile)
        writer.writerows(line for line in loss)

main()