import os
import csv
from sys import platform
import torch
from dataset import DataLoader
import gc
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from CNN_Model import *

torch.backends.cudnn.benchmark=True


def traindata(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data=list(reader)
        return data

if platform=='win32':
   root='C:/Users/douce/Desktop/MIT Fall 2018/6.869 Machine Vision/Final Project/'
else: root='/home/ubuntu'

#Get paths
PD_Path=os.path.join(root,'MRI_CNN/3D_CNN/data')

x,y=DataLoader.load_testing(dataset='train', records=-1)

def run():
    # Parameters
    num_epochs = 10
    output_period = 5
    batch_size = 2

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MRI_CNN()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    # TODO: May Need adjustment
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 12], gamma=0.1)

    epoch = 1
    while epoch <= num_epochs:
        running_loss = 0.0
        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))
        model.train()

        for batch_num,(inputs, labels) in DataLoader.batch_data(x, y, batch_size):
            inputs=torch.from_numpy(inputs)
            labels=torch.from_numpy(labels)
            inputs = inputs.unsqueeze(1).to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)

            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            #print('Running Loss:',running_loss)
            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f' % (
                    epoch, batch_num*1.0/(len(x)),
                    running_loss/output_period
                    ))
                running_loss = 0.0
                gc.collect()
        gc.collect()
        # save after every epoch
        torch.save(model.state_dict(), "Model/model.%d" % epoch)


print('Starting training')
run()
print('Training terminated')