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


if platform=='win32':
   root='C:/Users/douce/Desktop/MIT Fall 2018/6.869 Machine Vision/Final Project/'
else: root='/home/ubuntu'

#Get paths
PD_Path=os.path.join(root,'MRI_CNN/3D_CNN/data')

x,y=DataLoader.load_testing(dataset='train', records=500)
print(x.shape,y.shape)

def run():
    # Parameters
    num_epochs = 100
    output_period = 2
    batch_size = 5

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MRI_CNN()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    # TODO: May Need adjustment
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 12], gamma=0.1)

    write_outputs=[]
    epoch_loss=[]
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
                    epoch, batch_num*1.00/(len(x)/batch_size),
                    running_loss/output_period
                    ))
                running_loss = 0.0
                gc.collect()
        epoch_loss.append([epoch,running_loss])
        gc.collect()
        # save after every epoch
        torch.save(model.state_dict(), "Model/model.%d" % epoch)

        epoch=epoch+1
    csv_path=os.path.join(root,'MRI_CNN/3D_CNN/Model/epoch_loss.csv')
    with open(csv_path, 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(line for line in loss)


print('Starting training')
run()
print('Training terminated')

