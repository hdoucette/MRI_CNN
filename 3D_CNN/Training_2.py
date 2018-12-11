import os
import csv
from sys import platform
import torch
from dataset import DataLoader, Dataset
import gc
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from CNN_Model import *
from torch.utils import data

torch.backends.cudnn.benchmark=True


def run():
    if platform == 'win32':
        root = 'C:/Users/douce/Desktop/MIT Fall 2018/6.869 Machine Vision/Final Project/'
    else:
        root = '/home/ubuntu'

    # Get paths
    PD_Path = os.path.join(root, 'MRI_CNN/3D_CNN/data')

    # print(x.shape,y.shape)

    params = {'batch_size': 20,
              'shuffle': True,
              'num_workers': 4}

    num_epochs=100
    output_period=5
    partition = os.listdir('./data/train')

    # Generators
    training_set = Dataset(partition)
    training_generator = data.DataLoader(training_set, **params)


    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MRI_CNN()
    model = model.to(device)

    weights = torch.DoubleTensor([1.0, 2.0, 10.0])
    criterion = nn.CrossEntropyLoss(weight=weights).to(device)

    # TODO: May Need adjustment
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 12], gamma=0.1)

    write_outputs=[]
    epoch = 1
    csv_path = os.path.join(root, 'MRI_CNN/3D_CNN/Model/epoch_loss.csv')
    epoch_loss = []
    with open(csv_path, 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        while epoch <= num_epochs:
            running_loss = 0.0
            for param_group in optimizer.param_groups:
                print('Current learning rate: ' + str(param_group['lr']))
            model.train()

            for batch_num,(inputs, labels) in enumerate(training_generator):
                inputs = inputs.unsqueeze(1).to(device)
                labels = labels.long().to(device)
                print(inputs.shape)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs,labels)

                loss.backward()

                optimizer.step()
                running_loss += loss.item()
                #print('Running Loss:',running_loss)
                if batch_num % output_period == 0:
                    print('[%d:%.2f] loss: %.3f' % (
                        epoch, batch_num*1.00/(len(training_generator)/params['batch_size']),
                        running_loss/output_period
                        ))
                    running_loss = 0.0
                    gc.collect()
            epoch_loss.append([epoch,running_loss])
            gc.collect()
            # save after every epoch
            torch.save(model.state_dict(), "Model/model.%d" % epoch)

            epoch=epoch+1

        writer.writerows(line for line in epoch_loss)


print('Starting training')
run()
print('Training terminated')

