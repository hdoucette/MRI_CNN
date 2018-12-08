import torch
from torchvision import datasets, transforms
import os
import numpy as np
import torch.utils.data as utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_root = './data/'
train_root = data_root + 'train'
test_root = data_root + 'test'

i=0
if torch.cuda.is_available():
    for file in os.listdir(train_root):
        mri=np.load(os.path.join(train_root,file))
        mri=mri['data']
        if i==0:
            tensor_x=torch.from_numpy(np.expand_dims(mri[0][0],axis=0)).to(device)
            tensor_y=torch.from_numpy(np.expand_dims(mri[0][1],axis=0)).float().to(device)
        else:
            tensor_x2 = torch.from_numpy(np.expand_dims(mri[0][0], axis=0)).to(device)
            tensor_y2 = torch.from_numpy(np.expand_dims(mri[0][1], axis=0)).to(device)
            tensor_x = torch.cat((tensor_x, tensor_x2), dim=0).to(device)
            tensor_y = torch.cat((tensor_y, tensor_y2.float()), dim=0).to(device)
        i+=1
        print(i)
else:
    for file in os.listdir(train_root):
        mri=np.load(os.path.join(train_root,file))
        mri=mri['data']
        if i==0 and mri[0][0].shape==(176, 256, 256):
            tensor_x=torch.from_numpy(np.expand_dims(mri[0][0],axis=0))
            tensor_y = torch.from_numpy(np.expand_dims(mri[0][1],axis=0)).float()
            print('saved',i)
            i += 1
        else:
            if mri[0][0].shape==(176, 256, 256):
                tensor_x2=torch.from_numpy(np.expand_dims(mri[0][0],axis=0))
                tensor_y2=torch.from_numpy(np.expand_dims(mri[0][1], axis=0))
                tensor_x = torch.cat((tensor_x,tensor_x2),dim=0)
                tensor_y = torch.cat((tensor_y,tensor_y2.float()),dim=0)
                print('saved', i)
                i += 1

#print(tensor_x.shape,tensor_y.shape)
train_dataset=utils.TensorDataset(tensor_x,tensor_y)

def get_data_loaders(batch_size):
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader

# def get_test_test_loaders(batch_size):
#     test_loader = torch.utils.data.DataLoader(
#             test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#     return (test_loader)