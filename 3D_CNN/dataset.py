import torch
from torchvision import datasets, transforms
import os
import numpy as np
import torch.utils.data as utils
from torch.utils import data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_root = './data/'
train_root = data_root + 'train'
test_root = data_root + 'test'


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = np.load(os.path.join('./data/train/',ID))['data'][0][0]
        y = np.load(os.path.join('./data/train/',ID))['data'][0][1]
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

        return X, y





class DataLoader(object):
    data=[]
    @classmethod
    def concat(cls, dataset='train', records = -1):
        num=0
        if dataset=='train': dir=train_root
        else: dir=test_root
        denom=len(os.listdir(dir)[0:records])
        for i in os.listdir(dir)[0:records]:
            sample=np.load(os.path.join(dir,i))['data']
            if sample[0][0].shape==(176,256,256):
                if num==0:
                    sample_x=np.expand_dims(sample[0][0],0)
                    sample_y = np.expand_dims(sample[0][1], 0)
                else:
                    sample_x2 = np.expand_dims(sample[0][0], 0)
                    sample_y2 = np.expand_dims(sample[0][1], 0)
                    try:
                        sample_y = np.concatenate((sample_y, sample_y2), axis=0)
                        sample_x=np.concatenate((sample_x,sample_x2),axis=0)
                    except:
                        print('sample out of shape')
                num+=1
                print(num, ' of ',denom)
        np.savez_compressed(os.path.join(data_root,'mris_all_{0}'.format(dataset)),
                            data=sample_x,labels=sample_y)

    @classmethod
    def load(cls, dataset='train', records = -1):
        try:
            raw = np.load(data_root + '/mris_all_{0}.npz'.format(dataset), mmap_mode='r')
        except:
            DataLoader.concat(records=records)
            raw = np.load(data_root + '/mris_all_{0}.npz'.format(dataset), mmap_mode='r')
        if records > 0:
            # additional logic for efficient caching of small subsets of records
            raw_trunc = data_root + 'mris_all_{0}-n{1}.npz'.format(dataset,records)
            if os.path.isfile(raw_trunc):
                raw_x = np.load(raw_trunc, mmap_mode='r')['data']
                raw_y = np.load(raw_trunc, mmap_mode='r')['labels']
                return raw_x[0:records], raw_y[0:records]
            else :
                data, labels =  raw['data'][0:records], raw['labels'][0:records]
                np.savez(raw_trunc, data=data, labels=labels)
                return data, labels
        else:
            return raw['data'], raw['labels']

    @classmethod
    def load_training(cls, dataset='train', records=-1):
        return DataLoader.load(dataset=dataset, records=records)

    @classmethod
    def load_testing(cls, dataset='test', records=-1):
        return DataLoader.load(dataset=dataset, records=records)

    @classmethod
    def batch_data(cls, train_data, train_labels, batch_size):
        """ Simple sequential chunks of data, only for training"""
        for batch in range(int(np.ceil(train_data.shape[0] / batch_size))):
            start = batch_size * batch
            end = start + batch_size
            if end > train_data.shape[0]:
                yield batch, (train_data[start:train_data.shape[0]], \
                      train_labels[start:train_data.shape[0]])
            else:
                yield batch, (train_data[start:end], \
                      train_labels[start:end])






# x,y=DataLoader.load_testing(dataset='train', records=-1)
# print(y)


# for batch_num, (data_batch, label_batch) in DataLoader.batch_data(x,y,5):
#      print(batch_num,data_batch.shape)

# ##Test Visualization
# import pre_processing.Visualisation
# volume=x[0,:,:]
# volume = (volume * 255 / np.max(volume)).astype('uint8')
# multi_slice_viewer(volume)
# #plt.imshow(volume[:,:,0],cmap='gray')
# plt.show()