import torch
from torchvision import datasets, transforms
import os
import numpy as np
import torch.utils.data as utils
import tempfile
from my_savez import mySavez

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_root = './data/'


class DataLoader(object):
    data=[]
    @classmethod
    def concat(cls, dataset='train', records = -1):
        num=0
        denom=len(os.listdir(data_root+dataset))
        for i in os.listdir(data_root+dataset):
            sample=np.load(os.path.join(data_root+dataset,i))['data']
            if sample[0][0].shape==(176,256,256):
                if num==0:
                    sample_x=np.expand_dims(sample[0][0],0)
                    sample_y = np.expand_dims(sample[0][1], 0)
                    np.savez_compressed(os.path.join(data_root, 'mris_all_{0}'.format(dataset)),
                                        data=sample_x, labels=sample_y)

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
            DataLoader.concat(dataset=dataset)
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
                return data, labels[:, 0], labels[:, 1]
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
        """ Simple sequential chunks of data """
        for batch in range(int(np.ceil(train_data.shape[0] / batch_size))):
            start = batch_size * batch
            end = start + batch_size
            if end > train_data.shape[0]:
                yield batch, (train_data[-batch_size,:], \
                      train_labels[-batch_size,:])
            else:
                yield batch, (train_data[start:end], \
                      train_labels[start:end])


# x=DataLoader.load(dataset='test', records=-1)


# for batch_num, (data_batch, label_batch) in DataLoader.batch_data(x,y,5):
#      print(batch_num,data_batch.shape)

# ##Test Visualization
# import pre_processing.Visualisation
# volume=x[0,:,:]
# volume = (volume * 255 / np.max(volume)).astype('uint8')
# multi_slice_viewer(volume)
# #plt.imshow(volume[:,:,0],cmap='gray')
# plt.show()
