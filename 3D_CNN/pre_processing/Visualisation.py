import nibabel
import matplotlib.pyplot as plt                                                 #package imports
import os
import gzip

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'a':
        previous_slice(ax)
    elif event.key == 'q':
        next_slice(ax)
    fig.canvas.draw()

def multi_slice_viewer(volume):
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index =100
    ax.imshow(volume[:,:,ax.index],cmap='gray')
    fig.canvas.mpl_connect('key_press_event', process_key)

def previous_slice(ax):
    """Go to the previous slice."""
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[:,:,ax.index])

def next_slice(ax):
    """Go to the next slice."""
    volume = ax.volume
    ax.index = (ax.index + 1) % (volume.shape[0])
    ax.images[0].set_array(volume[:,:,ax.index])


# Visualize .nii.gz file
# import skimage
# from skimage import transform
# data_dir='C:/Users\douce\Desktop\MIT Fall 2018/6.869 Machine Vision\Final Project\oasis-scripts\scans\OAS30002_MR_d0653/anat2' #image directory
# img=nibabel.load(os.path.join(data_dir,'sub-OAS30002_ses-d0653_run-01_T1w.nii.gz'))
# img_data=img.get_data()
# # img_data = skimage.transform.resize(img_data, (176, 256, 256), mode='constant')
# multi_slice_viewer(img_data)
# plt.show()
#
# # #Visualize numpy compressed file
import numpy as np
root='C:/Users\douce\Desktop\MIT Fall 2018/6.869 Machine Vision\Final Project/MRI_CNN/3D_CNN\data/train'
file_name=root+'/sub-OAS30065_ses-d2009_T1w_stripped.nii.gz.npz'
print(type(file_name))
img = np.load(file_name)
img = img['data']
img_data=img[0][0]
print(img[0][1])
volume=img_data
#volume = (img_data * 255 / np.max(img_data)).astype('uint8')
multi_slice_viewer(volume)
#plt.imshow(volume[:,:,0],cmap='gray')
plt.show()