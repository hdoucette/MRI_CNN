##This program removes uploaded scans that will not be part of training or testing
#to free up disk space

import os
import nipype
import nipype.interfaces.fsl as fsl
#import win32con
from sys import platform
import pandas as pd


if platform=='win32':
    root='C:/Users/douce/Desktop/MIT Fall 2018/6.869 Machine Vision/Final Project/'
else: root='/home/ubuntu'

PD_Path=os.path.join(root,'oasis-scripts/Patient Data_Last.csv')
df = pd.read_csv(PD_Path, index_col='Subject')


file_path=os.path.join(root,'oasis-scripts/scans')
for file in os.listdir(os.path.join(file_path)):
    patientID = file[:8]
    scanID=df.loc[patientID,'Label']
    if scanID!=file:
        file_path_2=os.path.join(file_path,file)
        for scan in os.listdir(file_path_2):
            scan_path=os.path.join(file_path_2,scan)
            for image in os.listdir(scan_path):
                imgPath=os.path.join(scan_path,image)
                os.remove(imgPath)
            os.rmdir(scan_path)
        os.rmdir(file_path_2)