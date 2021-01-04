import os
import sys
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist
from numpy.linalg import norm
import csv


def read_feature(path):
    elements1 = []
    with open(path) as file:
       for line in file:
          line = line.strip().split()
          elements1.append(line)

    feature = np.array(elements1)
    feature = feature[:,0].astype(np.float64)
    return feature



Feature_path =  './Output/snapshot/2021-01-02_22-02-09/epoch20/Feature'
Feature_dir = os.listdir(Feature_path)
Feature_dir.sort(key=lambda x:int(x))

## Choose first sample as gallery
Gallery=[]
for ii in range(len(Feature_dir)):
    fea_file = os.listdir(os.path.join(Feature_path,Feature_dir[ii]))
    fea = read_feature(os.path.join(Feature_path,Feature_dir[ii],fea_file[0]))
    Gallery.append(fea.T)
Gallery = np.array(Gallery)
Acc = 0
counter = 0


## Compare distance 
for ii in range(len(Feature_dir)):
    fea_file = os.listdir(os.path.join(Feature_path, Feature_dir[ii]))
    for jj in range(len(fea_file)):
        Probe = read_feature(os.path.join(Feature_path,Feature_dir[ii],fea_file[jj]))
        distance = cdist(Gallery, Probe.reshape(-1,1).T , 'euclidean')
        value = distance.min()
        position = np.where(distance == value)
        if position[0] == (ii):
            Acc = Acc+1
        counter = counter +1
accuracy = Acc/counter
print('The accuracy = {}%\n'.format(accuracy*100))