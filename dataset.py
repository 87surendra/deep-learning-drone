#!/usr/bin/env python
# coding: utf-8

# In[11]:

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import h5py

import cv2
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #to get rid of tensorflow CUDA warnings

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Activation, Dense


# In[2]:


#Define some parameters here
train_path = "data/train/"
train_labels = os.listdir(train_path)
train = []
IMAGE_DIM = 150,150

print(train_labels)


# In[3]:


for label in train_labels:
    dir = os.path.join(train_path, label)
    for file in os.listdir(dir):
        img_path = 'data/train/{}/{}'.format(label,file)
        img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_array = cv2.resize(img_array, IMAGE_DIM)
        train.append([img_array, label])


# In[4]:


X = []
y = []

for data,label in train:
    X.append(data)
    y.append(label)


# In[9]:


#Make HDF5 file


# In[73]:


hdf_file = h5py.File('data.hdf5', 'w')


# In[74]:


hdf_file.create_dataset('data', dtype=np.int0, data=X)
hdf_file.create_dataset('label', dtype=np.int0, data=y)
hdf_file.flush()
hdf_file.close()


# In[75]:


#Read the data from hdf5
h5f_data = h5py.File('data.hdf5', 'r')


# In[76]:


data = h5f_data['data']
label = h5f_data['label']





plt.imshow(data[13])




