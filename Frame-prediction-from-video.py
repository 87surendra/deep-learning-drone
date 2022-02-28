#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
import h5py
import pickle


# In[2]:


batch_size = 32
img_height = 180
img_width = 180

train_path = "data/train/"
train_labels = os.listdir(train_path)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_path,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

class_names = train_ds.class_names


# In[3]:


Video_FILE = "flower.mp4"


# In[ ]:





# In[ ]:





# In[4]:


def get_frame(filename,index):
    counter=0
    video=cv2.VideoCapture(filename)
    while video.isOpened():
        rete,frame=video.read()
        if rete:
            if counter==index:
                return cv2.resize(frame, (180, 180)) 
            counter +=1
        else:
            break
    video.release()
    return None


# In[5]:


frame = get_frame(Video_FILE,150)
print('shape is', frame.shape)
print('pixel at (60,21)',frame[60,21,:])
print('pixel at (120,10)',frame[120,10,:])


# In[6]:


pickled_model = pickle.load(open('model.pkl', 'rb'))while


# In[14]:


for i in range(10):
    frame = get_frame(Video_FILE,i)

    img_array = tf.keras.utils.img_to_array(frame)
    img_array = tf.expand_dims(frame, 0) # Create a batch

    predictions = pickled_model.predict(img_array)
    
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


# In[ ]:





# In[ ]:





# In[18]:


video=cv2.VideoCapture(Video_FILE)

while video.isOpened():
    rete,frame=video.read()
    frame = cv2.resize(frame, (180, 180)) 
    img_array = tf.keras.utils.img_to_array(frame)
    img_array = tf.expand_dims(frame, 0) # Create a batch

    predictions = pickled_model.predict(img_array)

    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
video.release()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




