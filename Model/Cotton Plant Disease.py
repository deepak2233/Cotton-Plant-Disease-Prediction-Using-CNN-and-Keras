#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


# In[39]:


os.getcwd()


# In[40]:


cd/Users/IRON MAN/Cotton Plant/Train


# In[41]:


os.listdir(os.getcwd())


# In[42]:


# loading dataset
data = []
labels = []
classes = 4
cur_path = os.getcwd()

for i in os.listdir(cur_path):
    dir = cur_path + '/' + i
    for j in os.listdir(dir):
        img_path = dir+'/'+j
        img = cv2.imread(img_path,-1)
        img = cv2.resize(img, (150,150), interpolation = cv2.INTER_NEAREST)
        data.append(img)
        labels.append(i)
        
data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape)


# In[43]:


cd/Users/IRON MAN/Cotton Plant/Test


# In[44]:


# loading dataset
test_data = []
test_labels = []
classes = 4
cur_path = os.getcwd()

for i in os.listdir(cur_path):
    dir = cur_path + '/' + i
    for j in os.listdir(dir):
        img_path = dir+'/'+j
        img = cv2.imread(img_path,-1)
        img = cv2.resize(img, (150,150), interpolation = cv2.INTER_NEAREST)
        test_data.append(img)
        test_labels.append(i)
        
test_data = np.array(test_data)
test_labels = np.array(test_labels)
print(test_data.shape,test_labels.shape)


# In[84]:


#Splitting training and testing dataset
X_train, X_test, y_train, y_test = (data, test_data,labels,test_labels)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[51]:


cd/Users/IRON MAN/Cotton Plant/Test


# In[52]:


os.listdir(os.getcwd())


# In[53]:


cd/Users/IRON MAN/Cotton Plant/val


# In[54]:


os.listdir(os.getcwd())


# In[55]:


cd/Users/IRON MAN/Cotton Plant/Train


# In[56]:


training_data_path = os.getcwd()


# In[57]:


# this is the augmentation configuration we will use for training
# It generate more images using below parameters
training_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

# this is a generator that will read pictures found in
# at train_data_path, and indefinitely generate
# batches of augmented image data
training_data = training_datagen.flow_from_directory(training_data_path, # this is the target directory
                                      target_size=(150, 150), # all images will be resized to 150x150
                                      batch_size=32,
                                      class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels


# In[58]:


# As you can see
os.listdir(os.getcwd())


# In[59]:


training_data.class_indices


# In[60]:


cd/Users/IRON MAN/Cotton Plant/val


# In[61]:


validation_data_path = os.getcwd()


# In[62]:


# this is the augmentation configuration we will use for validation:
# only rescaling
valid_datagen = ImageDataGenerator(rescale=1./255)

# this is a similar generator, for validation data
valid_data = valid_datagen.flow_from_directory(validation_data_path,
                                  target_size=(150,150),
                                  batch_size=32,
                                  class_mode='binary')


# In[63]:


valid_data.class_indices


# In[64]:


cd/Users/IRON MAN/Cotton Plant/Train


# In[65]:


curr_path_train = os.getcwd()


# In[66]:


# number of images in each class for training datasets
data_dic = {}
for folder in os.listdir(curr_path_train):
    data_dic[folder] = len(os.listdir(curr_path_train + '/' + folder))

data_df= pd.Series(data_dic)
plt.figure(figsize = (15, 6))
data_df.sort_values().plot(kind = 'bar')
plt.xlabel('Training Classes')
plt.ylabel('Number of Traingn images')


# In[67]:


cd/Users/IRON MAN/Cotton Plant/val


# In[68]:


cur_val_path = os.getcwd()


# In[69]:


# number of images in each class for training datasets
data_dic = {}
for folder in os.listdir(cur_val_path):
    data_dic[folder] = len(os.listdir(cur_val_path + '/' + folder))

data_df= pd.Series(data_dic)
plt.figure(figsize = (15, 6))
data_df.sort_values().plot(kind = 'bar')
plt.xlabel('Valedation Classes')
plt.ylabel('Number of Valedation images')


# In[ ]:





# In[70]:


# show augmented images
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


# In[71]:


# showing augmented images
images = [training_data[0][0][0] for i in range(5)]
plotImages(images)


# In[72]:


cd/Users/IRON MAN/Cotton Plant


# In[73]:


# save best model using vall accuracy
model_path = 'Cotton Plant Disease.h5'
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[74]:


#Building the model
model = Sequential()

# First Layer
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=[150, 150, 3]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

# Second Layer 
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))


# Dense Layer
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(4, activation='softmax'))


# In[75]:



# compile cnn model
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[76]:


model.summary()


# In[77]:


get_ipython().system('pip install visualkeras')
import visualkeras
visualkeras.layered_view(model)


# In[79]:


# train cnn model
history = model.fit(training_data,epochs=50, batch_size = 128, verbose=1, validation_data= valid_data) 


# In[80]:


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[81]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
 
history.history


# In[ ]:




