
# coding: utf-8

# In[1]:

import keras
import h5py
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
from keras import backend as K
from tqdm import tqdm
import os

batch_size = 16
nb_epoch = 20
infodir = 'C:/Science_Research/xraynet_data/images'
imgdir = f'{infodir}/images_cat'
train_data_dir  = f'{imgdir}/train'
validation_data_dir = f'{imgdir}/test'
MODEL_NAME = 'xrayNet-keras-v2.1' #gve the model a name
# input image dimensions
img_size = 128

nb_train_samples = 100000
nb_validation_samples = 3416


# In[2]:

if not os.path.exists('path_to_label_trunc.txt'):
    with open('Data_Entry_2017.csv','r') as csv:
        subjects = []
        for row in csv:
            subjects.append(row.strip().split(',')[:2])
        subjects.pop(0)
        random.shuffle(subjects)
        #subjects.sort(key=lambda x: x[1])

    normal = [case for case in subjects if 'No Finding' == case[1]]
    abnormal = [case for case in subjects if 'No Finding' != case[1]]

    normal = normal[:min(len(normal),len(abnormal))]
    abnormal = abnormal[:min(len(normal),len(abnormal))]

    trunc_lists = normal + abnormal
    random.shuffle(trunc_lists)
    with open('path_to_label_trunc.txt','w') as out_txt:
        for row in trunc_lists:
            out_txt.write(f'{imgdir}/{row[0]} {row[1]}\n')

    with open('train_paths.txt','w') as out_txt1:
        for row in trunc_lists[:-2000]:
            out_txt1.write(f'{imgdir}/{row[0]} {0 if row[1] == "No Finding" else 1}\n')

    with open('test_paths.txt','w') as out_txt2:
        for row in trunc_lists[-2000:]:
            out_txt2.write(f'{imgdir}/{row[0]} {0 if row[1] == "No Finding" else 1}\n')


# In[3]:

tbCallBack = keras.callbacks.TensorBoard(log_dir=f'./TBLOGS/{MODEL_NAME}', histogram_freq=0,  
          write_graph=True, write_images=True)


# In[4]:

from keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(512, (3, 3), input_shape=(img_size,img_size,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              metrics=['accuracy'])


# In[ ]:

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale')


model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples//batch_size,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples//batch_size,
        callbacks=[tbCallBack])


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:






# In[ ]:

model.fit(X, Y,
          epochs=epochs,
          batch_size=batch_size,
          verbose=3,
          shuffle = 'batch', 
          validation_data=(test_X,test_Y),    #data to be be used as vailidation
          callbacks=[tbCallBack]) #pushes data to tensorboard


# In[ ]:




# In[ ]:

model.save('{}.{}.model'.format(MODEL_NAME,f'{batch_size}')) #save the model with the name in the working directory


# In[ ]:

#print(model.predict(np.array([X[-1]]))[0])
#print(Y[-1])

pred_Y = model.predict(test_X)
pred_Y = (pred_Y > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_Y, pred_Y)


# In[ ]:

print(cm)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# ### [0, 1] == Normal == 0
# 
# ### [1, 0] == Abnormal == 1
# 
# ### [Abnormalness, Normalness]

# In[ ]:




# In[ ]:




# In[ ]:



