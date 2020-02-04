#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 01:21:49 2020

@author: prakash
"""
import os
import glob
import h5py
import shutil
import imgaug as aug
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import imgaug.augmenters as iaa
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential, Model
#from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D,Activation,add,dot
from keras.layers import GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.utils.data_utils import get_file
from keras.optimizers import Adam, SGD, RMSprop,Nadam
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping,ReduceLROnPlateau
from keras.utils import to_categorical
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import cv2
from keras import backend as K
color = sns.color_palette()

import tensorflow as tf



# Set the numpy seed
np.random.seed(111)

# Disable multi-threading in tensorflow ops
session_conf = tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2)

# Set the random seed in tensorflow at graph level
tf.set_random_seed(111)

# Define a tensorflow session with above session configs
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

# Set the session in keras
K.set_session(sess)

# Make the augmentation sequence deterministic
aug.seed(111)


# Define path to the data directory
data_dir = Path('/home/prakash/Documents/test/data')

def preprocessing_data(data_dir):
    # Path to train directory (Fancy pathlib...no more os.path!!)
    train_dir = data_dir / 'train'
    #
    ## Path to validation directory
    val_dir = data_dir / 'valid'
    #
    ## Path to test directory
    test_dir = data_dir / 'test'

    # Get the path to the normal and pneumonia sub-directories
    white_frock_train = train_dir / 'white frock'
    medical_coat_train = train_dir / 'medical white coat in lab'

    # Get the list of all the images
    white_frock = white_frock_train.glob('*.jpg')
    medical_coat = medical_coat_train.glob ('*.jpg')

    def get_data(raw_data1,raw_data2):
        # An empty list. We will insert the data into this list in (img_path, label) format
        train_data = []

        # Go through all the normal cases. The label for these cases will be 0
        for img in raw_data1:
            train_data.append((img,0))

        # Go through all the pneumonia cases. The label for these cases will be 1
        for img in raw_data2:
            train_data.append((img, 1))

        return(train_data)
    train_data = get_data(white_frock,medical_coat)
    # Get a pandas dataframe from the data we have in our list
    train_data = pd.DataFrame(train_data, columns=['image', 'label'],index=None)

    # Shuffle the data
    train_data = train_data.sample(frac=1.).reset_index(drop=True)

    # How the dataframe looks like?
    train_data.head()


    # Get the counts for each class
    cases_count = train_data['label'].value_counts()
    print(cases_count)

    # Plot the results
    plt.figure(figsize=(10,8))
    sns.barplot(x=cases_count.index, y= cases_count.values)
    plt.title('Number of cases', fontsize=14)
    plt.xlabel('Case type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(range(len(cases_count.index)), ['White frock', 'Medical coat'])
    plt.show()


    medical_coat = (train_data[train_data['label']==1]['image'].iloc[:5]).tolist()
    white_frock = (train_data[train_data['label']==0]['image'].iloc[:5]).tolist()

    # Concat the data in a single list and del the above two list
    samples = medical_coat + white_frock
    del medical_coat, white_frock


    # Get the path to the sub-directories
    white_frock_dir = val_dir / 'white frock'
    medical_coat_dir = val_dir / 'medical white coat in lab'

    # Get the list of all the images
    white_frock_valid = white_frock_dir.glob('*.jpg')
    medical_coat_valid = medical_coat_dir.glob('*.jpg')

    # List that are going to contain validation images data and the corresponding labels
    valid_data = []
    valid_labels = []


    # Some images are in grayscale while majority of them contains 3 channels. So, if the image is grayscale, we will convert into a image with 3 channels.
    # We will normalize the pixel values and resizing all the images to 224x224

    # Normal cases
    for img in white_frock_valid:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (224,224))
        if img.shape[2] ==1:
            img = np.dstack([img, img, img])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.
        label = to_categorical(0, num_classes=2)
        valid_data.append(img)
        valid_labels.append(label)

    #  cases
    for img in medical_coat_valid:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (224,224))
        if img.shape[2] ==1:
            img = np.dstack([img, img, img])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.
        label = to_categorical(1, num_classes=2)
        valid_data.append(img)
        valid_labels.append(label)

    # Convert the list into numpy arrays
    valid_data = np.array(valid_data)
    valid_labels = np.array(valid_labels)

    # valid_data = pd.DataFrame(valid_data, columns=['image', 'label'],index=None)
    print("Total number of validation examples: ", valid_data.shape)
    print("Total number of labels:", valid_labels.shape)

    return(train_data,valid_data,valid_labels)

train_data,valid_data,valid_labels = preprocessing_data(data_dir)

def data_gen(data, batch_size):
    # Get total number of samples in the data
    n = len(data)
    steps = n//batch_size

    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size,2), dtype=np.float32)


    seq = iaa.OneOf([
        iaa.Fliplr(), # horizontal flips
        iaa.Affine(rotate=20), # roatation
        iaa.Multiply((1.2, 1.5))]) #random brightnes

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n)

    # Initialize a counter
    i =0
    while True:
        np.random.shuffle(indices)
        # Get the next batch
        count = 0
        next_batch = indices[(i*batch_size):(i+1)*batch_size]
        for j, idx in enumerate(next_batch):
            img_name = data.iloc[idx]['image']
            label = data.iloc[idx]['label']

            # one hot encoding
            encoded_label = to_categorical(label, num_classes=2)
            # read the image and resize
            img = cv2.imread(str(img_name))
            img = cv2.resize(img, (224,224))

            # check if it's grayscale
            if img.shape[2]==1:
                img = np.dstack([img, img, img])

            # cv2 reads in BGR mode by default
            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # normalize the image pixels
            orig_img = img.astype(np.float32)/255.

            batch_data[count] = orig_img
            batch_labels[count] = encoded_label

            # generating more samples of the undersampled class
            if label==0 and count < batch_size-2:
#                aug_img1 = seq.augment_image(img)
#                aug_img2 = seq.augment_image(img)
                aug_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                aug_img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                aug_img1 = aug_img1.astype(np.float32)/255.
                aug_img2 = aug_img2.astype(np.float32)/255.

                batch_data[count+1] = aug_img1
                batch_labels[count+1] = encoded_label
                batch_data[count+2] = aug_img2
                batch_labels[count+2] = encoded_label
                count +=2

            else:
                count+=1

            if count==batch_size-1:
                break

        i+=1
        yield batch_data, batch_labels

        if i>=steps:
            i=0



def optimal_epoch(model_hist):
    '''
    Function to return the epoch number where the validation loss is
    at its minimum

    Parameters:
        model_hist : training history of model

    Output:
        epoch number with minimum validation loss
    '''
    min_epoch = np.argmin(model_hist.history['val_loss']) + 1
    print("Minimum validation loss reached in epoch {}".format(min_epoch))
    return min_epoch
#
#def swish(x):
#    return (K.sigmoid(x) * x)
#
#get_custom_objects().update({'swish': Activation(swish)})


def l1_reg(weight_matrix):
    return 0.00001 * K.sum(K.abs(weight_matrix))


def build_model():

    input_img = Input(shape=(224,224,3), name='ImageInput')
    x = Conv2D(32, (3,3), activation='relu', padding='same', name='Conv1_1',)(input_img)
    x = Conv2D(32, (3,3), activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling2D((2,2), name='pool1')(x)

    residual = Conv2D(64, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv2_1',activity_regularizer=regularizers.l2(0.00005))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv2_2',activity_regularizer=regularizers.l2(0.00005))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv2_3',activity_regularizer=regularizers.l2(0.00005))(x)
    x = MaxPooling2D((2,2), name='pool2',strides=(2, 2))(x)

    x = add([x,residual])

#    x = Conv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1')(x)
#    x = Conv2D(128, (3,3), activation='relu', padding='same', name='Conv3_2',activity_regularizer=regularizers.l2(0.00005))(x)
#    x = Conv2D(128, (3,3), activation='relu', padding='same', name='Conv3_3')(x)
#    x = Conv2D(128, (3,3), activation='relu', padding='same', name='Conv3_4',activity_regularizer=regularizers.l2(0.00005))(x)
#    x = MaxPooling2D((2,2), name='pool3',strides=(2, 2))(x)

    # print('the maxpooling of the images',x)

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1',activity_regularizer=regularizers.l1(0.001),depthwise_regularizer = regularizers.l2(0.0001),bias_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2',depthwise_regularizer = regularizers.l2(0.0001),bias_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization(name='bn2')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3',depthwise_regularizer = regularizers.l2(0.0001),bias_regularizer=regularizers.l2(0.0001))(x)
    x = MaxPooling2D((2,2), name='pool3',strides=(2, 2))(x)


    x = add([x,residual])


    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1',depthwise_regularizer = regularizers.l2(0.00005),bias_regularizer=regularizers.l1(0.001))(x)
    x = BatchNormalization(name='bn3')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2',depthwise_regularizer = regularizers.l2(0.00005),bias_regularizer=regularizers.l1(0.001))(x)
    x = BatchNormalization(name='bn4')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3',depthwise_regularizer = regularizers.l2(0.00005),bias_regularizer=regularizers.l1(0.0001))(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_4',depthwise_regularizer = regularizers.l2(0.00005),bias_regularizer=regularizers.l1(0.0001))(x)
    x = MaxPooling2D((2,2), name='pool4',strides=(2, 2))(x)

#    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l1_l2(l1 = 0.0001,l2 = 0.0001))(x)
    x = Dropout(0.40, name='dropout1')(x)
    x = Dense(512, activation='relu', name='fc2',activity_regularizer=regularizers.l1(0.00001),kernel_regularizer=l1_reg)(x)
    x = Dropout(0.25,name='dropout2')(x)
    x = Dense(2, activation='softmax', name='fc3')(x)

    model = Model(inputs=input_img, outputs=x)
    return model


model =  build_model()
model.summary()

#f = h5py.File('/home/prakash/Downloads/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', 'r')

opt = RMSprop(lr=0.000000001, rho=0.9)
es = EarlyStopping(monitor="val_loss", mode="max", patience=5)
chkpt = ModelCheckpoint(filepath='best_model_todate', save_best_only=True)
model.compile(loss='hinge', metrics=['accuracy'],optimizer=opt)

batch_size = 64
nb_epochs = 150

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=5,
                                            verbose=0.2,
                                            factor=0.5,
                                            min_lr=0.00000000005)

#history = model.fit_generator(datagen.flow(x_train,y_train,batch_size=32),steps_per_epoch=len(x_train)/32,epochs=nEpochs,validation_data=(x_val,y_val), shuffle=True, callbacks = [MyCallback()], verbose=0) # randomly flip images


#validation_data = datagen.flow(data = validation_data batch_size = batch_size)


train_data_gen = data_gen(data=train_data, batch_size=batch_size)
#X_train,X_val,Y_train,Y_val = train_test_split(train_data_gen)

#train_data_gen = datagen.flow(x_train,y_train,batch_size=batch_size)

# Define the number of training steps
nb_train_steps = train_data.shape[0]//batch_size

print("Number of training and validation steps: {} and {}".format(nb_train_steps, len(valid_data)))


# # Fit the model
history = model.fit_generator(train_data_gen, epochs=nb_epochs, verbose = 2 , steps_per_epoch=nb_train_steps,
                              validation_data=(valid_data,valid_labels),callbacks=[learning_rate_reduction,])

history.model.save_weights("/home/prakash/Downloads/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")

#model.load_weights("/home/prakash/Downloads/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")

################################################testing model#############################

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()


# Preparing test data
#white_frock = test_dir / 'white frock'
#medical_coat = test_dir / 'medical white coat in lab'
#
#white_frock_dir = white_frock_dir.glob('*.jpeg')
#medical_coat_dir = medical_coat_dir.glob('*.jpeg')
#
#test_data = []
#test_labels = []
#
#for img in white_frock_dir:
#    img = cv2.imread(str(img))
#    img = cv2.resize(img, (224,224))
#    if img.shape[2] ==1:
#        img = np.dstack([img, img, img])
#    else:
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    img = img.astype(np.float32)/255.
#    label = to_categorical(0, num_classes=2)
#    test_data.append(img)
#    test_labels.append(label)
#
#for img in medical_coat_dir:
#    img = cv2.imread(str(img))
#    img = cv2.resize(img, (224,224))
#    if img.shape[2] ==1:
#        img = np.dstack([img, img, img])
#    else:
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    img = img.astype(np.float32)/255.
#    label = to_categorical(1, num_classes=2)
#    test_data.append(img)
#    test_labels.append(label)
#
#
#test_data = np.array(test_data)
#test_labels = np.array(test_labels)
#
#print("Total number of test examples: ", test_data.shape)
#print("Total number of labels:", test_labels.shape)


#
#test_loss, test_score = history.model.evaluate(test_data, test_labels)
#print("Loss on test set: ", test_loss)
#print("Accuracy on test set: ", test_score)
#

#preds = history.model.predict(test_data)
#preds = np.argmax(preds, axis=-1)

# Original labels
#orig_test_labels = np.argmax(test_labels, axis=-1)
#
#print(orig_test_labels.shape)
#print(preds.shape)
#
#cm  = confusion_matrix(orig_test_labels, preds)
#plt.figure()
#plot_confusion_matrix(cm,figsize=(5,3), hide_ticks=True, cmap=plt.cm.Blues)
#plt.xticks(range(2), ['White frock', 'medical white coat'], fontsize=5)
#plt.yticks(range(2), ['White frock', 'medical white coat'], fontsize=5)
#plt.show()


#tn, fp, fn, tp = cm.ravel()
#
#precision = tp/(tp+fp)
#recall = tp/(tp+fn)
#
#print("Recall of the model is {:.2f}".format(recall))
#print("Precision of the model is {:.2f}".format(precision))
#
