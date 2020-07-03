#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:31:43 2020

@author: nikos
"""


import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers.merge import concatenate #Concatenate (capital C) not working 
from keras import regularizers
from keras import constraints

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.backend import clear_session

import time
import skimage.transform

def rotateT(X,angle):
    #rotate image tensor, TF order, single channel
    X_rot = np.zeros_like(X)
    #repeat for every channel
    for ch in np.arange(X.shape[-1]):
        #print('channel',ch)
        #repeat for every image
        for i in np.arange(X.shape[0]):
            #print('image',i)
            X_rot[i,:,:,ch] = skimage.transform.rotate(X[i,:,:,ch],angle=angle,resize=False,preserve_range=True,mode='edge')
    return(X_rot)

def shiftT(X,dx,dy):
    #rotate image tensor, TF order, single channel
    X_shift = np.zeros_like(X)
    #repeat for every image
    tform = skimage.transform.SimilarityTransform(translation=(dx, dy))
    for i in np.arange(X.shape[0]):
        #print('image',i)
        X_shift[i,:,:,:] = skimage.transform.warp(X[i,:,:,:],tform,mode='edge')
    return(X_shift)

import keras
print(keras.__version__)#used with '2.3.1'
import tensorflow
print(tensorflow.__version__)#used with '1.15.0'

#%% set-up the generator for data augmentation

def aug_generator(X_raw=None,Y_raw=None,
                  batch_size=4,
                  flip_axes=[1],
                  rotation_angles=[5,15]):
    
    batch_size=batch_size#recommended batch size    
    Ndatapoints = len(X_raw)
    
    while(True):
        #print('start!')
        ix_randomized = np.random.choice(Ndatapoints,size=Ndatapoints,replace=False)
        ix_batches = np.array_split(ix_randomized,int(Ndatapoints/batch_size))
        for b in range(len(ix_batches)):
            #print('step',b,'of',len(ix_batches))
            ix_batch = ix_batches[b]
            current_batch_size=len(ix_batch)
            #print('size of current batch',current_batch_size)
            #print(ix_batch)
            X_batch = X_raw[ix_batch,:,:,:]
            Y_batch = Y_raw[ix_batch,:,:,:]
            
            #now do augmentation on images and masks
            #iterate over each image in the batch
            for img in range(current_batch_size):
                #print('current_image',img,': ',ix_batch[img])
                do_aug=np.random.choice([True, False],size=1)[0]#50-50 chance
                if do_aug == True:
                    flip_axis_selected = np.random.choice(flip_axes,1,replace=False)[0]
                    #flip an axis
                    X_batch[img,:,:,:] = np.flip(X_batch[img,:,:,:],axis=flip_axis_selected)
                    Y_batch[img,:,:,:] = np.flip(Y_batch[img,:,:,:],axis=flip_axis_selected)
                    #print('Flip on axis',flip_axis_selected)
                
                do_aug=np.random.choice([True, False],size=1)[0]#50-50 chance
                if do_aug == True:
                    rotation_angle_selected = np.random.uniform(low=rotation_angles[0],high=rotation_angles[1],size=1)[0]
                    #rotate the image
                    X_batch[img,:,:,:] = rotateT(np.expand_dims(X_batch[img,:,:,:],axis=0),angle=rotation_angle_selected)
                    Y_batch[img,:,:,:] = rotateT(np.expand_dims(Y_batch[img,:,:,:],axis=0),angle=rotation_angle_selected)
                    #print('Rotate angle',rotation_angle_selected)
                
            yield(X_batch,Y_batch)
            #print('step end after',b,'of',len(ix_batches))

#%% #load the data from already split files
# use your data here
X_tr = np.load('./X_train.npy')
Y_tr = np.load('./Y_train.npy')

X_val = np.load('./X_valid.npy')
Y_val = np.load('./Y_valid.npy')

# X_ts = np.load('./X_test.npy')#test set not needed for training
# Y_ts = np.load('./Y_test.npy')#test set not needed for training

#%% set-up the S-UNET model
#   train 100 instances to use in an ensemble

NRUNS = 100

#model parameters
bnorm_axis = -1
nfilters = np.array([64, 128, 256, 512, 1024])
nfilters = (nfilters/8).astype('int')

lmb1 = 0.01
filepath_orig = 'sunet'#filename to save trained model

for run in range(NRUNS):
    clear_session()
    print('clearing Keras session...')
    #input
    input_tensor = Input(shape=X_tr.shape[1:], name='input_tensor')
    
    ####################################
    # encoder (contracting path)
    ####################################
    #encoder block 0
    e0 = Conv2D(filters=1, use_bias=False, kernel_size=(1,1), padding='same',
                name='input_filter',
                kernel_regularizer = regularizers.l1(lmb1),
                kernel_constraint = constraints.NonNeg())(input_tensor)
    e0 = Conv2D(filters=nfilters[0], kernel_size=(3,3), padding='same')(e0)
    e0 = BatchNormalization(axis=bnorm_axis)(e0)
    e0 = Activation('relu')(e0)
    e0 = Conv2D(filters=nfilters[0], kernel_size=(3,3), padding='same')(e0)
    e0 = BatchNormalization(axis=bnorm_axis)(e0)
    e0 = Activation('relu')(e0)
    
    #encoder block 1
    e1 = MaxPooling2D((2, 2))(e0)
    e1 = Conv2D(filters=nfilters[1], kernel_size=(3,3), padding='same')(e1)
    e1 = BatchNormalization(axis=bnorm_axis)(e1)
    e1 = Activation('relu')(e1)
    e1 = Conv2D(filters=nfilters[1], kernel_size=(3,3), padding='same')(e1)
    e1 = BatchNormalization(axis=bnorm_axis)(e1)
    e1 = Activation('relu')(e1)
    
    #encoder block 2
    e2 = MaxPooling2D((2, 2))(e1)
    e2 = Conv2D(filters=nfilters[2], kernel_size=(3,3), padding='same')(e2)
    e2 = BatchNormalization(axis=bnorm_axis)(e2)
    e2 = Activation('relu')(e2)
    e2 = Conv2D(filters=nfilters[2], kernel_size=(3,3), padding='same')(e2)
    e2 = BatchNormalization(axis=bnorm_axis)(e2)
    e2 = Activation('relu')(e2)
    
    #encoder block 3
    e3 = MaxPooling2D((2, 2))(e2)
    e3 = Conv2D(filters=nfilters[3], kernel_size=(3,3), padding='same')(e3)
    e3 = BatchNormalization(axis=bnorm_axis)(e3)
    e3 = Activation('relu')(e3)
    e3 = Conv2D(filters=nfilters[3], kernel_size=(3,3), padding='same')(e3)
    e3 = BatchNormalization(axis=bnorm_axis)(e3)
    e3 = Activation('relu')(e3)
    
    #encoder block 4
    e4 = MaxPooling2D((2, 2))(e3)
    e4 = Conv2D(filters=nfilters[4], kernel_size=(3,3), padding='same')(e4)
    e4 = BatchNormalization(axis=bnorm_axis)(e4)
    e4 = Activation('relu')(e4)
    e4 = Conv2D(filters=nfilters[4], kernel_size=(3,3), padding='same')(e4)
    e4 = BatchNormalization(axis=bnorm_axis)(e4)
    e4 = Activation('relu')(e4)
    #e4 = MaxPooling2D((2, 2))(e4)
    
    ####################################
    # encoder (expansive path)
    ####################################
    
    #decoder block 3
    d3=UpSampling2D((2, 2),)(e4)
    d3=concatenate([e3,d3], axis=-1)#skip connection
    d3=Conv2DTranspose(nfilters[3], (3, 3), padding='same')(d3)
    d3=BatchNormalization(axis=bnorm_axis)(d3)
    d3=Activation('relu')(d3)
    d3=Conv2DTranspose(nfilters[3], (3, 3), padding='same')(d3)
    d3=BatchNormalization(axis=bnorm_axis)(d3)
    d3=Activation('relu')(d3)
    
    #decoder block 2
    d2=UpSampling2D((2, 2),)(d3)
    d2=concatenate([e2,d2], axis=-1)#skip connection
    d2=Conv2DTranspose(nfilters[2], (3, 3), padding='same')(d2)
    d2=BatchNormalization(axis=bnorm_axis)(d2)
    d2=Activation('relu')(d2)
    d2=Conv2DTranspose(nfilters[2], (3, 3), padding='same')(d2)
    d2=BatchNormalization(axis=bnorm_axis)(d2)
    d2=Activation('relu')(d2)
    
    #decoder block 1
    d1=UpSampling2D((2, 2),)(d2)
    d1=concatenate([e1,d1], axis=-1)#skip connection
    d1=Conv2DTranspose(nfilters[1], (3, 3), padding='same')(d1)
    d1=BatchNormalization(axis=bnorm_axis)(d1)
    d1=Activation('relu')(d1)
    d1=Conv2DTranspose(nfilters[1], (3, 3), padding='same')(d1)
    d1=BatchNormalization(axis=bnorm_axis)(d1)
    d1=Activation('relu')(d1)
    
    #decoder block 0
    d0=UpSampling2D((2, 2),)(d1)
    d0=concatenate([e0,d0], axis=-1)#skip connection
    d0=Conv2DTranspose(nfilters[0], (3, 3), padding='same')(d0)
    d0=BatchNormalization(axis=bnorm_axis)(d0)
    d0=Activation('relu')(d0)
    d0=Conv2DTranspose(nfilters[0], (3, 3), padding='same')(d0)
    d0=BatchNormalization(axis=bnorm_axis)(d0)
    d0=Activation('relu')(d0)
    
    #output
    out_class = Dense(1)(d0)
    out_class = Activation('sigmoid',name='output')(out_class)
    
    #create and compile the model
    model=Model(inputs=input_tensor,outputs=out_class)
    model.compile(loss={'output':'binary_crossentropy'},
                  metrics={'output':'accuracy'},
                  optimizer='adam')
    
    #%%print the model's summary
    print(model.summary())
    
    #%% train the model
    
    filepath = filepath_orig+'_run'+str(run)
    print(filepath)
    checkpoint = ModelCheckpoint('./trained_models/'+filepath+'.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    csvlog = CSVLogger('./trained_models/'+filepath+'_train_log.csv',append=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20)
    batch_size=4
    
    #initialize the generator
    gen_train = aug_generator(X_tr,Y_tr,batch_size=batch_size)
    #split the array and see how many splits there are to determine #steps
    steps_per_epoch_tr = len(np.array_split(np.zeros(len(X_tr)),int(len(X_tr)/batch_size)))
    
    tic=time.time()
    model.fit_generator(generator=gen_train,
                        steps_per_epoch=steps_per_epoch_tr,#the generator internally goes over the entire dataset in one iteration
                        validation_data=(X_val,Y_val),
                        epochs=200,
                        verbose=2,
                        initial_epoch=0,
                        callbacks=[checkpoint, csvlog, early_stopping])
    toc=time.time()
    model.save('./trained_models/'+filepath+'_model.hdf5')
    file = open('./trained_models/'+filepath+'_time.txt','w')
    file.write('training time:'+format(toc-tic, '.2f')+'seconds')
    file.close()
    
    #model.layers[1].get_weights()[0][0][0]