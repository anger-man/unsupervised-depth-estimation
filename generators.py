#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 13:16:12 2021

@author: c

"""

#%%

import os
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from matplotlib.image import imread
from tensorflow_addons.layers import InstanceNormalization, GroupNormalization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

#%%

#helper functions

def resnet_block(f,input_layer,red = False, norm= True):
    shortcut = input_layer
    if red:
        g = Conv2D(f, (3,3), padding='same', strides =(2,2), kernel_initializer='he_normal')(input_layer)
        shortcut = Conv2D(f, (1,1), padding='same', strides=(2,2), kernel_initializer='he_normal')(shortcut)
        if norm:
            shortcut = GroupNormalization(groups=min(f,norm))(shortcut)
    else:
        g = Conv2D(f, (3,3), padding='same', kernel_initializer='he_normal')(input_layer)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)
    g = Conv2D(f, (3,3), padding='same', kernel_initializer='he_normal')(g)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Add()([g, shortcut])
    g = Activation('relu')(g)
    return g


def up_block(f, input_layer, skip, norm = True):
    g = UpSampling2D(size=(2,2),interpolation='nearest')(input_layer)
    g = Conv2D(f, (3,3), strides=(1,1), padding='same')(g)
    if norm:
        g = GroupNormalization(groups=f)(g)
    g = Activation('elu')(g)
    g = Concatenate()([g,skip])
    g = Conv2D(f, (3,3), padding='same',strides=(1,1))(g)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('elu')(g)   
    return g

def unet_enc(f, input_layer, red=True, norm=True):
    g = Conv2D(f, (3,3), padding='same', kernel_initializer='he_normal')(input_layer)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)
    g = Conv2D(f, (3,3), padding='same', kernel_initializer='he_normal')(g)
    if norm:
        g = GroupNormalization(groups=f)(g)
    g = Activation('relu')(g)
    shortcut = g
    if red:
        g = Conv2D(f, (3,3), padding='same', strides=(2,2), kernel_initializer='he_normal')(g)
        if norm:
            g = GroupNormalization(groups=min(f,norm))(g)
        g = Activation('relu')(g)
    return(g,shortcut)

def unet_dec(f, input_layer, skip, short=True, norm = True):
    g = Conv2DTranspose(f, (3,3), strides=(2,2), padding='same', kernel_initializer='he_normal')(input_layer)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)
    if short:
        g = Concatenate()([g,skip])
    g = Conv2D(f, (3,3), padding='same', kernel_initializer='he_normal')(g)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)
    g = Conv2D(f, (3,3), padding='same', kernel_initializer='he_normal')(g)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)
    return(g)
    
def residual_block(f, input_layer,norm=True):
    g = Conv2D(f, (3,3), padding='same',kernel_initializer='he_normal')(input_layer)
    #if norm:
        #g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)
    g = Conv2D(f, (3,3), padding='same',kernel_initializer='he_normal')(g)
    #if norm:
        #g = GroupNormalization(groups=min(f,norm))(g)
    g = Add()([g, input_layer])
    return g


#%%

def define_styletransfer(inp_dim,out_dim,f,norm, out_act = 'tanh'):
    f = int(f); norm = int(norm)

    in_image = Input(shape=inp_dim)
    g = Conv2D(f, (7,7), padding='same', strides=(1,1), kernel_initializer='he_normal')(in_image)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)

    g = Conv2D(2*f, (3,3), strides=(2,2), padding='same',kernel_initializer='he_normal')(g)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)

    g = Conv2D(4*f, (3,3), strides=(2,2), padding='same',kernel_initializer='he_normal')(g)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)

    for kkk in range(9):
        g = residual_block(4*f, g,norm=norm)

    g = Conv2DTranspose(2*f, (3,3), strides=(2,2), padding='same',kernel_initializer='he_normal')(g)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)

    g = Conv2DTranspose(f, (3,3), strides=(2,2), padding='same',kernel_initializer='he_normal')(g)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)

    g = Conv2D(out_dim[-1], (7,7), padding='same', strides=(1,1), kernel_initializer='he_normal')(g)
    out_image = Activation(out_act)(g)

    model = Model(in_image, out_image)
    return model

#%%

def define_unet(inp_dim,out_dim,f,norm, out_act = 'tanh'):
    f=int(f); norm=int(norm)
    
    in_image = Input(shape=inp_dim)
    g, g_0 = unet_enc(f,  in_image, red= True, norm=norm)
    g, g_1 = unet_enc(2*f, g, red= True, norm=norm)
    g, g_2 = unet_enc(4*f, g, red= True, norm=norm)
    g, g_3 = unet_enc(8*f, g, red= True, norm=norm)
    g, foo = unet_enc(16*f, g, red= False, norm=norm)
    
    g = unet_dec(8*f, g, g_3, norm=norm)
    g = unet_dec(4*f, g, g_2, norm=norm)
    #disp1 = Conv2D(out_dim[-1], (3,3), padding='same',strides=(1,1),
                       #activation = out_act)(g)
    #disp1 = UpSampling2D(size=(4,4),interpolation='nearest')(disp1)
    g = unet_dec(2*f, g, g_1, norm=norm)
    #disp2 = Conv2D(out_dim[-1], (3,3), padding='same',strides=(1,1),
                       #activation = out_act)(g)
    #disp2 = UpSampling2D(size=(2,2),interpolation='nearest')(disp2)
    g = unet_dec(1*f, g, g_0, norm=norm)
    out_image = Conv2D(out_dim[-1], (3,3), padding='same',strides=(1,1),
                       activation = out_act, kernel_initializer='he_normal')(g)
        
    model = Model(in_image, out_image)
    return model
    

#%%

def define_resnet18(inp_dim,out_dim,f, norm,out_act = 'tanh'):
    f=int(f);norm=int(norm)
    # taken from Goddard - Digging Into Self-Supervised Monocular Depth Estimation
    # ResNet18 + Decoder with multiscale depth

    in_image = Input(shape=inp_dim)
    g = Conv2D(f,(7,7),padding='same',strides=(2,2), kernel_initializer='he_normal')(in_image)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)
    g_0=g
    g = MaxPooling2D(pool_size=(3,3), strides=(2,2),padding='same')(g)
    
    g = resnet_block(f, g, norm = norm)
    g = resnet_block(f, g, norm = norm)
    g_1 = g
    
    g = resnet_block(2*f, g, red = True, norm = norm)
    g = resnet_block(2*f, g, norm = norm)
    g_2 = g
    
    g = resnet_block(4*f, g, red = True, norm = norm)
    g = resnet_block(4*f, g, norm = norm)
    g_4 = g
    
    g = resnet_block(8*f, g, red = True, norm = norm)
    g = resnet_block(8*f, g, norm = norm)
    
    g = up_block(8*f, g, g_4, norm = norm)
    g = up_block(4*f, g, g_2, norm = norm)
    g = up_block(2*f, g, g_1, norm = norm)
    g = up_block(f, g, g_0, norm = norm)
   
    g = UpSampling2D(size=(2,2),interpolation='nearest')(g)
    g = Conv2D(f//2, (3,3), strides=(1,1), padding='same', kernel_initializer='he_normal')(g)
    if norm:
        g = GroupNormalization(groups = min(f//2,norm))(g)
    g = Activation('elu')(g)
    g = Conv2D(f//2, (3,3), strides=(1,1), padding='same', kernel_initializer='he_normal')(g)
    if norm:
        g = GroupNormalization(groups = min(f//2,norm))(g)
    g = Activation('elu')(g)
    out_image = Conv2D(out_dim[-1], (3,3), padding='same',strides=(1,1),
                       activation = out_act, kernel_initializer='he_normal')(g)
        
    model = Model(inputs = in_image, outputs= out_image)
    return model