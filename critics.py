#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 13:22:56 2021

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

def define_dcgan(in_image,inp_dim,f,norm,k = 4,alpha=0.2):
    f=int(f); norm=int(norm)
    
    # taken from Jung - DEPTH PREDICTION FROM A SINGLE IMAGE WITH...
    # DCGan based on Radford et al, proven to have stability properties! basic DCGAN constraints:
    # use strided convolution
    # use leaky relu with 0.2
    # generator with relu, output with tanh
    # batchnormalization in generator and discriminator
    # no fully connected layers for deep architectures

    in_scaled = in_image

    d = Conv2D(f, (k,k), strides=(2,2), padding='same',kernel_initializer='he_normal')(in_scaled)
    if norm:
        d = GroupNormalization(groups = min(f,norm), scale=False)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(2*f, (k,k), strides=(2,2), padding='same',kernel_initializer='he_normal')(d)
    if norm:
        d = GroupNormalization(groups = min(2*f,norm), scale=False)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(4*f, (k,k), strides=(2,2), padding='same',kernel_initializer='he_normal')(d)
    if norm:
        d = GroupNormalization(groups = min(4*f,norm), scale=False)(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    
    d = Conv2D(8*f, (k,k), strides=(2,2), padding='same',kernel_initializer='he_normal')(d)
    if norm:
        d = GroupNormalization(groups = min(8*f,norm), scale=False)(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(8*f, (k,k), strides=(1,1), padding='same',kernel_initializer='he_normal')(d)
    if norm:
        d = GroupNormalization(groups = min(8*f,norm), scale=False)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(16*f, (k,k), strides=(1,1), padding='valid',kernel_initializer='he_normal')(d)
    if norm:
        d = GroupNormalization(groups = min(16*f,norm), scale=False)(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    patch_out = GlobalAvgPool2D()(d)
    patch_out = Dense(1, activation = 'linear',kernel_initializer='he_normal')(patch_out)
    
    model = Model(in_image, patch_out)
   
    return model

#%%

def define_patchgan(in_image,inp_dim,f,norm,k = 4,alpha=0.2):
    f=int(f); norm=int(norm)
    
    # taken from Jung - DEPTH PREDICTION FROM A SINGLE IMAGE WITH...
    # DCGan based on Radford et al, proven to have stability properties! basic DCGAN constraints:
    # use strided convolution
    # use leaky relu with 0.2
    # generator with relu, output with tanh
    # batchnormalization in generator and discriminator
    # no fully connected layers for deep architectures

    in_scaled =in_image
    d = Conv2D(f, (k,k), strides=(2,2), padding='same',kernel_initializer='he_normal')(in_scaled)
    if norm:
        d = GroupNormalization(groups = min(f,norm), scale=False)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(2*f, (k,k), strides=(2,2), padding='same',kernel_initializer='he_normal')(d)
    if norm:
        d = GroupNormalization(groups = min(2*f,norm), scale=False)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(4*f, (k,k), strides=(2,2), padding='same',kernel_initializer='he_normal')(d)
    if norm:
        d = GroupNormalization(groups = min(4*f,norm), scale=False)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(8*f, (k,k), strides=(2,2), padding='same',kernel_initializer='he_normal')(d)
    if norm:
        d = GroupNormalization(groups = min(8*f,norm), scale=False)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(8*f, (k,k), strides=(1,1), padding='same',kernel_initializer='he_normal')(d)
    if norm:
        d = GroupNormalization(groups = min(8*f,norm), scale=False)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(16*f, (k,k), strides=(1,1), padding='valid',kernel_initializer='he_normal')(d)
    if norm:
        d = GroupNormalization(groups = min(16*f,norm), scale=False)(d)
    d = LeakyReLU(alpha=0.2)(d)
    patch_out = Conv2D(1, (4,4), strides=(1,1), padding='valid',kernel_initializer='he_normal')(d)
    model = Model(in_image, patch_out)
    return model