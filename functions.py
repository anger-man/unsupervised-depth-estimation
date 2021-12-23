#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:14:01 2021

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


d0=4.
f = lambda x,y: 1.-np.exp(-((x-128)**2+(y-128)**2)/(2*d0**2))
filt_mat = np.ones((256,256))
for i in range(filt_mat.shape[0]):
    for j in range(filt_mat.shape[1]):
        filt_mat[i,j]= f(i,j)

def MyLossC(inputs):    
    ytrue = inputs[0]; ypred = inputs[1]
    t1 = (ytrue+1.)/2.
    y1 = 0.299*t1[...,0]+0.587*t1[...,1]+0.114*t1[...,2]
    gamma = tf.stop_gradient(-.3*2.303/K.log(K.clip(K.mean(y1,axis=(1,2)),0.032,0.871)))
    res = tf.transpose(tf.transpose(y1+K.epsilon(),(1,2,0))**gamma,(2,0,1))
    ft = tf.signal.fftshift(tf.signal.fft2d(tf.cast(res, tf.complex64)))
    ft = ft * tf.constant(filt_mat,dtype=tf.complex64)
    res_true = K.abs(tf.signal.ifft2d(tf.signal.ifftshift(ft)))
   
    t2 = (ypred+1.)/2.
    y2 = 0.299*t2[...,0]+0.587*t2[...,1]+0.114*t2[...,2]
    gamma = tf.stop_gradient(-.3*2.303/K.log(K.clip(K.mean(y2,axis=(1,2)),0.032,0.871)))
    res = tf.transpose(tf.transpose(y2+K.epsilon(),(1,2,0))**gamma,(2,0,1))
    ft = tf.signal.fftshift(tf.signal.fft2d(tf.cast(res, tf.complex64)))
    ft = ft * tf.constant(filt_mat,dtype=tf.complex64)
    res_pred = K.abs(tf.signal.ifft2d(tf.signal.ifftshift(ft)))
    
    inter = K.mean(res_true,axis=(1,2))

    return(.5*K.mean(K.abs(res_true-res_pred),axis=(1,2))/inter)


def nMAE(l):
    #normalization via inter-length bad results for cycle- and perceptual loss
    true,pred = l
    # true = K.clip(true,0.,1e7)
    # pred = K.clip(pred,0.,1e7)
    inter = K.mean(true,axis=(1,2,3))#-K.min(true,axis=(1,2,3))
    return(K.mean(K.abs(true-pred),axis=(1,2,3))/inter)

def MAE(l):
    true,pred = l
    return K.mean(K.abs(true-pred),axis=(1,2,3))

def MSE(l):
    x1,x2 = l
    return(K.mean(K.square(x1-x2), axis=(1,2,3)))  


def MAEd(l):
    x1,x2 = l
    mult = tf.constant(np.arange(0,20,1),dtype=tf.float32)
    x1 = K.sum(x1* mult, axis=-1)
    x2 = K.sum(x2* mult, axis=-1)
    return(K.mean(K.abs(x1-x2), axis=(1,2)))

    
def MSEd(l):
    x1,x2 = l
    mult = tf.constant(np.arange(0,20,1),dtype=tf.float32)
    x1 = K.sum(x1* mult, axis=-1)
    x2 = K.sum(x2* mult, axis=-1)
    return(K.mean(K.square(x1-x2), axis=(1,2)))

def wasserstein(ytrue,ypred):
    return(K.mean(ytrue*ypred))

def identity(ytrue,ypred):
    return(ypred)
        

def make_grayscale(x):
    tmp = x[...,0]*.299 + x[...,1]*.587 + x[...,2]*.11
    tmp = np.expand_dims(tmp,-1)
    tmp = np.concatenate((tmp,tmp,tmp),axis=-1)
    return tmp

def generate_fake_samples(generator, dataset, patch_shape):
    # generate fake instance
    X = generator.predict(dataset)
    # X = np.concatenate((X[0],X[1],X[2]),axis=0)
    # create 'fake' class labels (0)
    try:
        y = np.ones((len(X), patch_shape[0], patch_shape[1], 1))
    except:
        y = np.ones((len(X), 1))
    return X, y

def generate_real_samples(dataset, patch_shape):
    X = dataset
    # generate 'real' class labels (1)
    try:
        y = -np.ones((len(X), patch_shape[0], patch_shape[1], 1))
    except:
        y = -np.ones((len(X), 1))
    return X, y

def update_image_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			# stock the pool
			pool.append(image)
			selected.append(image)
		elif np.random.uniform() < 0.5:
			# use image, but don't add it to the pool
			selected.append(image)
		else:
			# replace an existing image and use replaced image
			ix = np.random.randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return np.asarray(selected)
        

def plot_curves_gp(dA,dB,gA,gB,cAval,cBval,criticW,name):
    dA=np.array(dA); dB=np.array(dB)
    
    plt.figure()
    l1 = np.convolve(-dA[20:],np.ones(20)/20,mode='valid')
    a,=plt.plot(l1, linewidth = 1); 
    l2 = np.convolve(-dB[20:],np.ones(20)/20,mode='valid')
    b,=plt.plot(l2, linewidth = 1); 
    l3 = np.array(cAval)
    c,=plt.plot(np.convolve(l3[20:],np.ones(20)/20,mode='valid'), linewidth = .3); 
    l4 = np.array(cBval)
    d,=plt.plot(np.convolve(l4[20:],np.ones(20)/20,mode='valid'), linewidth = .3); 
    plt.legend((a,b,c,d),('W1_C','W1_D','W1_Cval','W1_Dval'))
    plt.title(str('rgb_domain: %.2f,  depth_domain: %.2f' %(np.mean(l1[max(len(l1)-100,0):]),np.mean(l2[max(len(l1)-100,0):]))))
    plt.savefig('critic_loss/%s.pdf' %name,dpi=200)
    
    plt.figure(figsize=(8,4))
    # l1 = loss_weights[0]*np.array(gB)[:,0]
    # a,=plt.plot(np.convolve(l1,np.ones(20)/20,mode='valid')); 
    l2 = np.array(gB)[20:,1]
    b,=plt.plot(np.convolve(l2,np.ones(20)/20,mode='valid')); 
    l3 = (np.array(gB)[20:,2]+np.array(gB)[20:,3])
    c,=plt.plot(np.convolve(l3,np.ones(20)/20,mode='valid')); 
    l4 = (np.array(gB)[20:,4]+np.array(gB)[20:,5])
    d,=plt.plot(np.convolve(l4,np.ones(20)/20,mode='valid')); 
    plt.legend((b,c,d),('adversarial loss','image loss','perceptual loss'))
    #plt.ylim(-10,20)
    plt.savefig('generator_loss/%s.pdf' %name,dpi=200)
    
    



def set_trainable(model, value = True):
	for layer in model.layers:
		layer.trainable = value
	pass


class GradientPenalty(Layer):
    def __init__(self, penalty):
        self.penalty = penalty
        super(GradientPenalty, self).__init__()
    def call(self, inputs):
        (y_pred,averaged_samples) = inputs
        gradients = K.gradients(y_pred, averaged_samples)[0]
        norm_grad = K.sqrt(K.sum(K.square(gradients),        axis=[1,2,3]))
        gradient_penalty = self.penalty * K.square(K.clip(norm_grad-1,0.,1e7))
        return (gradient_penalty)

class PixelNormalization(Layer):
	# initialize the layer
    def __init__(self,groups,scale=True, **kwargs):
        self.scale = scale
        self.groups = groups
        super(PixelNormalization, self).__init__(**kwargs)
	# perform the operation
    def call(self, inputs):
		# calculate square pixel values
        values = inputs**2.0
		# calculate the mean pixel values
        mean_values = K.mean(values, axis=-1, keepdims=True)
		# ensure the mean is not zero
        mean_values += 1e-8
		# calculate the sqrt of the mean squared value (L2 norm)
        l2 = K.sqrt(mean_values)
		# normalize values by the l2 norm
        normalized = inputs / l2
        return normalized
	# define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape

#%%

def evall(step, generator_CtoD,generator_DtoC,evaluation,name,task,batch_size):
    X1 = [];X2=[]
    try:
        samples = np.random.choice(range(len(evaluation)),400,replace=False)
    except:
        samples = np.random.permutation(np.arange(len(evaluation)))
    n_samples = 5
    if task=='sur/':
        subs = 1
    else:
        subs = 2
    for k in samples:
        if task in ['rir/','bod/']:
            x1 = np.load(evaluation['input'][k])
        else:
            x1 = imread(evaluation['input'][k])
        x2 = np.load(evaluation['target'][k])
        X1.append(x1)
        X2.append(x2)
    X1=np.array(X1);X2=np.array(X2)
 
        
    temp = (X1-127.5) / 127.5; 
    preds_B = generator_CtoD.predict(temp, batch_size=2)
    preds_A = generator_DtoC.predict(X2, batch_size=2)
    preds_A = (preds_A +1) * 127.5
    metric1 = (np.mean(np.square(X2-preds_B)))**.5
    metric2 = np.mean(np.abs(X2-preds_B))
        
        
    fig = plt.figure(figsize=(5.5,4.5))
    for i in range(n_samples):
        ax=fig.add_subplot(4, n_samples, 1 + i)
        plt.axis('off')
        ax.imshow(X2[i,...],vmin=X2[i].min(),vmax=X2[i].max(),cmap='Greys_r')
    for i in range(n_samples):
        ax=fig.add_subplot(4, n_samples, 1 + n_samples + i)
        plt.axis('off')
        ax.imshow(preds_B[i,...],vmin=X2[i].min(),vmax=X2[i].max(),cmap='Greys_r')
    for i in range(n_samples):
        ax=fig.add_subplot(4, n_samples, 1 + 2*n_samples + i)
        plt.axis('off')
        ax.imshow(X1[i])
    for i in range(n_samples):
        ax=fig.add_subplot(4, n_samples, 1 + 3*n_samples + i)
        plt.axis('off')
        ax.imshow(preds_A[i].astype(np.uint8))
    # save plot to file
    filename ='plots/%s_%04d.png' % (name,(step+1))
    fig.tight_layout(pad=.1)
    fig.suptitle(str('M1 = %.2f, M2 = %.2f' %(metric1,metric2)), color = 'red')
    plt.savefig( filename,dpi=150)

    return(metric1,metric2)