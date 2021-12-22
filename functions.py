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
        y = ones((len(X), patch_shape[0], patch_shape[1], 1))
    except:
        y = ones((len(X), 1))
    return X, y

def generate_real_samples(dataset, patch_shape):
    X = dataset
    # generate 'real' class labels (1)
    try:
        y = -ones((len(X), patch_shape[0], patch_shape[1], 1))
    except:
        y = -ones((len(X), 1))
    return X, y

def update_image_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			# stock the pool
			pool.append(image)
			selected.append(image)
		elif random() < 0.5:
			# use image, but don't add it to the pool
			selected.append(image)
		else:
			# replace an existing image and use replaced image
			ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return asarray(selected)
        

def plot_curves_gp(dA,dB,gA,gB,alpha,cAval,cBval,criticW,name):
    dA=np.array(dA); dB=np.array(dB)
    
    plt.figure()
    l1 = np.convolve(-dA[30:],np.ones(20)/20,mode='valid')
    a,=plt.plot(l1, linewidth = 1); 
    l2 = np.convolve(-dB[30:],np.ones(20)/20,mode='valid')
    b,=plt.plot(l2, linewidth = 1); 
    l3 = np.array(cAval)
    c,=plt.plot(np.convolve(l3[30:],np.ones(20)/20,mode='valid'), linewidth = .3); 
    l4 = np.array(cBval)
    d,=plt.plot(np.convolve(l4[30:],np.ones(20)/20,mode='valid'), linewidth = .3); 
    plt.legend((a,b,c,d),('W1_C','W1_D','W1_Cval','W1_Dval'))
    plt.title(str('W1_C: %.2f,  W1_D: %.2f' %(np.mean(l1[max(len(l1)-100,0):]),np.mean(l2[max(len(l1)-100,0):]))))
    plt.savefig('losses/%s.pdf' %name,dpi=200)
    
    plt.figure(figsize=(8,4))
    # l1 = loss_weights[0]*np.array(gB)[:,0]
    # a,=plt.plot(np.convolve(l1,np.ones(20)/20,mode='valid')); 
    l2 = np.array(gB)[30:,1]
    b,=plt.plot(np.convolve(l2,np.ones(20)/20,mode='valid')); 
    l3 = alpha*np.array(gB)[30:,2]
    c,=plt.plot(np.convolve(l3,np.ones(20)/20,mode='valid')); 
    l4 = alpha*np.array(gB)[30:,3]
    d,=plt.plot(np.convolve(l4,np.ones(20)/20,mode='valid')); 
    plt.legend((b,c,d),('gen','forw','back'))
    #plt.ylim(-10,20)
    plt.savefig('CtoBlosses/%s.pdf' %name,dpi=200)
    
    plt.figure(figsize=(8,4))
    plt.hist(criticW.reshape(-1), color='firebrick',bins=40,
              weights=np.zeros_like(criticW) + 1. / criticW.size)
    plt.savefig('criticD/%s.pdf' %name,dpi=200)
    

def Scale(x):
    return(x/1.)


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


class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=0.005):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}


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
            x1 = np.load(evaluation['input'][k])[::subs,::subs]
        else:
            x1 = imread(evaluation['input'][k])[::subs,::subs]
        x2 = np.load(evaluation['target'][k])[::subs,::subs]
        
        X1.append(x1)
        X2.append(x2)
    X1=np.array(X1);X2=np.array(X2)
    
    if task == 'rir/':
        preds_B = generator_CtoD.predict((X1-127.5) / 127.5, batch_size=2)
        preds_B = (preds_B +1) * 127.5
        preds_A = generator_DtoC.predict((X2-127.5) / 127.5, batch_size=2)
        preds_A = (preds_A +1) * 127.5
        metric1 = np.mean(np.abs(X2-preds_B))
        metric2 = 10 * math.log10(255**2/np.mean(np.square(X2-preds_B)))
        
    if task == 'sur/':
        preds_B = generator_CtoD.predict((X1-127.5) / 127.5, batch_size=2)
        preds_A = generator_DtoC.predict(X2, batch_size=2)
        preds_A = (preds_A +1) * 127.5
        metric1 = np.sqrt(np.mean(np.square(X2*0.4725-preds_B*0.4725)))
        metric2 = .1-np.mean(np.abs(X2*0.4725-preds_B*0.4725))
    
    # if task in ['bod/']:
    #     temp = X1/255.
    #     preds_B = generator_CtoD.predict(temp, batch_size=2)
    #     temp = np.concatenate((X2,X2!=0),axis=-1)
    #     preds_A = generator_DtoC.predict(temp, batch_size=2)
    #     preds_A *= 255.
    #     metric1 = (np.mean(np.square(X2[...,0]-preds_B[...,0])))**.5
    #     metric2 = (1-np.mean(np.abs(X2[...,0]-preds_B[...,0])))
        
    if task in ['tex/','bod/']:
        temp = (X1-127.5) / 127.5; 
        preds_B = generator_CtoD.predict(temp, batch_size=2)
        preds_B = (preds_B +1) /2
        temp = (X2*2-1); 
        preds_A = generator_DtoC.predict(temp, batch_size=2)
        preds_A = (preds_A +1) * 127.5
        metric1 = (np.mean(np.square(X2-preds_B)))**.5
        metric2 = (1-np.mean(np.abs(X2-preds_B)))
        
        
    if task == 'nyu/':
        preds_B = generator_CtoD.predict((X1-127.5) / 127.5, batch_size=2)
        preds_B = (preds_B +1) *5
        preds_A = generator_DtoC.predict((X2/10-.5)*2, batch_size=2)
        preds_A = (preds_A +1) * 127.5
        metric1 = np.sqrt(np.mean(np.square(X2-preds_B)))
        tmp = np.max(np.concatenate((X2/(preds_B+1e-7),preds_B/(X2+1e-7)),-1),-1)
        metric2 = np.mean(tmp<1.25, axis = (0,1,2))
    
    if task == 'dt4/':
        preds_B = generator_CtoD.predict((X1-127.5) / 127.5, batch_size=2)
        preds_B = (preds_B +1) * (uq-lq)/2 +lq
        tmp = 2 * (X2-lq) / (uq-lq)- 1
        tmp[tmp<-1]=-1; tmp[tmp>1]=1
        X2[X2<lq]=lq; X2[X2>uq]=uq
        preds_A = generator_DtoC.predict(tmp, batch_size=2)
        preds_A = (preds_A +1) * 127.5
        
        
        metric1 = np.sqrt(np.mean(np.square(X2-preds_B)))
        #b_gt = np.reshape(X2,(X2.shape[0],-1)); b_gt = -np.sort(-b_gt, axis = -1)
        #b_gt = b_gt[:,np.linspace(0,b_gt.shape[1]-1,10000).astype(int)]
        #b_es = np.reshape(preds_B,(preds_B.shape[0],-1)); b_es = -np.sort(-b_es, axis = -1)
        #b_es = b_es[:,np.linspace(0,b_es.shape[1]-1,10000).astype(int)]
        metric2 = .5-np.mean(np.abs(X2-preds_B))
    
        
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
        if task == 'rir/':
            ax.imshow(X1[i],vmin=X1[i].min(),vmax=X1[i].max(),cmap='Greys_r')
        else:
            ax.imshow(X1[i])
    for i in range(n_samples):
        ax=fig.add_subplot(4, n_samples, 1 + 3*n_samples + i)
        plt.axis('off')
        if task=='rir/':
            ax.imshow(preds_A[i].astype(np.uint8),vmin=X1[i].min(),vmax=X1[i].max(),cmap='Greys_r')
        else:
            ax.imshow(preds_A[i].astype(np.uint8))
    # save plot to file
    filename ='plots/%s_%04d.png' % (name,(step+1))
    fig.tight_layout(pad=.1)
    fig.suptitle(str('M1 = %.2f, M2 = %.2f' %(metric1,metric2)), color = 'red')
    plt.savefig( filename,dpi=150)

    return(metric1,metric2)