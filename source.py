# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:41:17 2021

@author: Christoph Angermann
"""


import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)
import logging
logging.disable(logging.WARNING)
from random import random
from numpy import zeros
from numpy import ones
from numpy import asarray
import math
from numpy.random import randint
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAvgPool2D
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2D, SeparableConv2D
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, Lambda, ReLU
from tensorflow.keras.layers import LeakyReLU, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from matplotlib.image import imread
from tensorflow_addons.layers import InstanceNormalization, GroupNormalization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import optparse

#%%

# dt median mins
lq = -7.4
# dt median maxs
uq = 1.8

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
        

def plot_curves(dA,dB,gA,gB,loss_weights,dAval,dBval,criticW,name):
    
    plt.figure()
    l1 = np.convolve(np.sum(np.array(dA),axis=-1),np.ones(20)/20,mode='valid')
    a,=plt.plot(l1); 
    l2 = np.convolve(np.sum(np.array(dB),axis=-1),np.ones(20)/20,mode='valid')
    b,=plt.plot(l2); 
    l3 = np.array(dAval)
    c,=plt.plot(np.convolve(l3,np.ones(20)/20,mode='valid')); 
    l4 = np.array(dBval)
    d,=plt.plot(np.convolve(l4,np.ones(20)/20,mode='valid')); 
    plt.legend((a,b,c,d),('W1_C','W1_D','W1_Cval','W1_Dval'))
    plt.title(str('W1_C: %.2f,  W1_D: %.2f' %(np.mean(l1[max(len(l1)-100,0):]),np.mean(l2[max(len(l1)-100,0):]))))
    plt.savefig('losses/%s.pdf' %name,dpi=200)
    # np.save(options.direc +'losses.npy',[l1,l2,l3,l4])
    
    plt.figure(figsize=(8,4))
    l1 = loss_weights[0]*np.array(dB)[:,0]
    a,=plt.plot(np.convolve(l1,np.ones(20)/20,mode='valid')); 
    l2 = loss_weights[0]*np.array(dB)[:,1]
    b,=plt.plot(np.convolve(l2,np.ones(20)/20,mode='valid')); 
    l3 = loss_weights[1]*np.array(gB)[:,2]
    c,=plt.plot(np.convolve(l3,np.ones(20)/20,mode='valid')); 
    l4 = loss_weights[2]*np.array(gB)[:,3]
    d,=plt.plot(np.convolve(l4,np.ones(20)/20,mode='valid')); 
    plt.legend((a,b,c,d),('c_real','c_fake','forw','back'))
    plt.savefig('CtoBlosses/%s.pdf' %name,dpi=200)
    
    plt.figure(figsize=(8,4))
    plt.hist(criticW.reshape(-1), color='firebrick',bins=40,
              weights=np.zeros_like(criticW) + 1. / criticW.size)
    plt.savefig('criticD/%s.pdf' %name,dpi=200)
    
    # plt.figure(figsize=(8,4))
    # plt.hist(critic_B.reshape(-1), color='firebrick',bins=40,
    #          weights=np.zeros_like(critic_B) + 1. / critic_B.size)
    # plt.savefig(options.direc + 'criticB_wDist.pdf',dpi=200)
    
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

#%%
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


def resnet_block2(f,input_layer,num_blocks,red = False, norm= True):
    #init = RandomNormal(stddev = 0.01)
    short = input_layer  
    if red:
        short = Conv2D(4*f, (1,1), padding='same',strides=(2,2))(short)
    else:
        short = Conv2D(4*f, (1,1), padding='same')(short)
    if norm:
        short = GroupNormalization(groups=min(4*f,norm))(short)
        
    if red:
        g = Conv2D(f, (1,1), padding='same',strides=(2,2))(input_layer)
    else:
        g = Conv2D(f, (1,1), padding='same')(input_layer)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)
    
    g = Conv2D(f, (3,3), padding='same')(g)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)
    
    g = Conv2D(4*f, (1,1), padding='same')(g)
    if norm:
        g = GroupNormalization(groups=min(4*f,norm))(g)
        
    g = Add()([short,g])
    g = Activation('relu')(g); short = g
    
    for j in range(num_blocks-1):
    
        g = Conv2D(f, (1,1), padding='same')(g)
        if norm:
            g = GroupNormalization(groups=min(f,norm))(g)
        g = Activation('relu')(g)
        g = Conv2D(f, (3,3), padding='same')(g)
        if norm:
            g = GroupNormalization(groups=min(f,norm))(g)
        g = Activation('relu')(g)
        g = Conv2D(4*f, (1,1), padding='same')(g)
        if norm:
            g = GroupNormalization(groups=min(4*f,norm))(g)
        g = Add()([short,g])
        g = Activation('relu')(g); short = g

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

def define_resnet18_supervision(inp_dim,out_dim,f, norm,out_act = 'tanh'):
    f=int(f);norm=int(norm)
    # taken from Goddard - Digging Into Self-Supervised Monocular Depth Estimation
    # ResNet18 + Decoder with multiscale depth

    in_image = Input(shape=inp_dim)
    g = Conv2D(f,(7,7),padding='same',strides=(2,2))(in_image)
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
    
    g = up_block(4*f, g, g_4, norm = norm)
    
    g = up_block(2*f, g, g_2, norm = norm)
    disp = Conv2D(out_dim[-1], (3,3), padding='same',strides=(1,1),
                       activation = out_act)(g)
    disp2 = UpSampling2D(size=(8,8),interpolation='nearest')(disp)
    
    g = up_block(f, g, g_1, norm = norm)
    disp = Conv2D(out_dim[-1], (3,3), padding='same',strides=(1,1),
                       activation = out_act)(g)
    disp1 = UpSampling2D(size=(4,4),interpolation='nearest')(disp)
    
    g = up_block(f, g, g_0, norm = norm)
    disp = Conv2D(out_dim[-1], (3,3), padding='same',strides=(1,1),
                       activation = out_act)(g)
    disp0 = UpSampling2D(size=(2,2),interpolation='nearest')(disp)
    
    g = Conv2D(f//2, (4,4), strides=(1,1), padding='same')(g)
    if norm:
        g = GroupNormalization(groups = min(f//2,norm))(g)
    g = Activation('elu')(g)
    g = UpSampling2D(size=(2,2),interpolation='nearest')(g)
    g = Concatenate()([g,disp0,disp1,disp2])
    g = Conv2D(f//2, (3,3), strides=(1,1), padding='same')(g)
    if norm:
        g = GroupNormalization(groups = min(f//2,norm))(g)
    g = Activation('elu')(g)
    out_image = Conv2D(out_dim[-1], (3,3), padding='same',strides=(1,1),
                       activation = out_act)(g)
        
    model = Model(in_image, out_image)
    return model
    
def define_resnet50(inp_dim,out_dim,f, norm,out_act = 'tanh'):
    f=int(f);norm=int(norm)
    # taken from Goddard - Digging Into Self-Supervised Monocular Depth Estimation
    # ResNet18 + Decoder with multiscale depth

    in_image = Input(shape=inp_dim)
    g = Conv2D(f,(7,7),padding='same',strides=(2,2))(in_image)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)
    g_0=g
    g = MaxPooling2D(pool_size=(3,3), strides=(2,2),padding='same')(g)
    
    g = resnet_block2(f, g, num_blocks=3, norm = norm)
    g_1 = g
    g = resnet_block2(2*f, g, num_blocks=4, red = True, norm = norm)
    g_2 = g
    g = resnet_block2(4*f, g, num_blocks=6, red = True, norm = norm)
    g_4=g
    g = resnet_block2(8*f, g, num_blocks=3, red = True, norm = norm)
   
    g = up_block(8*f, g, g_4, norm = norm)
    g = up_block(4*f, g, g_2, norm = norm)
    g = up_block(2*f, g, g_1, norm = norm)
    g = up_block(f, g, g_0, norm = norm)
    g = Conv2D(f//2, (4,4), strides=(1,1), padding='same')(g)
    if norm:
        g = GroupNormalization(groups = min(f//2,norm))(g)
    g = Activation('relu')(g)
    g = UpSampling2D(size=(2,2),interpolation='nearest')(g)
    g = Conv2D(f//2, (3,3), strides=(1,1), padding='same')(g)
    if norm:
        g = GroupNormalization(groups = min(f//2,norm))(g)
    g = Activation('relu')(g)
    out_image = Conv2D(out_dim[-1], (3,3), padding='same',strides=(1,1),
                       activation = out_act)(g)
    
    model = Model(inputs = in_image, outputs= out_image)
    return model
    


#%%

def define_discriminator(in_image,inp_dim,f,norm,K,drop_rate = 0.1,alpha=0.2,k=4):
    f=int(f); norm=int(norm)
    
    # taken from Jung - DEPTH PREDICTION FROM A SINGLE IMAGE WITH...
    # DCGan based on Radford et al, proven to have stability properties! basic DCGAN constraints:
    # use strided convolution
    # use leaky relu with 0.2
    # generator with relu, output with tanh
    # batchnormalization in generator and discriminator
    # no fully connected layers for deep architectures
    init = RandomNormal(stddev=.5*K)
    w_clip = WeightClip(K)

    d = Conv2D(f, (k,k), strides=(2,2), padding='same',kernel_initializer=init,
               kernel_constraint=w_clip, bias_constraint=w_clip)(in_image)
    if norm:
        d = GroupNormalization(groups = min(f,norm), scale=False, 
                              beta_constraint=w_clip)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Dropout(drop_rate)(d)

    d = Conv2D(2*f,  (k,k), strides=(2,2), padding='same',kernel_initializer=init,
               kernel_constraint=w_clip, bias_constraint=w_clip)(d)
    if norm:
        d = GroupNormalization(groups = min(2*f,norm), scale=False, 
                              beta_constraint=w_clip)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Dropout(drop_rate)(d)

    d = Conv2D(4*f,  (k,k), strides=(2,2), padding='same',kernel_initializer=init,
               kernel_constraint=w_clip, bias_constraint=w_clip)(d)
    if norm:
        d = GroupNormalization(groups = min(4*f,norm), scale=False, 
                              beta_constraint=w_clip)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Dropout(drop_rate)(d)
    

    d = Conv2D(8*f,  (k,k), strides=(2,2), padding='same',kernel_initializer=init,
               kernel_constraint=w_clip, bias_constraint=w_clip)(d)
    if norm:
        d = GroupNormalization(groups = min(8*f,norm), scale=False, 
                              beta_constraint=w_clip)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Dropout(drop_rate)(d)
    
    d = Conv2D(8*f,  (k,k), strides=(1,1), padding='valid',kernel_initializer=init,
                kernel_constraint=w_clip, bias_constraint=w_clip)(d)
    if norm:
        d = GroupNormalization(groups = min(8*f,norm), scale=False, 
                              beta_constraint=w_clip)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Dropout(drop_rate)(d)
    
    d = GlobalAvgPool2D()(d)
    patch_out = Dense(1,activation='linear',kernel_constraint = w_clip, bias_constraint = w_clip)(d)
    
    #patch_out = Conv2D(1,  (k,k), strides=(1,1), padding='valid',kernel_initializer=init,
               #kernel_constraint=w_clip, bias_constraint=w_clip)(d)
   
	# define model
    model = Model(in_image, patch_out)
   
    return model


def define_discriminator_gp(in_image,inp_dim,f,norm,k = 4,alpha=0.2):
    f=int(f); norm=int(norm)
    
    # taken from Jung - DEPTH PREDICTION FROM A SINGLE IMAGE WITH...
    # DCGan based on Radford et al, proven to have stability properties! basic DCGAN constraints:
    # use strided convolution
    # use leaky relu with 0.2
    # generator with relu, output with tanh
    # batchnormalization in generator and discriminator
    # no fully connected layers for deep architectures

    in_scaled = Lambda(Scale)(in_image)

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


#def define_discriminator_gp(in_image,inp_dim,f,norm,k = 4,alpha=0.2):
    #f=int(f); norm=int(norm)
    
    ## taken from Jung - DEPTH PREDICTION FROM A SINGLE IMAGE WITH...
    ## DCGan based on Radford et al, proven to have stability properties! basic DCGAN constraints:
    ## use strided convolution
    ## use leaky relu with 0.2
    ## generator with relu, output with tanh
    ## batchnormalization in generator and discriminator
    ## no fully connected layers for deep architectures

    #in_scaled = Lambda(Scale)(in_image)

    #d = Conv2D(f, (k,k), strides=(2,2), padding='same',kernel_initializer='he_normal')(in_scaled)
    #if norm:
        #d = GroupNormalization(groups = min(f,norm), scale=False)(d)
    #d = LeakyReLU(alpha=0.2)(d)

    #d = Conv2D(2*f, (k,k), strides=(2,2), padding='same',kernel_initializer='he_normal')(d)
    #if norm:
        #d = GroupNormalization(groups = min(2*f,norm), scale=False)(d)
    #d = LeakyReLU(alpha=0.2)(d)

    #d = Conv2D(4*f, (k,k), strides=(2,2), padding='same',kernel_initializer='he_normal')(d)
    #if norm:
        #d = GroupNormalization(groups = min(4*f,norm), scale=False)(d)
    #d = LeakyReLU(alpha=0.2)(d)
    
    #d = Conv2D(8*f, (k,k), strides=(2,2), padding='same',kernel_initializer='he_normal')(d)
    #if norm:
        #d = GroupNormalization(groups = min(8*f,norm), scale=False)(d)
    #d = LeakyReLU(alpha=0.2)(d)
    
    #d = Conv2D(8*f, (k,k), strides=(1,1), padding='same',kernel_initializer='he_normal')(d)
    #if norm:
        #d = GroupNormalization(groups = min(8*f,norm), scale=False)(d)
    #d = LeakyReLU(alpha=0.2)(d)

    #d = Conv2D(16*f, (k,k), strides=(1,1), padding='valid',kernel_initializer='he_normal')(d)
    #if norm:
        #d = GroupNormalization(groups = min(16*f,norm), scale=False)(d)
    #d = LeakyReLU(alpha=0.2)(d)
    
    #patch_out = Conv2D(1, (4,4), strides=(1,1), padding='valid',kernel_initializer='he_normal')(d)
    
    
    #model = Model(in_image, patch_out)
   
    #return model
#%%

d0=4.
f = lambda x,y: 1.-np.exp(-((x-128)**2+(y-128)**2)/(2*d0**2))
filt_mat = np.ones((256,256))
for i in range(filt_mat.shape[0]):
    for j in range(filt_mat.shape[1]):
        filt_mat[i,j]= f(i,j)
        
def MyLossC(inputs):    
    #unfortunately, tensorflow histogram has no gradient implementation
    # values_range = tf.constant([0., 255.], dtype = tf.float32)
    # histogram = tf.histogram_fixed_width(tf.cast(y1,tf.float32), values_range, 256)
    # # histogram = tf.cast(histogram, tf.float32)
    # cdf = tf.cumsum(histogram)
    # cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]
    # img_shape = tf.shape(y1)
    # pix_cnt = img_shape[-3] * img_shape[-2]
    # px_map = tf.cast(cdf - cdf_min, tf.float32) * 255. / tf.cast(pix_cnt - 1, tf.float32)
    # px_map = tf.cast(px_map, tf.int32)
    # eq_hist1 = tf.expand_dims(tf.gather_nd(px_map,  tf.cast(y1, tf.int32)), 2)
    # eqh1.append(eq_hist1)

    
    ytrue = inputs[0]; ypred = inputs[1]
    # for f in os.listdir('input_sur'):
    #     ytrue = imread('input_sur/'+f)/127.5-1.
    #     ytrue = tf.expand_dims(ytrue,0)
    #     if ytrue.shape[1]!=256:
    #         ytrue=ytrue[:,::2,::2]
    t1 = (ytrue+1.)/2.
    y1 = 0.299*t1[...,0]+0.587*t1[...,1]+0.114*t1[...,2]
    gamma = tf.stop_gradient(-.3*2.303/K.log(K.clip(K.mean(y1,axis=(1,2)),0.032,0.871)))
    res = tf.transpose(tf.transpose(y1+K.epsilon(),(1,2,0))**gamma,(2,0,1))
    ft = tf.signal.fftshift(tf.signal.fft2d(tf.cast(res, tf.complex64)))
    ft = ft * tf.constant(filt_mat,dtype=tf.complex64)
    res_true = K.abs(tf.signal.ifft2d(tf.signal.ifftshift(ft)))
    # plt.imshow(res[0]);plt.show()
    # print(np.mean(res))
    # rad_true =[]
    # rad_true.append(K.sum(res,axis=1))
    # rad_true.append(K.sum(res,axis=2))
    # res = tfa.image.rotate(tf.expand_dims(res,-1),np.pi/4,interpolation='nearest')[...,0]
    # rad_true.append(K.sum(res,axis=1))
    # rad_true.append(K.sum(res,axis=2))
    # rad_true = tf.stack(rad_true,axis=-1)
   
    t2 = (ypred+1.)/2.
    y2 = 0.299*t2[...,0]+0.587*t2[...,1]+0.114*t2[...,2]
    gamma = tf.stop_gradient(-.3*2.303/K.log(K.clip(K.mean(y2,axis=(1,2)),0.032,0.871)))
    res = tf.transpose(tf.transpose(y2+K.epsilon(),(1,2,0))**gamma,(2,0,1))
    ft = tf.signal.fftshift(tf.signal.fft2d(tf.cast(res, tf.complex64)))
    ft = ft * tf.constant(filt_mat,dtype=tf.complex64)
    res_pred = K.abs(tf.signal.ifft2d(tf.signal.ifftshift(ft)))
    
    inter = K.mean(res_true,axis=(1,2))

    return(.5*K.mean(K.abs(res_true-res_pred),axis=(1,2))/inter)




def MyLossD(inputs):     
    ytrue = inputs[0]; ypred = inputs[1]
    t1 = (ytrue+1.)/2.
    y1 = t1[...,0]
    gamma = tf.stop_gradient(-.3*2.303/K.log(K.clip(K.mean(y1,axis=(1,2)),0.032,0.871)))
    res = tf.transpose(tf.transpose(y1+K.epsilon(),(1,2,0))**gamma,(2,0,1))
    rad_true =[]
    rad_true.append(K.sum(res,axis=1))
    rad_true.append(K.sum(res,axis=2))
    res = tfa.image.rotate(tf.expand_dims(res,-1),np.pi/4,interpolation='nearest')[...,0]
    rad_true.append(K.sum(res,axis=1))
    rad_true.append(K.sum(res,axis=2))
    rad_true = tf.stack(rad_true,axis=-1)
   
    t2 = (ypred+1.)/2.
    y2 = t2[...,0]
    gamma = tf.stop_gradient(-.3*2.303/K.log(K.clip(K.mean(y2,axis=(1,2)),0.032,0.871)))
    res = tf.transpose(tf.transpose(y2+K.epsilon(),(1,2,0))**gamma,(2,0,1))
    rad_pred =[]
    rad_pred.append(K.sum(res,axis=1))
    rad_pred.append(K.sum(res,axis=2))
    res = tfa.image.rotate(tf.expand_dims(res,-1),np.pi/4,interpolation='nearest')[...,0]
    rad_pred.append(K.sum(res,axis=1))
    rad_pred.append(K.sum(res,axis=2))
    rad_pred = tf.stack(rad_pred,axis=-1)
    
    inter = K.mean(rad_true,axis=(1,2))
  
    return(K.mean(K.abs(rad_true-rad_pred),axis=(1,2))/inter)


#%%

def myAverage(x):
    return (K.mean(x, axis = (1,2)))


class FrobReg(regularizers.Regularizer):
    def __init__(self, xdim, ydim, filter_size = 5):
        self.xdim = xdim
        self.ydim = ydim
        self.filter_size = filter_size

    def __call__(self, x):
        xdim_out = tf.math.floor((self.xdim-self.filter_size)/2.)+1.
        ydim_out = tf.math.floor((self.ydim-self.filter_size)/2.)+1.
        
        result = K.sqrt(xdim_out*ydim_out*K.sum(K.square(x)))
        result = K.clip(result-2,0.,1e7)
        return(result)

def calculate_spectralbound(w, dense = False):
    
    if dense:
        inf_norm = np.linalg.norm(w.transpose((1,0)),ord=np.inf)
        one_norm = np.linalg.norm(w.transpose((1,0)),ord=1)
        
    else:
        shape = w.shape
        inf_norm = np.linalg.norm(w.transpose((3,0,1,2)).reshape(shape[-1],shape[0]*shape[1]*shape[2]),
                                  ord=np.inf)
        # one_norm = np.linalg.norm(w.transpose((3,0,1,2)).reshape(shape[-1],shape[0]*shape[1]*shape[2]),
        #                           ord=1)
        one_norm = np.max(np.sum(np.abs(w), axis=(0,1,3))*.75)
    return(np.sqrt(inf_norm*one_norm))


def sndense_regularizer(weights):
    weights = tf.transpose(weights, (1,0))
    sigma = K.max(tf.linalg.svd(weights,compute_uv = False))
    pen = K.clip(sigma+1e-7-1.,0,1e7)
    return(1000*pen)
    


class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, norm, lipschitz, **kwargs):
        self.norm = norm
        self.lipschitz = lipschitz
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)
        self.w = self.layer.kernel
        super(SpectralNormalization, self).build()

    def call(self, inputs):
        self.update_weights()
        output = self.layer(inputs)
        self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output
    
    def update_weights(self):
            self.layer.kernel.assign(
               self.w / K.clip(self.norm/self.lipschitz,1,1e7)**(1/6))

    def restore_weights(self):
        self.layer.kernel.assign(self.w)
    

def sb_regularizer(weights,norm=1,lipschitz=1):
    if norm > lipschitz:
        inf_norm = tf.norm(weights, ord=np.inf,axis=[0,1])
        one_norm = K.max(K.sum(K.abs(weights),axis=(0,1,3))*.75)
        wnorm = K.sqrt(inf_norm*one_norm)
        return(K.clip(wnorm-2.,0,1e7))
    else:
        return(0)
        


    


#%%

# def define_discriminator_gp(inp_dim,f,norm,drop_rate,alpha=0.2):
    
#     norm=int(norm)

#     def sep_bn(x, filters, kernel_size, strides=1):
#         x = SeparableConv2D(filters=filters, 
#                             kernel_size = kernel_size, 
#                             strides=strides, 
#                             padding = 'same', 
#                             use_bias = False)(x)
#         return x
    
#     # taken from Jung - DEPTH PREDICTION FROM A SINGLE IMAGE WITH...
#     # DCGan based on Radford et al, proven to have stability properties! basic DCGAN constraints:
#     # use strided convolution
#     # use leaky relu with 0.2
#     # generator with relu, output with tanh
#     # batchnormalization in generator and discriminator
#     # no fully connected layers for deep architectures
                    
    
#     in_image = Input(shape=inp_dim)
#     in_scaled = Lambda(Scale)(in_image)
   
#     x = sep_bn(in_scaled, filters =32, kernel_size =3, strides=2)
#     if norm:
#         x = GroupNormalization(groups=min(32,norm), scale = False)(x)
#     x = ReLU()(x)
    
#     x = sep_bn(x, filters =64, kernel_size =3, strides=1)
#     if norm:
#         x = GroupNormalization(groups=min(64,norm), scale = False)(x)
#     x = ReLU()(x)
    
#     x = sep_bn(x, filters =128, kernel_size =3, strides=2)
#     if norm:
#         x = GroupNormalization(groups=min(128,norm), scale = False)(x)
#     x = ReLU()(x)
    
#     x = sep_bn(x, filters =128, kernel_size =3, strides=1)
#     if norm:
#         x = GroupNormalization(groups=min(128,norm), scale = False)(x)
#     x = ReLU()(x)
    
#     x = sep_bn(x, filters =256, kernel_size =3, strides=2)
#     if norm:
#         x = GroupNormalization(groups=min(256,norm), scale = False)(x)
#     x = ReLU()(x)
    
#     x = sep_bn(x, filters =256, kernel_size =3, strides=1)
#     if norm:
#         x = GroupNormalization(groups=min(256,norm), scale = False)(x)
#     x = ReLU()(x)
    
#     x = sep_bn(x, filters =512, kernel_size =3, strides=2)
#     if norm:
#         x = GroupNormalization(groups=min(512,norm), scale = False)(x)
#     x = ReLU()(x)
    
#     for i in range(5):
#         x = sep_bn(x, filters =512, kernel_size =3, strides=1)
#         if norm:
#             x = GroupNormalization(groups=min(512,norm), scale = False)(x)
#         x = ReLU()(x)
        
#     x = sep_bn(x, filters =1024, kernel_size =3, strides=2)
#     if norm:
#         x = GroupNormalization(groups=min(1024,norm), scale = False)(x)
#     x = ReLU()(x)
    
#     x = GlobalAvgPool2D()(x)
#     output = Dense (units = 1, activation = 'linear')(x)
    
#     model = Model (inputs=in_image, outputs=output)
#     return model







#def define_discriminator_gp(in_image,inp_dim,f,norm,k = 4,alpha=0.2):
    #f=int(f); norm=int(norm)
    
    ## taken from Jung - DEPTH PREDICTION FROM A SINGLE IMAGE WITH...
    ## DCGan based on Radford et al, proven to have stability properties! basic DCGAN constraints:
    ## use strided convolution
    ## use leaky relu with 0.2
    ## generator with relu, output with tanh
    ## batchnormalization in generator and discriminator
    ## no fully connected layers for deep architectures

    #init = RandomNormal(stddev=.005)
    #in_scaled = Lambda(Scale)(in_image)

    #d = Conv2D(f, (k,k), strides=(2,2), padding='same',kernel_initializer=init)(in_scaled)
    #if norm:
        #d = GroupNormalization(groups = min(f,norm), scale=False)(d)
    #d = LeakyReLU(alpha=0.2)(d)

    #d = Conv2D(2*f, (k,k), strides=(2,2), padding='same',kernel_initializer=init)(d)
    #if norm:
        #d = GroupNormalization(groups = min(2*f,norm), scale=False)(d)
    #d = LeakyReLU(alpha=0.2)(d)

    #d = Conv2D(4*f, (k,k), strides=(2,2), padding='same',kernel_initializer=init)(d)
    #if norm:
        #d = GroupNormalization(groups = min(4*f,norm), scale=False)(d)
    #d = LeakyReLU(alpha=0.2)(d)
    
    #d = Conv2D(8*f, (k,k), strides=(2,2), padding='same',kernel_initializer=init)(d)
    #if norm:
        #d = GroupNormalization(groups = min(8*f,norm), scale=False)(d)
    #d = LeakyReLU(alpha=0.2)(d)

    #d = Conv2D(8*f, (k,k), strides=(1,1), padding='same',kernel_initializer=init)(d)
    #if norm:
        #d = GroupNormalization(groups = min(8*f,norm), scale=False)(d)
    #d = LeakyReLU(alpha=0.2)(d)
    
    #patch_out = Conv2D(1, (4,4), strides=(1,1), padding='valid',kernel_initializer=init)(d)
    
    #model = Model(in_image, patch_out)
   
    #return model



#def define_discriminator_gp(in_image,inp_dim,f,norm,k = 4,alpha=0.2):
    #f=int(f); norm=int(norm)
    
    ## taken from Jung - DEPTH PREDICTION FROM A SINGLE IMAGE WITH...
    ## DCGan based on Radford et al, proven to have stability properties! basic DCGAN constraints:
    ## use strided convolution
    ## use leaky relu with 0.2
    ## generator with relu, output with tanh
    ## batchnormalization in generator and discriminator
    ## no fully connected layers for deep architectures

    
    #in_scaled = Lambda(Scale)(in_image)

    #d = resnet_block(64, in_scaled, red=True, norm = norm)
    #d = resnet_block(128, d, red=True, norm = norm)
    #d = resnet_block(256, d, red=True, norm = norm)
    #d = resnet_block(512, d, red=True, norm = norm)
    
    #patch_out = GlobalAvgPool2D()(d)
    #patch_out = Dense(1, activation = 'linear')(patch_out)
    
    #model = Model(in_image, patch_out)
   
    #return model




    

# def define_discriminator_gp(inp_dim,f,norm,drop_rate,lipnorm,lipschitz,alpha=0.2):
#     f=int(f); norm=int(norm)
    
#     # taken from Jung - DEPTH PREDICTION FROM A SINGLE IMAGE WITH...
#     # DCGan based on Radford et al, proven to have stability properties! basic DCGAN constraints:
#     # use strided convolution
#     # use leaky relu with 0.2
#     # generator with relu, output with tanh
#     # batchnormalization in generator and discriminator
#     # no fully connected layers for deep architectures
    
#     init = RandomNormal(stddev=.001)
   
#     def init(shape, dtype=None):
#         w = tf.random.normal(shape, stddev=.1)
#         try:
#             tmp = tf.transpose(w, (3,0,1,2))
#             tmp_shape = tf.shape(tmp)
#             tmp = tf.reshape(tmp,(tmp_shape[0],tmp_shape[1]*tmp_shape[2]*tmp_shape[3]))
#             inf_norm = tf.norm(tmp,ord=np.inf,axis=[0,1])
#             one_norm = K.max(K.sum(K.abs(w), axis=(0,1,3))*.75)
#         except:
#             inf_norm = tf.norm(tf.transpose(w,(1,0)),ord=np.inf,axis=[0,1])
#             one_norm = tf.norm(tf.transpose(w,(1,0)),ord=1,axis=[0,1])
            
#         norm = K.sqrt(inf_norm * one_norm)
#         return (w/norm)*1.5
    
    
#     in_image = Input(shape=inp_dim)
#     d = SpectralNormalization(
#         Conv2D(f, (5,5), strides=(2,2), padding='same',kernel_initializer=init, use_bias = True),
#         norm=lipnorm,lipschitz=lipschitz)(in_image)
#     d = LeakyReLU(alpha)(d)


#     d = SpectralNormalization(
#         Conv2D(2*f, (5,5), strides=(2,2), padding='same',kernel_initializer=init, use_bias = True),
#         norm=lipnorm,lipschitz=lipschitz)(d)
#     d = LeakyReLU(alpha)(d)


#     d = SpectralNormalization(
#         Conv2D(4*f, (5,5), strides=(2,2), padding='same',kernel_initializer=init, use_bias = True),
#         norm=lipnorm,lipschitz=lipschitz)(d)
#     d = LeakyReLU(alpha)(d)


#     d = SpectralNormalization(
#         Conv2D(8*f, (5,5), strides=(2,2), padding='same',kernel_initializer=init, use_bias = True),
#         norm=lipnorm,lipschitz=lipschitz)(d)
#     d = LeakyReLU(alpha)(d)
    
    
#     d = SpectralNormalization(
#         Conv2D(8*f, (5,5), strides=(1,1), padding='same',kernel_initializer=init, use_bias = True),
#         norm=lipnorm,lipschitz=lipschitz)(d)
#     d = LeakyReLU(alpha)(d)
    
    
#     # patch_out = SpectralNormalization(
#     #     Conv2D(1, (5,5), strides=(1,1), padding='valid',kernel_initializer=init, use_bias = False),
#     #     norm=lipnorm,lipschitz=lipschitz)(d)
    
#     d = Flatten()(d)
#     patch_out = SpectralNormalization(Dense(1,activation='linear',kernel_initializer=init,use_bias = False),
#                                       norm=lipnorm, lipschitz=lipschitz)(d)
#  	# define model
#     model = Model(in_image, patch_out)
   
#     return model


#def myAverage(x):
    #return (K.mean(x, axis = (1,2)))




#def define_discriminator_gp(inp_dim,f,norm,drop_rate,lipnorm,lipschitz,alpha=0.2):
    #f=int(f); norm=int(norm)
    
    ## taken from Jung - DEPTH PREDICTION FROM A SINGLE IMAGE WITH...
    ## DCGan based on Radford et al, proven to have stability properties! basic DCGAN constraints:
    ## use strided convolution
    ## use leaky relu with 0.2
    ## generator with relu, output with tanh
    ## batchnormalization in generator and discriminator
    ## no fully connected layers for deep architectures
    #init = RandomNormal(stddev=.001)
    
    
    #in_image = Input(shape=inp_dim)
    #d = SpectralNormalization(
        #Conv2D(128, (4,4), strides=(2,2), padding='same',kernel_initializer=init, use_bias = True),
        #norm=lipnorm,lipschitz=lipschitz)(in_image)
    #tmp = d
    
    #d = Activation('relu')(d)
    #d = SpectralNormalization(
        #Conv2D(128, (4,4), strides=(1,1), padding='same',kernel_initializer=init, use_bias = True),
        #norm=lipnorm,lipschitz=lipschitz)(d)
    #d = Activation('relu')(d)
    #d = SpectralNormalization(
        #Conv2D(128, (4,4), strides=(1,1), padding='same',kernel_initializer=init, use_bias = True),
        #norm=lipnorm,lipschitz=lipschitz)(d)
    #d = Add()([tmp,d])
    #d = Activation('relu')(d)
    
    
    #d = SpectralNormalization(
        #Conv2D(128, (4,4), strides=(2,2), padding='same',kernel_initializer=init, use_bias = True),
        #norm=lipnorm,lipschitz=lipschitz)(d)
    #tmp = d
    
    #d = Activation('relu')(d)
    #d = SpectralNormalization(
        #Conv2D(128, (4,4), strides=(1,1), padding='same',kernel_initializer=init, use_bias = True),
        #norm=lipnorm,lipschitz=lipschitz)(d)
    #d = Activation('relu')(d)
    #d = SpectralNormalization(
        #Conv2D(128, (4,4), strides=(1,1), padding='same',kernel_initializer=init, use_bias = True),
        #norm=lipnorm,lipschitz=lipschitz)(d)
    #d = Add()([tmp,d])
    #d = Activation('relu')(d)
    
    #d = SpectralNormalization(
        #Conv2D(128, (4,4), strides=(2,2), padding='same',kernel_initializer=init, use_bias = True),
        #norm=lipnorm,lipschitz=lipschitz)(d)
    #tmp = d
    
    #d = Activation('relu')(d)
    #d = SpectralNormalization(
        #Conv2D(128, (4,4), strides=(1,1), padding='same',kernel_initializer=init, use_bias = True),
        #norm=lipnorm,lipschitz=lipschitz)(d)
    #d = Activation('relu')(d)
    #d = SpectralNormalization(
        #Conv2D(128, (4,4), strides=(1,1), padding='same',kernel_initializer=init, use_bias = True),
        #norm=lipnorm,lipschitz=lipschitz)(d)
    #d = Add()([tmp,d])
    #d = Activation('relu')(d)
    
    #d = Flatten()(d)
    
    ## patch_out = SpectralNormalization(
    ##     Conv2D(1, (5,5), strides=(1,1), padding='valid',kernel_initializer=init, use_bias = False),
    ##     norm=lipnorm,lipschitz=lipschitz)(d)
    
    #patch_out = SpectralNormalization(Dense(1,activation='linear',kernel_initializer=init,use_bias = False),
                                      #norm=lipnorm, lipschitz=lipschitz)(d)
 	## define model
    #model = Model(in_image, patch_out)
   
    #return model
    


# def define_transformer(inp_dim,out_dim,f, norm,out_act = 'tanh'):
#     f=int(f);norm=int(norm)
#     # taken from Kwak 2020

#     in_image = Input(shape=inp_dim)
#     x = Conv2D(32,(3,3),padding='same')(in_image)
#     x = BatchNormalization()(x)
#     x00 = x
#     x = Conv2D(32,(3,3),padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Concatenate()([x00,x])
#     x01 = x
#     x = MaxPooling2D((2,2))(x)
    
#     x = Conv2D(64,(3,3),padding='same')(x)
#     x = BatchNormalization()(x)
#     x10 = x
#     x = Conv2D(64,(3,3),padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Concatenate()([x10,x])
#     x11 = x
#     x = MaxPooling2D((2,2))(x)
    
#     x = Conv2D(128,(3,3),padding='same')(x)
#     x = BatchNormalization()(x)
#     x20 = x
#     x = Conv2D(128,(3,3),padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Concatenate()([x20,x])
#     x21 = x
#     x = MaxPooling2D((2,2))(x)
    
#     x = Conv2D(256,(3,3),padding='same')(x)
#     x = BatchNormalization()(x)
#     x30 = x
#     x = Conv2D(256,(3,3),padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Concatenate()([x30,x])
#     x31 = x
#     x = MaxPooling2D((2,2))(x)
    
#     for dummy in range(5):
#         res_in = x
#         x = Conv2D(512,(3,3),padding='same')(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = Conv2D(512,(3,3),padding='same')(x)
#         x = BatchNormalization()(x)
#         x = Concatenate()([res_in,x])
#         x = Activation('relu')(x)
        
#     x = UpSampling2D((2,2), interpolation = 'nearest')(x)
#     x = Concatenate()([x31,x])
#     x = Conv2D(256,(3,3),padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Conv2D(256,(3,3),padding='same')(x)
#     x = BatchNormalization()(x)
    
#     x = UpSampling2D((2,2), interpolation = 'nearest')(x)
#     x = Concatenate()([x21,x])
#     x = Conv2D(128,(3,3),padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Conv2D(128,(3,3),padding='same')(x)
#     x = BatchNormalization()(x)
    
#     x = UpSampling2D((2,2), interpolation = 'nearest')(x)
#     x = Concatenate()([x11,x])
#     x = Conv2D(64,(3,3),padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Conv2D(64,(3,3),padding='same')(x)
#     x = BatchNormalization()(x)
    
#     x = UpSampling2D((2,2), interpolation = 'nearest')(x)
#     x = Concatenate()([x01,x])
#     x = Conv2D(32,(3,3),padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Conv2D(out_dim[-1],(3,3),padding='same', activation = out_act)(x)
    
#     model = Model(inputs = in_image, outputs = x)
#     return(model)
        
    
    
