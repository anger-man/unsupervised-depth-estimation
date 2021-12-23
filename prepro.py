#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:17:42 2021

@author: c
"""

#%%

import h5py
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave,imread
import numpy as np
import time
import pandas as pd
from scipy.ndimage import zoom
from scipy.ndimage import rotate
import cv2

def run_histogram_equalization(image):
    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return(equalized_img)

def zoom_in(img,csize):
    if len(img.shape)==2:
        img = np.expand_dims(img,-1)
    img =img[csize:img.shape[0]-csize,csize:img.shape[1]-csize]
    img = zoom(img,(512/img.shape[0],512/img.shape[1],1), order=1)
    return(img)

def center_crop(img):
    h,w = img.shape[:2]
    if h<w:
        crop = int((w-h)/2)
        img = img[:,crop:w-crop,:]
    if w<h:
        crop = int((h-w)/2)
        img = img[crop:h-crop,:,:]
    return img
    

#%%

path = 'tex/'
evaluationp = 'evaluation/'
inputp = 'input/'
targetp = 'target/'

try:
    os.chdir('data/'+path)
except:
    print('already here')

#%%

        
files = [os.path.join('preprocessed',f[:30]) for f in os.listdir('preprocessed')]
files = sorted(files[:len(files)-1]); files = np.unique(files)

objects = np.unique([f[23:28] for f in files])
eval_obj = objects[::7]
train_obj = [f for f in objects if f not in eval_obj]
lenn = len(train_obj)//2
count1 = 0; count2 = 0

# os.mkdir('input_tex'); os.mkdir('target_tex'); os.mkdir('evaluation_tex')

for f in files:
    if f[23:28] in train_obj[::2]:
        ind = f[19:27]
        image = imread(f + 'Portrait.png')
        image = (image*255).astype(np.uint8)
        image = center_crop(image)
        image = zoom(image,(512/image.shape[0],512/image.shape[1],1),order=1)
        
        tmp = image.copy()
      
        count = 0
        for k in range(2):
            image = zoom_in(image,csize=np.random.choice(range(32)))
            imsave(inputp + str('image_%s_%02d.jpg' %(ind,count)),image[::2,::2]); count += 1
            imsave(inputp + str('image_%s_%02d.jpg' %(ind,count)),np.flip(image[::2,::2],axis=1)); count += 1
        
        eq_image = run_histogram_equalization(tmp)
        for k in range(2):
            eq_image = zoom_in(eq_image,csize=np.random.choice(range(32)))
            imsave(inputp + str('image_%s_%02d.jpg' %(ind,count)),eq_image[::2,::2]); count += 1
            imsave(inputp + str('image_%s_%02d.jpg' %(ind,count)),np.flip(eq_image[::2,::2],axis=1)); count += 1
        
        blurred = cv2.GaussianBlur(tmp, (5,5),cv2.BORDER_DEFAULT)
        for k in range(2):
            blurred = zoom_in(blurred,csize=np.random.choice(range(32)))
            imsave(inputp + str('image_%s_%02d.jpg' %(ind,count)),blurred[::2,::2]); count += 1
            imsave(inputp + str('image_%s_%02d.jpg' %(ind,count)),np.flip(blurred[::2,::2],axis=1)); count += 1
        
        count1+=1
        
        
        
    elif f[23:28] in train_obj[1::2]:
        ind = f[19:27]
        depth = np.expand_dims(imread(f + 'Range.png'),-1)
        depth = center_crop(depth)
        count = 0
        
        depth = zoom(depth,(512/depth.shape[0],512/depth.shape[1],1),order=1)
        depth=depth*2. - 1.
        tmp = depth.copy()
        
        for k in range(2):
            depth = zoom_in(depth, csize=np.random.choice(range(32)))
            np.save(targetp + str('depth_%s_%02d.npy' %(ind,count)),depth[::2,::2]); count += 1
            np.save(targetp + str('depth_%s_%02d.npy' %(ind,count)),np.flip(depth[::2,::2],axis=1)); count += 1
        
        for k in range(2):
            blurred = zoom_in(cv2.GaussianBlur(tmp, (3,3),cv2.BORDER_DEFAULT),csize=np.random.choice(range(32)))
            np.save(targetp + str('depth_%s_%02d.npy' %(ind,count)),blurred[::2,::2]); count += 1
            np.save(targetp + str('depth_%s_%02d.npy' %(ind,count)),np.flip(blurred[::2,::2],axis=1)); count += 1
        
        for k in range(2):
            blurred = zoom_in(cv2.GaussianBlur(tmp, (5,5),cv2.BORDER_DEFAULT),csize=np.random.choice(range(32)))
            np.save(targetp + str('depth_%s_%02d.npy' %(ind,count)),blurred[::2,::2]); count += 1
            np.save(targetp + str('depth_%s_%02d.npy' %(ind,count)),np.flip(blurred[::2,::2],axis=1)); count += 1
                
            count2+=1
                
    elif f[23:28] in eval_obj:
        ind = f[19:27]
        image = imread(f + 'Portrait.png')
        image = (image*255).astype(np.uint8)
        depth = np.expand_dims(imread(f + 'Range.png'),-1)
        
        image = center_crop(image); depth = center_crop(depth)
        image = zoom(image,(512/image.shape[0],512/image.shape[1],1),order=1)
        
        depth = zoom(depth,(512/depth.shape[0],512/depth.shape[1],1),order=1)
        depth = depth*2.-1.
        print(depth.min(), depth.max())
        
        count = 0
        for k in range(2):
            csize = np.random.choice(range(32))
            imsave(evaluationp + str('image_%s_%02d.jpg' %(ind,count)),zoom_in(image,csize)[::2,::2]); 
            np.save(evaluationp + str('depth_%s_%02d.npy' %(ind,count)),zoom_in(depth,csize)[::2,::2]); count+=1
        
        plt.imshow(image);plt.show()
    else:
        print('fail')
            
         
            
            
#%%
    
    
df = pd.DataFrame(columns=['input','target'])
def get_paths(scr,d):
    return [os.path.join(scr+d, f) for f in os.listdir(d)]    
# df['input']=pd.Series(np.random.permutation(
#     get_paths('/scratch/shared-christoph-adela/christoph/',inputp)))
# df['target']=pd.Series(np.random.permutation(
#     get_paths('/scratch/shared-christoph-adela/christoph/',targetp)))
df['input']=pd.Series(np.random.permutation(
    get_paths('',inputp)))
df['target']=pd.Series(np.random.permutation(
    get_paths('',targetp)))

df.to_csv('filenames_%s.csv' %path[:3])

# evall = ['/scratch/shared-christoph-adela/christoph/'+evaluationp + f for f in os.listdir(evaluationp)]
evall = [evaluationp + f for f in os.listdir(evaluationp)]

df_e = pd.DataFrame(columns=['input','target'])
df_e['input']= pd.Series(np.sort(np.array(evall)[np.array(['jpg' in f for f in evall])]))
df_e['target']= pd.Series(np.sort(np.array(evall)[np.array(['npy' in f for f in evall])]))
df_e.to_csv('evaluation_%s.csv' %path[:3])



