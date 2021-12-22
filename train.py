# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:41:17 2021

@author: Christoph Angermann
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
import time
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import configparser
import optparse
from generators import define_resnet18, define_unet, define_styletransfer
from critics import define_dcgan, define_patchgan
from functions import *


# config = configparser.ConfigParser(allow_no_value = True)

# config.add_section('generator')
# config.set('generator', '; architecture: [resnet18,unet,styletransfer]')
# config.set('generator', 'arch_g', 'resnet18')
# config.set('generator', '; learning rate')
# config.set('generator', 'lrg', '1e-4')
# config.set('generator', '; amount of filters for the first convolutional layer')
# config.set('generator', 'fg', '32')
# config.set('generator', '; normalization after convolution: [None,Layer,Instance]')
# config.set('generator', 'normg', 'Instance')

# config.add_section('critic')
# config.set('critic', '; architecture: [dcgan,patchgan]')
# config.set('critic', 'arch_c', 'dcgan')
# config.set('critic', '; learning rate')
# config.set('critic', 'lrc', '1e-4')
# config.set('critic', '; amount of filters for the first convolutional layer')
# config.set('critic', 'fc', '32')
# config.set('critic', '; normalization after convolution: [None,Layer,Instance]')
# config.set('critic', 'normc', 'Layer')
# config.set('critic', '; number of intermediate critic iterations (doubled during the first 10% of training)')
# config.set('critic', 'nc', '12')
# config.set('critic', '; influence of the gradient penalty')
# config.set('critic', 'p', '100')

# config.add_section('training')
# config.set('training', '; epochs')
# config.set('training', 'ep', '120')
# config.set('training', '; batches per epoch')
# config.set('training', 'bat_per_epo', '85')
# config.set('training', '; minibatch size')
# config.set('training', 'batch_size', '8')
# config.set('training', '; reconstruction loss weight')
# config.set('training', 'lam', '100')
# config.set('training', '; whether to use the novel perceptual reconstruction (1) or the standard MAE on image space (0)')
# config.set('training', 'nl', '1')


# with open('config_file.ini', 'w') as configfile:
#     config.write(configfile)
    


#%%

parser = optparse.OptionParser()
config = configparser.ConfigParser(allow_no_value = True)
config.read('config_file.ini')

parser.add_option('--direc', action="store", dest="task", default="tex/")
options,args = parser.parse_args()

epochs = float(config['training']['ep'])
batch_size = float(config['training']['batch_size'])
bat_per_epo= float(config['training']['bat_per_epo'])
lam = float(config['training']['lam'])
new_loss = float(config['training']['nl'])

arch_g = config['generator']['arch_g']
fg = float(config['generator']['fg'])
lrg = float(config['generator']['lrg'])
tmp = config['generator']['normg']
normg = 0 if tmp=='None' else (1 if tmp=='Layer' else 1e6)

arch_c = config['critic']['arch_c']
fc = float(config['critic']['fc'])
lrc = float(config['critic']['lrc'])
tmp = config['critic']['normc']
normc = 0 if tmp=='None' else (1 if tmp=='Layer' else 1e6)
nc = float(config['critic']['nc'])
p = float(config['critic']['p'])

dc=0.; dd=0.;
loss_feat = MAE

#%%

def validate(discriminator_A,discriminator_B,generator_AtoB,generator_BtoA,evaluation,batch_size):
    X1 = [];X2=[]
    try:
        samples = np.random.choice(range(len(evaluation)),200,replace=False)
    except:
        samples = range(len(evaluation))
    
    lenn = len(samples)//2
    
    for k in samples[:lenn]:
        if task in ['rir/','bod/']:
            x1 = np.load(evaluation['input'][k])
        else:
            x1 = imread(evaluation['input'][k])
        X1.append(x1)
        
    for k in samples[lenn:]:
        x2 = np.load(evaluation['target'][k])
        X2.append(x2) 
        
    preds_B = generator_AtoB.predict(X1, batch_size=2)
    preds_A = generator_BtoA.predict(X2, batch_size=2)
    
    lossA = np.mean(discriminator_A.predict(X1)-discriminator_A.predict(preds_A))
    lossB = np.mean(discriminator_B.predict(X2)-discriminator_B.predict(preds_B))
    
    return(lossA,lossB)


def load_data(number_of_faces, index, inp, tar):
    X1 = [];X2=[]
    for k in range(number_of_faces):
        i1 = (index+k)%len(inp)
        i2 = (index+k)%len(tar)
        if task in ['rir/','bod/']:
            x1 = np.load(inp[i1])
        else:
            x1 = imread(inp[i1])
            
        x2 = np.load(tar[i2])
        X1.append(x1)
        X2.append(x2)    
    X1=np.array(X1);X2=np.array(X2)
    
    return ([X1, X2])


#%%

task = options.task; os.chdir(task)
files = os.listdir()
if 'results.csv' not in files:
    TABLE = pd.DataFrame(
        columns=['critic','generator','f_critic','f_generator',
             'lr_critic','lr_generator','norm_critic','norm_generator',
             'reconstuction','penalty','num_cri','imagespace_loss',
             'new_loss','RMSE','MAE','comments'])
    TABLE.to_csv('results.csv',index=False)
    
table = pd.DataFrame(np.array([arch_c,arch_g,fc,fg,lrc,lrg,normc,normg,lam,
                    p,nc,loss_feat.__name__,new_loss,
                    0,0,'']).reshape(1,16),
    columns=['critic','generator','f_critic','f_generator',
             'lr_critic','lr_generator','norm_critic','norm_generator',
             'reconstuction','penalty','num_cri','imagespace_loss',
             'new_loss','RMSE','MAE','comments'])

TABLE = pd.read_csv('results.csv',encoding = 'unicode_escape')
name = '%04d' %(TABLE.shape[0])
TABLE=TABLE.append(table,ignore_index=True)
TABLE.to_csv('results.csv',index=False)

losses = [wasserstein,identity,identity,identity,identity]
dyn_cyc = K.variable(lam)
dyn_feat = K.variable(0.)
    
inp= np.random.permutation([os.path.join('input',f) for f in os.listdir('input')])
tar= np.random.permutation([os.path.join('input',f) for f in os.listdir('target')])
evall = [evaluationp + f for f in os.listdir(evaluationp)]
df_e = pd.DataFrame(columns=['input','target'])
df_e['input']= pd.Series(np.sort(np.array(evall)[np.array(['jpg' in f for f in evall])]))
df_e['target']= pd.Series(np.sort(np.array(evall)[np.array(['npy' in f for f in evall])]))
evaluation = df_e

#%%

class BestLayerEver(Layer):
    def __init__(self, penalty, diam):
        self.penalty = penalty
        self.diam = diam
        super(BestLayerEver, self).__init__()
        
    def call(self, inputs):
       
        f_real, in_real, f_fake, in_fake = inputs        
        return(0.0005*(K.mean(K.square(f_real))+K.mean(K.square(f_fake))))
    

# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model(generator_1, discriminator, feb, fef, generator_2,inp_dim,out_dim,lr,losses,wcf,wcb,wf,
                           loss_forward,loss_backward,loss_feat,channel_4 = 'forward'):
    set_trainable(generator_1, True)
    set_trainable(discriminator, False)
    set_trainable(fef, False); set_trainable(feb, False)
    set_trainable(generator_2, False)
    
    input_gen = Input(shape=inp_dim)
    gen1_out = generator_1(input_gen)
    output_d = discriminator(gen1_out)
    output_f = generator_2(gen1_out)
    
    input_id = Input(shape=out_dim)
    gen2_out = generator_2(input_id)
    output_b = generator_1(gen2_out)
    
    cyc_f = Lambda(loss_forward)([input_gen, output_f])
    cyc_b = Lambda(loss_backward)([input_id, output_b])
    feat_f = Lambda(loss_feat)([fef(input_gen),fef(output_f)])
    feat_b = Lambda(loss_feat)([feb(input_id),feb(output_b)])
    
    model = Model([input_gen, input_id], [output_d, cyc_f, cyc_b, feat_f, feat_b])
    model.compile(loss=losses,
            loss_weights=[1.,wcf,wcb,wf,wf], optimizer=Adam(beta_1 = 0, beta_2 = 0.9, lr = lr))
    return model



def define_critic_with_gp(critic,inp_dim,penalty,diam,lrc):

    set_trainable(critic, True)

    inp1 = Input(shape=inp_dim)
    cri1 = critic(inp1)
    
    inp2 = Input(shape=inp_dim)
    cri2 = critic(inp2)
        
    inp4 = Input(shape=inp_dim)
    temp   = critic(inp4)
    gp     = GradientPenalty(penalty)([temp,inp4])
    
            
    pen = BestLayerEver(penalty, diam)([cri1,inp1,cri2,inp2])
    
    model = Model([inp1,inp2,inp4], [cri1,cri2,pen,gp])

    model.compile(loss=[wasserstein,wasserstein,identity,identity],
                loss_weights=[1,1,1,1],
                optimizer=Adam(beta_1 = 0, beta_2 = 0.9, lr=lrc))
    return model


#%%
inp_dim=load_data(1,0, inp, tar)[0].shape[1:]
out_dim=load_data(1,0, inp, tar)[1].shape[1:]
in_act = 'tanh'
out_act = 'tanh'

diamC = .2* np.sqrt(np.sum(np.square(np.ones(inp_dim)))) #*10
diamD = .2* np.sqrt(np.sum(np.square(np.ones(out_dim)))) #*10

        
def factor(d0):
    return(113/76800*d0**2+3/64*d0+62/15)
        

if a == 'resnet50':
    generator_CtoD=define_resnet50(inp_dim,out_dim, out_act = out_act,
                                norm=ngd,f=fgd)
    generator_DtoC=define_resnet50(out_dim,inp_dim, out_act = in_act,
                                norm=ngc,f=fgc)
        
if a == 'unet':
    generator_CtoD=define_unet(inp_dim,out_dim, out_act = out_act,
                                norm=ngd,f=fgd)
    generator_DtoC=define_unet(out_dim,inp_dim, out_act = in_act,
                                norm=ngc,f=fgc)
    
if a == 'resnet18':
    generator_CtoD=define_resnet18(inp_dim,out_dim, out_act = out_act,
                                norm=ngd,f=fgd)
    generator_DtoC=define_resnet18(out_dim,inp_dim, out_act = in_act,
                                norm=ngc,f=fgc)
    
if a == 'styletransfer':
    generator_CtoD=define_styletransfer(inp_dim,out_dim, out_act = out_act,
                                norm=ngd,f=fgd)
    generator_DtoC=define_styletransfer(out_dim,inp_dim, out_act = in_act,
                                norm=ngc,f=fgc)
    

    
generator_CtoD.summary()


in_image = Input(shape=inp_dim)
discriminator_C = define_discriminator_gp(in_image,inp_dim, norm = ncc, f=fcc)
features = discriminator_C.layers[-3].output
fe_C = Model(in_image, features)
critic_C = define_critic_with_gp(discriminator_C,inp_dim,p, 
                                 diam = diamC, lrc = lrcc)


in_image = Input(shape=out_dim)
discriminator_D = define_discriminator_gp(in_image,out_dim, norm = ncd, f=fcd)
features = discriminator_D.layers[-3].output
fe_D = Model(in_image, features)

# fe_D.set_weights(w[:4])
# t = fe_D.predict(x_realD)
# plt.imshow(t[0,...,5])

channels = [None,None]
if task == 'bod/':
    channels =['forward','backward']

critic_D = define_critic_with_gp(discriminator_D,out_dim,p, 
                                 diam = diamD, lrc = lrcd)

if new_loss:
    glosses = [MyLossC,MAE,loss_feat]
else:
    glosses = [MAE,MAE,loss_feat]
    
weights = [dyn_cyc,dyn_cyc,dyn_feat]


function_CtoD=define_composite_model(generator_CtoD,discriminator_D,fe_D,fe_C,
                                     generator_DtoC,inp_dim,out_dim,lrgd,losses, channel_4=channels[0],
                                     loss_forward = glosses[0], loss_backward = glosses[1],
                                    loss_feat = glosses[2], wcf=weights[0], wcb = weights[1], wf = weights[2])

function_DtoC=define_composite_model(generator_DtoC,discriminator_C,fe_C,fe_D,
                                     generator_CtoD, out_dim,inp_dim,lrgc,losses,channel_4 = channels[1],
                                     loss_forward = glosses[1], loss_backward = glosses[0],
                                    loss_feat = glosses[2], wcf=weights[1], wcb = weights[0], wf = weights[2] )

discriminator_D.summary()


            
dA=[]; dB=[]; gA=[]; gB=[]; dAval=[]; dBval=[]; DIF1=[]; DIF2=[]; gradD = []; gradC = []


patches = discriminator_C.output_shape[1:]
inp= np.random.permutation(df['input'].dropna())
tar= np.random.permutation(df['target'].dropna())

steps = bat_per_epo * epochs
count=0; i=0

poolC = list(); poolD = list()




#%%
# import scipy
# import cv2
# from PIL import Image
# import open3d as o3d


# generator_CtoD.set_weights(np.load('CtoDweights.npy', allow_pickle=True))
# generator_DtoC.set_weights(np.load('DtoCweights.npy', allow_pickle=True))


# img = imread('evaluation_tex/image_0011_008_00.jpg')[::2,::2]
# fac = 2*inp_dim[0]/img.shape[0]
# tmp = scipy.ndimage.zoom(img,(fac,fac*1.,1),order=1)[::2,::2]
# plt.imshow(tmp.astype(np.uint8));plt.show()
# tmp = (tmp-127.5)/127.5
# pred = generator_CtoD.predict(np.expand_dims(tmp,0))[0]*.5+.5
# plt.imshow(pred,cmap = 'Greys_r');plt.show()


# img = imread('evaluation_tex/image_0011_008_00.jpg')[::2,::2]
# pred = np.load('evaluation_tex/depth_0011_008_00.npy')[::2,::2]/.999
# pred[pred>0.01]=pred[pred>0.01]*.2+.8
# pred[pred<0.01]=.8
# color_raw = o3d.geometry.Image(np.ascontiguousarray(img))
# depth_raw = o3d.geometry.Image(pred)

# rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
# print(rgbd_image)
# plt.imshow(rgbd_image.color)
# plt.imshow(rgbd_image.depth);plt.colorbar()

# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,o3d.camera.PinholeCameraIntrinsic(
#         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# o3d.io.write_point_cloud('eval_example.ply',pcd)
#%%


# mid = t.shape[0]//2
# f = lambda x: (np.sin(x)+1)/2
# ii = np.linspace(-np.pi/2,np.pi/2,mid//4)
# foo = np.hstack((np.zeros(3*mid//4),f(ii)))
# filt = np.hstack((foo,np.flip(foo)))
# filt_mat = np.ones(t.shape)
# filt_mat = np.matmul(filt_mat,np.diag(filt))
# filt_mat = np.matmul(filt_mat.transpose(),np.diag(filt))
# ft = np.fft.fftshift(np.fft.fft2(t))
# ft *= filt_mat
# nt = np.fft.ifft2(np.fft.ifftshift(ft))
# plt.imshow(filt_mat);plt.show()
# plt.imshow(np.abs(nt));plt.show()


    


# y = cv2.cvtColor(t, cv2.COLOR_RGB2YCrCb)[...,0]
# y1 = (0.299*t[...,0]+0.587*t[...,1]+0.114*t[...,2]).astype(np.uint8)
# y= cv2.equalizeHist(y)
# y1=cv2.equalizeHist(y1)
# y1=np.expand_dims(y1,-1)
# plt.imshow(y); plt.colorbar(); plt.show()
# plt.imshow(y1); plt.colorbar(); plt.show()


# values_range = tf.constant([0., 255.], dtype = tf.float32)
# histogram = tf.histogram_fixed_width(y1, values_range, 256)
# cdf = tf.cumsum(histogram)
# cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]
# img_shape = tf.shape(y1)
# pix_cnt = y1[-3] * y1[-2]
# px_map = tf.round(tf.cast(cdf - cdf_min,tf.float32) * 255. / tf.cast(pix_cnt - 1,tf.float32))
# px_map = tf.cast(px_map, tf.uint8)
# eq_hist = tf.expand_dims(tf.gather_nd(px_map, tf.cast(y1, tf.int32)), 2)

# plt.imshow(eq_hist[...,0])


for i in np.arange(0,steps,1):
    placeholder = np.zeros((batch_size,1))
    
    if i < 0.1*steps:
        cri_iter = 2*nc
    else:
        cri_iter = nc
    
    mem = []
    for iters in range(cri_iter):
        x_realC, y_realC = generate_real_samples(load_data(batch_size,count,
                                                            inp,tar)[0], patches)
        x_realD, y_realD = generate_real_samples(load_data(batch_size,count,
                                                        inp,tar)[1], patches)
        
        count += batch_size

        
        x_fakeC, y_fakeC = generate_fake_samples(generator_DtoC, x_realD, patches)
        x_fakeC = update_image_pool(poolC, x_fakeC)
        
        x_fakeD, y_fakeD = generate_fake_samples(generator_CtoD, x_realC, patches)
        x_fakeD = update_image_pool(poolD, x_fakeD)
    
        x_mixC = np.zeros(x_realC.shape);
        for b in range(x_mixC.shape[0]):
            eps = np.random.uniform()
            x_mixC[b] = eps*x_realC[b] + (1.-eps)*x_fakeC[b];
            
    
        x_mixD = np.zeros(x_realD.shape);
        for b in range(x_mixD.shape[0]):
            eps = np.random.uniform()
            x_mixD[b] = eps*x_realD[b] + (1.-eps)*x_fakeD[b];
        
        cA_loss = critic_C.train_on_batch([x_realC,x_fakeC,x_mixC], [y_realC,y_fakeC,placeholder,
                                                                      placeholder])
        
        cB_loss = critic_D.train_on_batch([x_realD,x_fakeD,x_mixD], [y_realD,y_fakeD,placeholder,
                                                                      placeholder])
        
        if np.isnan(cB_loss[0])==True:
            break;
    if np.isnan(cB_loss[0])==True:
        break;
            
    
    print(str('W1_A %+08.2f, pen %+07.3f, gradL %+07.3f; W1_B %+08.2f, pen %+07.3f, gradL %+07.3f' 
              %(-(cA_loss[1]+cA_loss[2]),cA_loss[3],cA_loss[4],-(cB_loss[1]+cB_loss[2]),cB_loss[3],cB_loss[4])),
          flush=True)
    
    if i%5==0:
        vali = validate(discriminator_C,discriminator_D,generator_CtoD,generator_DtoC,evaluation,batch_size)
        for dummy in range(5):
            dAval.append(vali[0]); dBval.append(vali[1])
        
        print(str('\ncA r %+09.3f f %+09.3f \ncB r %+09.3f f %+09.3f\n' 
            %(np.mean(discriminator_C.predict(x_realC)),np.mean(discriminator_C.predict(x_fakeC)),
                np.mean(discriminator_D.predict(x_realD)),np.mean(discriminator_D.predict(x_fakeD)))),
              flush=True)
    
    if new_loss:
        K.set_value(dyn_feat, np.min([lamc*i/steps,lamc])); print(K.get_value(dyn_feat))
        #K.set_value(dyn_cyc,  np.max([lamc*(1.-i/steps),0.])); print(K.get_value(dyn_cyc))

    
    x_realC, y_realC = generate_real_samples(load_data(batch_size,count,
                                                    inp,tar)[0], patches)
    x_realD, y_realD = generate_real_samples(load_data(batch_size,count,
                                                        inp,tar)[1], patches)
    count += batch_size
    
    gA.append(function_DtoC.train_on_batch([x_realD, x_realC],
                                           [y_realC,placeholder,placeholder,placeholder,placeholder]))
    gB.append(function_CtoD.train_on_batch([x_realC, x_realD],
                                           [y_realD,placeholder,placeholder,placeholder,placeholder]))
        
    
    print('count: ',count,';  gB: ',gB[-1],'\n')
    
    dA.append(cA_loss[1]+cA_loss[2]); dB.append(cB_loss[1]+cB_loss[2]);
    gradD.append(cA_loss[-1]); gradC.append(cB_loss[-1])

    
    if (i+1) % (bat_per_epo ) == 0:
        tmp = discriminator_D.get_weights()
        criticW=np.hstack([item.reshape(-1) for sublist in tmp for item in sublist])
        plot_curves_gp(dA,dB,gA,gB,lamc,dAval,dBval,criticW,name+'fi')
        dif1,dif2 = evall(i, generator_CtoD,generator_DtoC, evaluation,name+'fi',task,batch_size)
        DIF1.append(dif1); DIF2.append(dif2)
        plt.figure()
        plt.plot(DIF1); plt.plot(DIF2)
        plt.title(str('%.3f,  %.3f' %(np.min(DIF1),np.max(DIF2))))
        plt.savefig( 'metric/%s.pdf' %(name+'fi'),dpi=300)
        
        try:
            wg1 = generator_CtoD.get_weights()
            np.save('/scratch/shared-christoph-adela/christoph/' + 
                    task + str('CtoDweights/%s_%04d' % (name,i+1)),wg1, allow_pickle=True)
            
            wg2 = generator_DtoC.get_weights()
            np.save('/scratch/shared-christoph-adela/christoph/' + 
                    task + str('DtoCweights/%s_%04d' % (name,i+1)),wg2, allow_pickle=True)
            
            wd = discriminator_D.get_weights()
            np.save('/scratch/shared-christoph-adela/christoph/' + 
                    task + str('criticDweights/%s_%04d' % (name,i+1)),wd, allow_pickle=True)
            
        except:
            print('no weights saved')
        
        # wc = discriminator_D.get_weights()
        # np.save('/scratch/shared-christoph-adela/christoph/' + 
        #         task + str('criticDweights/%s_%04d' % (name,i+1)),wc, allow_pickle=True)
        
        inp= np.random.permutation(df['input'].dropna())
        tar= np.random.permutation(df['target'].dropna())
        
        #if task == 'bod/':
            #x_realC, y_realC = generate_real_samples(load_data(batch_size,count,
                                                            #inp,tar)[0], patches)
            #x_realD, y_realD = generate_real_samples(load_data(batch_size,count,
                                                            #inp,tar)[1], patches)
            
            #count += batch_size
            #x_fakeC, y_fakeC = generate_fake_samples(generator_DtoC, x_realD, patches)
            #x_fakeD, y_fakeD = generate_fake_samples(generator_CtoD, x_realC, patches)
            #n_samples = np.max([x_fakeD.shape[0]//2,4])
            
            #fig = plt.figure(figsize=(5.5,5.5))
            #for ii in range(n_samples):
                #ax=fig.add_subplot(4, n_samples, 1 + ii)
                #plt.axis('off')
                #ax.imshow(x_realC[ii]*.5+.5)
            #for ii in range(n_samples):
                #ax=fig.add_subplot(4, n_samples, 1 + n_samples + ii)
                #plt.axis('off')
                #ax.imshow(x_fakeD[ii,...,0],cmap='Greys_r')
            #for ii in range(n_samples):
                #ax=fig.add_subplot(4, n_samples, 1 + 2*n_samples + ii)
                #plt.axis('off')
                #ax.imshow(x_realD[ii,...,0],cmap='Greys_r')
            #for ii in range(n_samples):
                #ax=fig.add_subplot(4, n_samples, 1 + 3*n_samples + ii)
                #plt.axis('off')
                #ax.imshow(x_fakeC[ii]*.5+.5)
            
            
            #fig.tight_layout(pad=.01)
            #filename ='bod_plots/%s_%04d.png' % (name,(i+1))
            #plt.savefig( filename,dpi=200)
            

TABLE=pd.read_csv('TABLEfinal.csv',encoding = 'unicode_escape')
TABLE.metric1[int(name)]=np.min(DIF1)
TABLE.metric2[int(name)]=np.max(DIF2)
TABLE.to_csv('TABLEfinal.csv',index=False)

for foo in range(50):
    print('###########################################################################################')


#%%

# inc_model = InceptionV3(include_top=False, pooling='avg',weights='imagenet',
#                                         input_shape=[256,256,3])


# act1 = inc_model.predict(images1)
# act2 = inc_model.predict(images2)
# mu1, sigma1 = tf.reduce_mean(act1,axis=0), np.cov(act1, rowvar=False,bias=False)
# mu2, sigma2 = tf.reduce_mean(act2,axis=0), np.cov(act2, rowvar=False,bias =False)
# l = tf.cast(tf.shape(act1)[0],tf.float32)

# mean_x = tf.reduce_mean(act1, axis=0, keepdims=True)
# mx = tf.matmul(tf.transpose(mean_x), mean_x)
# vx = tf.matmul(tf.transpose(act1), act1)/l
# sigma1 = (vx - mx)*l/(l-1)

# mean_y = tf.reduce_mean(act2, axis=0, keepdims=True)
# my = tf.matmul(tf.transpose(mean_y), mean_y)
# vy = tf.matmul(tf.transpose(act2), act2)/l
# sigma2 = (vy - my)*l/(l-1)

# covmean = 2.*tf.linalg.sqrtm(tf.cast(tf.matmul(sigma1,sigma2),tf.complex64))

# res = K.sum(K.square(mu1-mu2)) + tf.linalg.trace(sigma1+sigma2)-2*tf.matmul(sigma1,sigma2))


