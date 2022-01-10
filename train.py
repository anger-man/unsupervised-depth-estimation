# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:41:17 2021

@author: c
"""

#%%

import os
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
# config.set('training', '; batches per epoch (>25)')
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

parser.add_option('--direc', action="store", dest="task", default="face_depth/")
options,args = parser.parse_args()

epochs = float(config['training']['ep'])
batch_size = int(config['training']['batch_size'])
bat_per_epo= np.max([float(config['training']['bat_per_epo']),25])
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
nc = np.int(float(config['critic']['nc']))
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
        
    X1 = np.array(X1); X2 = np.array(X2)
    
    X1 = X1/127.5 -1.
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
    X1 = X1/127.5 -1.
    
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
                    0.,0.,'']).reshape(1,16),
    columns=['critic','generator','f_critic','f_generator',
             'lr_critic','lr_generator','norm_critic','norm_generator',
             'reconstuction','penalty','num_cri','imagespace_loss',
             'new_loss','RMSE','MAE','comments'])

TABLE = pd.read_csv('results.csv',encoding = 'unicode_escape')
name = '%04d' %(TABLE.shape[0])
TABLE=TABLE.append(table,ignore_index=True)
TABLE.to_csv('results.csv',index=False)

try:
    os.mkdir('rgb_generator')
    os.mkdir('depth_generator')
    os.mkdir('metrics')
    os.mkdir('plots')
    os.mkdir('generator_loss')
    os.mkdir('critic_loss')
except:
    'directories already generated'

losses = [wasserstein,identity,identity,identity,identity]
dyn_cyc = K.variable(lam)
dyn_feat = K.variable(0.)
    
inp= np.random.permutation([os.path.join('input',f) for f in os.listdir('input')])
tar= np.random.permutation([os.path.join('target',f) for f in os.listdir('target')])
tmp = [os.path.join('evaluation', f) for f in os.listdir('evaluation')]
df_e = pd.DataFrame(columns=['input','target'])
df_e['input']= pd.Series(np.sort(np.array(tmp)[np.array(['jpg' in f for f in tmp])]))
df_e['target']= pd.Series(np.sort(np.array(tmp)[np.array(['npy' in f for f in tmp])]))
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

        


if arch_g == 'resnet18':
    generator_CtoD=define_resnet18(inp_dim,out_dim, out_act = out_act,
                                norm=normg,f=fg)
    generator_DtoC=define_resnet18(out_dim,inp_dim, out_act = in_act,
                                norm=normg,f=fg)
    
elif arch_g == 'styletransfer':
    generator_CtoD=define_styletransfer(inp_dim,out_dim, out_act = out_act,
                                norm=normg,f=fg)
    generator_DtoC=define_styletransfer(out_dim,inp_dim, out_act = in_act,
                                norm=normg,f=fg)
    
else:
    generator_CtoD=define_unet(inp_dim,out_dim, out_act = out_act,
                                norm=normg,f=fg)
    generator_DtoC=define_unet(out_dim,inp_dim, out_act = in_act,
                                norm=normg,f=fg)
    

    
generator_CtoD.summary()


in_image = Input(shape=inp_dim)

if arch_c == 'patchgan':
    discriminator_C = define_patchgan(in_image,inp_dim, norm = normc, f=fc)
else:
    discriminator_C = define_dcgan(in_image,inp_dim, norm = normc, f=fc)
features = discriminator_C.layers[-3].output
fe_C = Model(in_image, features)
critic_C = define_critic_with_gp(discriminator_C,inp_dim,p, 
                                 diam = diamC, lrc = lrc)

in_image = Input(shape=out_dim)
if arch_c == 'patchgan':
    discriminator_D = define_patchgan(in_image,out_dim, norm = normc, f=fc)
else:
    discriminator_D = define_dcgan(in_image,out_dim, norm = normc, f=fc)
features = discriminator_D.layers[-3].output
fe_D = Model(in_image, features)
critic_D = define_critic_with_gp(discriminator_D,out_dim,p, 
                                 diam = diamD, lrc = lrc)

if new_loss:
    glosses = [MyLossC,MAE,loss_feat]
else:
    glosses = [MAE,MAE,loss_feat]
    
weights = [dyn_cyc,dyn_cyc,dyn_feat]


function_CtoD=define_composite_model(generator_CtoD,discriminator_D,fe_D,fe_C,
                                     generator_DtoC,inp_dim,out_dim,lrg,losses,
                                     loss_forward = glosses[0], loss_backward = glosses[1],
                                    loss_feat = glosses[2], wcf=weights[0], wcb = weights[1], wf = weights[2])

function_DtoC=define_composite_model(generator_DtoC,discriminator_C,fe_C,fe_D,
                                     generator_CtoD, out_dim,inp_dim,lrg,losses,
                                     loss_forward = glosses[1], loss_backward = glosses[0],
                                    loss_feat = glosses[2], wcf=weights[1], wcb = weights[0], wf = weights[2] )

discriminator_D.summary()


            
dA=[]; dB=[]; gA=[]; gB=[]; dAval=[]; dBval=[]; DIF1=[]; DIF2=[]; gradD = []; gradC = []


patches = discriminator_C.output_shape[1:]
inp= np.random.permutation(inp)
tar= np.random.permutation(tar)

steps = bat_per_epo * epochs
count=0; i=0

poolC = list(); poolD = list()


#%%

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
        K.set_value(dyn_feat, np.min([lam*i/steps,lam])); 
        print('perceptual loss weight: ',K.get_value(dyn_feat))

    x_realC, y_realC = generate_real_samples(load_data(batch_size,count,
                                                    inp,tar)[0], patches)
    x_realD, y_realD = generate_real_samples(load_data(batch_size,count,
                                                        inp,tar)[1], patches)
    count += batch_size
    
    gA.append(function_DtoC.train_on_batch([x_realD, x_realC],
                                           [y_realC,placeholder,placeholder,placeholder,placeholder]))
    gB.append(function_CtoD.train_on_batch([x_realC, x_realD],
                                           [y_realD,placeholder,placeholder,placeholder,placeholder]))
    gB[-1][2] *= K.get_value(dyn_cyc); gB[-1][3] *= K.get_value(dyn_cyc)
    gB[-1][4] *= K.get_value(dyn_feat); gB[-1][5] *= K.get_value(dyn_feat)
        
    
    # print('count: ',count,';  gB: ',gB[-1],'\n')
    
    dA.append(cA_loss[1]+cA_loss[2]); dB.append(cB_loss[1]+cB_loss[2]);
    gradD.append(cA_loss[-1]); gradC.append(cB_loss[-1])

    
    if (i+1) % (bat_per_epo ) == 0:
        tmp = discriminator_D.get_weights()
        criticW=np.hstack([item.reshape(-1) for sublist in tmp for item in sublist])
        plot_curves_gp(dA,dB,gA,gB,dAval,dBval,criticW,name+'fi')
        dif1,dif2 = evall(i, generator_CtoD,generator_DtoC, evaluation,name+'fi',task,batch_size)
        DIF1.append(dif1); DIF2.append(dif2)
        plt.figure()
        plt.plot(DIF1); plt.plot(DIF2)
        plt.title(str('%.3f,  %.3f' %(np.min(DIF1),np.min(DIF2))))
        plt.savefig( 'metrics/%s.pdf' %(name+'fi'),dpi=300)
        

        wg1 = generator_CtoD.get_weights()
        np.save(str('depth_generator/%s_%04d' % (name,i+1)),wg1, allow_pickle=True)
        
        wg2 = generator_DtoC.get_weights()
        np.save(str('rgb_generator/%s_%04d' % (name,i+1)),wg2, allow_pickle=True)
        
        inp= np.random.permutation(inp)
        tar= np.random.permutation(tar)
        
        TABLE=pd.read_csv('results.csv',encoding = 'unicode_escape')
        TABLE.RMSE[int(name)]=np.min(DIF1)
        TABLE.MAE[int(name)]=np.min(DIF2)
        TABLE.to_csv('results.csv',index=False)

