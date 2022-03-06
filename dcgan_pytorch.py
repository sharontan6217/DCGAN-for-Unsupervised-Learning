# -*- coding: utf-8 -*-
"""ASD_DCGAN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ueRKpg9wlvEBsnHcBgCr3KhO26jzqvDL
"""




import pandas as pd
import numpy as np
import struct
import wave
import sys
import inspect
import pydub
from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub.silence import split_on_silence
import sklearn
from sklearn import cluster, preprocessing
from sklearn.preprocessing import normalize, scale, MinMaxScaler, LabelBinarizer
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, roc_auc_score,roc_curve, precision_recall_curve,auc, f1_score,silhouette_score,normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import reshape,cat,multiply, flatten
from torch.nn import init,Sequential, Module
from torch.nn import Linear, BatchNorm1d,BatchNorm2d,BatchNorm3d, Dropout, Dropout2d, Flatten, ZeroPad2d, ConvTranspose2d, ConvTranspose3d
from torch.nn import LeakyReLU, ReLU, Softmax, Tanh, Sigmoid
from torch.nn import UpsamplingBilinear2d, Upsample,Conv2d,MaxPool2d,AvgPool1d,AvgPool2d,Conv1d,AdaptiveAvgPool1d,AdaptiveMaxPool1d, MaxPool1d,UpsamplingNearest2d,MaxPool3d
from torch.nn import MSELoss,L1Loss,CrossEntropyLoss,SoftMarginLoss,BCELoss,BCEWithLogitsLoss,HingeEmbeddingLoss
from torch.autograd import Variable
from torch.optim import Adamax, Adam
import torchvision.utils as vutils
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from torch.nn.modules.activation import Softmax2d

import datetime
from datetime import timedelta



import tensorflow as tf

from tensorflow.keras import utils

from tensorflow.python.client import device_lib

import scipy
from scipy.io import wavfile
from scipy.signal import butter,lfilter,filtfilt,lfilter_zi,sosfilt
import pickle
import random
from scipy.spatial.distance import sqeuclidean
import matplotlib.dates as mdates
import datetime
from datetime import timedelta
from plotnine import *
import plotnine
import matplotlib.pyplot as plt
import os
import importlib
import math


sys.setrecursionlimit(10000)



# init data
wave_dir_1='drive/MyDrive/realData/train/'
wave_dir_2='drive/MyDrive/realData/test/'
chunk_dir='drive/MyDrive/asd/chunk/pytorch/dcgan/'
csv_dir='drive/MyDrive/asd/csv/pytorch/dcgan/'
log_dir='drive/MyDrive/asd/log/pytorch/dcgan/'
graph_dir='drive/MyDrive/asd/graph/line/pytorch/dcgan/'
graph_dir_1='drive/MyDrive/asd/graph/loss/pytorch/dcgan/'
graph_dir_2='drive/MyDrive/asd/graph/error/pytorch/dcgan/'
graph_dir_3='drive/MyDrive/asd/graph/distribution/pytorch/dcgan/'
graph_dir_4='drive/MyDrive/asd/graph/rocCurve/pytorch/dcgan/'
graph_dir_5='drive/MyDrive/asd/graph/prCurve/pytorch/dcgan/'




file_format='.wav'

noise_factor=0.2

sample_count=100
predict_count=10



PopulationSize=int(4410*(sample_count-int(predict_count/2)))
PredictSize=int(predict_count*4410)

GapSize=0
TotalSize=PopulationSize+GapSize+PredictSize

#n0=random.randint(0,5)*4410

n0=0
n1=n0+PopulationSize
n2=n0+PopulationSize
n3=n0+PopulationSize+PredictSize
n4=n0+PopulationSize+GapSize+PredictSize


learning_rate_d=0.00002
learning_rate_g=0.00002
learning_rate_c=0.00004
criterion=torch.nn.MSELoss()
#criterion=torch.nn.BCELoss()
beta1=0.5
batch_size=16



evaluation_rate=0.015



#tf.compat.v1.disable_eager_execution()

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype=torch.int64



class data_preprocessing():


    def file_alias():
        
        file_string_a='normal_id_000'
        file_string_b='anomaly_id_000'
        

        file_name_a=[]
        file_name_b=[]
        for i in range(0,sample_count+int(n0/4410)):
            file_number_a=random.randint(1,228)
            if file_number_a<10:
                file_number_a='00'+str(object=file_number_a)
            elif file_number_a<100:
                file_number_a='0'+str(object=file_number_a)
            else:
                file_number_a=str(object=file_number_a)
            
            
            file_number_b=random.randint(1,120)
            if file_number_b<10:
                file_number_b='00'+str(object=file_number_b)
            elif file_number_b<100:
                file_number_b='0'+str(object=file_number_b)
            else:
                file_number_b=str(object=file_number_b)
            file_name_a.append(file_string_a+file_number_a)
            file_name_b.append(file_string_b+file_number_b)
            
            i+=1

        return file_name_a, file_name_b
    
        
    def populationInit():


        global m
        file_name_a, file_name_b=data_preprocessing.file_alias()
        i=0
        df_a=[]

        df_b=[]
        y_a=[]
        y_b=[]
        

        for i in range(0,sample_count+int(n0/4410)):
            print(i,file_name_a[i])
            file_alias=wave_dir_1+file_name_a[i]+file_format
            Fs, audioData=wavfile.read(file_alias)
            n=audioData.size
            t=round(Fs/10)
            m=round(n/t)
            
            #print(n,t,m)
            
            
            wavFile=wave.open(file_alias)
            audioString=wavFile.readframes(wavFile.getnframes())
            audioText=struct.unpack('%ih' % (wavFile.getnframes()*wavFile.getnchannels()),audioString)
            
            noise = np.random.uniform(low=-1.0, high=1.0, size=len(audioText))*noise_factor
            audioText=audioText+noise
            
            #audioText=data_preprocessing.denoise(self=self,xn=audioText)
            audioText=[float(val)/pow(2,15) for val in audioText]
            #print(len(audioText)/m,round(len(audioText)/m))
            audio_rows=round(len(audioText)/m)
            audio_cols=m
            textArray=np.array_split(audioText,round(len(audioText)/m))
            df=pd.DataFrame(data=textArray)
            df.to_csv(csv_dir+file_name_a[i]+'.csv')
            problem=[]
            #problem_=[]
            for j in range (round(len(audioText)/m)):
                problem.append(0)
                #problem_.append(0)
                j+=1
            #df['Test']=problem_
            df['IssueOrNot']=problem
            df_a.append(df)
            y_a.append(0)
            


            file_alias_b=wave_dir_2+file_name_b[i]+file_format
            Fs_b, audioData_b=wavfile.read(file_alias_b)
            n_b=audioData_b.size
            t_b=round(Fs_b/10)
            m_b=round(n_b/t_b)
            #print(m)
            
            wavFile_b=wave.open(file_alias_b)
            audioString_b=wavFile_b.readframes(wavFile_b.getnframes())
            audioText_b=struct.unpack('%ih' % (wavFile_b.getnframes()*wavFile_b.getnchannels()),audioString_b)

            audioText_b=audioText_b+noise
            
            #audioText_b=data_preprocessing.denoise(self=self,xn=audioText_b)
            audioText_b=[float(val_b)/pow(2,15) for val_b in audioText_b]
            #print(len(audioText_b)/m_b,round(len(audioText_b)/m_b))
            textArray_b=np.array_split(audioText_b,round(len(audioText_b)/m_b))
            df_=pd.DataFrame(data=textArray_b)
            df_.to_csv(csv_dir+file_name_b[i]+'.csv')
            problem_b=[]
            #problem_b_=[]
            for i in range (round(len(audioText_b)/m_b)):
                problem_b.append(1)
                #problem_b_.append(0)
                i+=1
            #df_['Test']=problem_b_
            df_['IssueOrNot']=problem_b
            df_b.append(df_)
            y_b.append(1)
            #print(i)
            
            i+=1
        


        df_a=pd.concat(df_a,axis=0)
        df_b=pd.concat(df_b,axis=0)

        df=df_a.append(df_b)
        print(df)
        df_y_a=pd.DataFrame(data=y_a)
        df_y_b=pd.DataFrame(data=y_b)

        df_y=df_y_a.append(df_y_b)
        df_y=df_y.reindex()


        feature=[]
        df_consolidated=pd.DataFrame()
        for i in range(0,m):
            feature_=df[i]
            feature_name='Feature_'+str(object=i)
            feature_=scale(feature_)
            feature_=normalize(np.array(np.reshape(feature_,(-1,1))))
            feature_=pd.Series(np.reshape(feature_,(-1)))

            issue_or_not=df['IssueOrNot']
            issue_or_not=pd.Series(np.reshape(np.array(issue_or_not),(-1)))

            df_consolidated.loc[:,feature_name]=feature_

            i+=1

        df_consolidated['IssueOrNot']=issue_or_not


        
        df_=df_consolidated.values

        scaler=MinMaxScaler()
        scaler.fit(df_)
        scaled_data=scaler.transform(df_)
        
        y=np.array(df_y.values)
        
 

                

        #print(n0,n2,n3)
        
        x_train=df_[n0:n1,:][:,-m-1:-1]
        y_train=df_[n0:n1,:][:,-1:]
        print(y_train)



        x_test=df_[n2:n3,:][:,-m-1:-1]


        y_test=df_[n2:n3,:][:,-1:]

        print("y_test is: ", y_test)
        

        


        

        


        features=x_train.shape[1]
  
        x_train=x_train.astype('float32')
        x_test=x_test.astype('float32')
        




        x_train=np.reshape(x_train,(int(sample_count-int(predict_count/2)),1,t,features))
        y_train=np.reshape(y_train,(int(sample_count-int(predict_count/2)),t,1))
        x_test=np.reshape(x_test,(int(predict_count),1,t,features))
        y_test=np.reshape(y_test,(int(predict_count),t,1))

        y_train=utils.to_categorical(y_train,2)
        y_test=utils.to_categorical(y_test,2)
        y_train=y_train.astype('float32')
        y_test=y_test.astype('float32')

     



        dataset_train=TensorDataset(torch.Tensor(x_train),torch.Tensor(y_train))
           
        dataset_test=TensorDataset(torch.Tensor(x_test),torch.Tensor(y_test))       



        

        data_train=torch.utils.data.DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=True)
        data_test=torch.utils.data.DataLoader(dataset=dataset_test,batch_size=batch_size,shuffle=True)
        


        

        print(x_train.shape)        

        
        print("'x_test's shape: ", x_test.shape)
        print("'y_test's shape: ", y_test.shape)


        
        
        

        print('m is : ',m)
        print('features is : ',features)

        return x_train,y_train,x_test,y_test,features,df_consolidated,data_train,data_test,y,m,t

    
    def LoaderSplit(x):
        X=x[:-1]
        Y=x[-1:]
        return X,Y
    @tf.function(experimental_follow_type_hints=True)
    def f_with_hints(x:tf.Tensor):
        print('Tracing')
        return x
    
    
   
    
class BuildGenerator(torch.nn.Module):
    
    def __init__(self,
                 gpu_unit):
        super(BuildGenerator,self).__init__()
        

        self.gpu_unit=gpu_unit

        
        # Build the generator.
        self.generatorLayer_1=Sequential(
                ConvTranspose2d(1,16,kernel_size=1,stride=1,padding=0),
                BatchNorm2d(num_features=16,momentum=0.8),
                LeakyReLU(0.2),
                Dropout2d(0.1))
        self.generatorLayer_2=Sequential(
                ConvTranspose2d(16,32,kernel_size=1,stride=1,padding=0),
                BatchNorm2d(num_features=32,momentum=0.8),
                LeakyReLU(0.2),
                Dropout2d(0.1))

        self.generatorLayer_3=Sequential(
                ConvTranspose2d(32,64,kernel_size=1,stride=1,padding=0),
                BatchNorm2d(num_features=64,momentum=0.8),
                LeakyReLU(0.2),
                Dropout2d(0.1))

        self.generatorLayer_4=Sequential(
                ConvTranspose2d(64,1,kernel_size=1),
                BatchNorm2d(num_features=1,momentum=0.8),
                LeakyReLU(0.2),
                Softmax2d(),
                Dropout2d(0.2))
              

    def forward(self,input):
        main_g=self.generatorLayer_1(input)
        #print("generator layer 1: ", main_g.shape)
        main_g=self.generatorLayer_2(main_g)
        #print("generator layer 2: ", main_g.shape)
        main_g=self.generatorLayer_3(main_g)
        #print("generator layer 3: ", main_g.shape)
        out=self.generatorLayer_4(main_g)
        #print("generator layer 4: ", out.shape)
        return out

        
class BuildDiscriminator(torch.nn.Module):
    def __init__(self,
                 gpu_unit,
                 features):
        super(BuildDiscriminator,self).__init__()
        
        self.gpu_unit=gpu_unit
        self.features=features

        
        # Build the discriminator.
        self.discriminatorLayer_1=Sequential(Conv2d(1,16,kernel_size=2,stride=1,padding="same"),
                         BatchNorm2d(num_features=16,momentum=0.8),
                         LeakyReLU(0.2),
                         Dropout2d(0.1))
        self.discriminatorLayer_2=Sequential(Conv2d(16,32,kernel_size=2,stride=1,padding="same"),
                         BatchNorm2d(num_features=32,momentum=0.8),
                         LeakyReLU(0.2),
                         Dropout2d(0.1))
      
        self.discriminatorLayer_3=Sequential(Conv2d(32,64,kernel_size=2,stride=1,padding="same"),
                         BatchNorm2d(num_features=64,momentum=0.8),
                         LeakyReLU(0.2),
                         Dropout2d(0.1))    
        
        self.discriminatorLayer_4=Sequential(Conv2d(64,1,kernel_size=2,stride=1,padding="same"),
                         BatchNorm2d(num_features=1,momentum=0.8),
                         LeakyReLU(0.2),
                         Dropout2d(0.1))
        self.fc_1=Linear(self.features,self.features,bias=True)
        torch.nn.init.xavier_uniform_(self.fc_1.weight)
        self.fcLayer_1=Sequential(self.fc_1,ReLU(True),Dropout2d(0.2))
        self.fc_2=Linear(self.features,2,bias=True)
        torch.nn.init.xavier_uniform_(self.fc_2.weight)
        self.fcLayer_2=Sequential(self.fc_2,Sigmoid())

    def forward(self,input):
        main_d=self.discriminatorLayer_1(input)
        #print("discriminator layer 1: ", main_d.shape)
        main_d=self.discriminatorLayer_2(main_d)
        #print("discriminator layer 2: ", main_d.shape)
        main_d=self.discriminatorLayer_3(main_d)
        #print("discriminator layer 3: ", main_d.shape)
        main_d=self.discriminatorLayer_4(main_d)
        #print("discriminator layer 4: ", main_d.shape)
        fc1=self.fcLayer_1(main_d)
        #print(fc1.shape)
        fl=flatten(fc1, start_dim=1,end_dim=2)
        #print(fl.shape)
        out=self.fcLayer_2(fl)
        #print(out.shape)
        return out

class DCGANAnalysis():
    
    def weights_init(m):
        classname=m.__class__.__name__
        if classname.find('Conv')!=-1:
            nn.init.normal_(m.weight.data,0.0,0.02)
        elif classname.find('BatchNorm')!=-1:
            nn.init.normal_(m.weight.data,1,0.02)
            nn.init.constant_(m.bias.data,0)
            
    def __init__(self,
                 features=None,
                 latency_dim=None,
                 epochs=None,
                 gpu_unit=None):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        
        for arg, val in values.items():
            setattr(self, arg, val)
        
        global optimizer_c,optimizer_d,optimizer_g
        
        self.features=features
        self.latency_dim=latency_dim
        self.epochs=epochs
        self.gpu_unit=gpu_unit
        
        self.generator=BuildGenerator(self.gpu_unit).to(device)
        #if (device.type=='cuda') and (self.gpu_unit>1):
            #self.generator=nn.DataParallel(self.generator,list(range(self.gpu_unit)))
            
        #self.generator.apply(DCGANAnalysis("").weights_init)
        #print(self.generator)
        optimizer_g = Adam(self.generator.parameters(),lr=learning_rate_g, betas=(beta1,0.999))
        
   

        
        self.discriminator=BuildDiscriminator(self.gpu_unit,self.features).to(device)
        #print(self.discriminator)      
        #if (device.type=='cuda') and (self.gpu_unit>1):
            #self.discriminator=nn.DataParallel(self.discriminator,list(range(self.gpu_unit)))
            
        #self.discriminator.apply(DCGANAnalysis("").weights_init)                

        optimizer_d = Adam(self.discriminator.parameters(),lr=learning_rate_d, betas=(beta1,0.999))




 
    def fit(self,            
            X,
            epochs=None,
            latency_dim=None):
        
        global scaler
        
        num_train = X.shape[0]
        start = 0

        #scaler=MinMaxScaler()
        generator_losses=[]
        discriminator_losses=[]
        audio_list=[]

        for epoch in range(self.epochs):

            self.discriminator.zero_grad()

            idx=np.random.randint(low=0,high=X.shape[0],size=batch_size)
            raw_data=X[idx]

            # Get a batch of real returns data...

            stop = start + batch_size
            raw_data = X[start:stop].to(device)
            #print('shape of raw data',raw_data.shape)
            output_real=self.discriminator(raw_data)
            #print('shape of output real data',output_real.shape)
            valid=torch.full(size=(output_real.size(0),output_real.size(1),output_real.size(2)),fill_value=1,dtype=torch.float,device=device)
            #print('shape of valid', valid.shape)
            d_loss_real = criterion(output_real, valid)                
            d_loss_real.backward()
            D_x=output_real.mean().item()

            # Train the discriminator.
            noise=torch.randn(raw_data.size(0),raw_data.size(1),raw_data.size(2),raw_data.size(3),device=device)
            generated_data = self.generator(noise)
            output_fake=self.discriminator(generated_data.detach())
            fake=torch.full(size=(output_fake.size(0),output_fake.size(1),output_fake.size(2)),fill_value=0,dtype=torch.float,device=device)

            d_loss_fake = criterion(output_fake,fake)
            d_loss_fake.backward()
            D_G_z1=output_fake.mean().item()
            d_loss = 0.5 * torch.add(d_loss_real, d_loss_fake)
            
            optimizer_d.step()
            
            self.generator.zero_grad()    
            output=self.discriminator(generated_data)
            g_loss = criterion(output, valid)
            g_loss.backward()
            D_G_z2=output.mean().item()

            optimizer_g.step()
            
            start +=batch_size
            

            if start>num_train-batch_size:
                start=0
     
            if epoch %5==0:
                print("Epoch={},\t batch={},\t Loss_D={:2.4f},\t Loss_G={},\t D_x={},\t D(G(z))_1={:2.4f},\t D(G(z))_2={}".format(epoch+1,i,d_loss.item(),g_loss.item(),D_x,D_G_z1,D_G_z2))

         
            
            generator_losses.append(g_loss.item())
            discriminator_losses.append(d_loss.item())
            
            epoch +=1
 

        return output


class MyDCGAN():
    
    def predict(self):
        
        global random_int, timeSequence, m, features, latency_dim, gpu_unit
        
        random_int=random.randint(0,10000)
        currentTime=datetime.datetime.now()
        timeSequence=str(object=currentTime)[20:26]
        


               
        x_train,y_train,x_test,y_test,features,df_consolidated,data_train,data_test,y,m,t=data_preprocessing.populationInit()   

        features=features
        latency_dim=t
        epochs=3000
        gpu_unit=1

        y_train_predict=[]
        for i, (batch_X,batch_Y) in enumerate (data_train):
                X=Variable(batch_X)
                Y=Variable(batch_Y)
                
                X,Y=X.to(device),Y.to(device)      
                dcgan = DCGANAnalysis(features=features,
                      latency_dim=latency_dim,
                      epochs=epochs,
                      gpu_unit=gpu_unit)
        
                y_train_predict_=dcgan.fit(X=X,epochs=epochs,latency_dim=latency_dim)

                y_train_predict.append(y_train_predict_.detach().cpu().numpy())
                
                i+=1
        
        y_train_predict=np.concatenate(y_train_predict,axis=0)

        y_train_predict=np.reshape(y_train_predict,(int(y_train_predict.shape[0]*y_train_predict.shape[1]),y_train_predict.shape[2]))
        y_train=np.reshape(y_train,(int(y_train.shape[0]*y_train.shape[1]),y_train.shape[2]))
        y_test_predict=[]
        for j, (batch_x_test,batch_y_test) in enumerate (data_test):
                X_=Variable(batch_x_test)   
                X_=X_.to(device)
          
                dcgan = DCGANAnalysis(features=features,
                      latency_dim=latency_dim,
                      epochs=epochs,
                      gpu_unit=gpu_unit)
        
                y_test_predict_=dcgan.fit(X=X_,epochs=epochs,latency_dim=latency_dim)

                y_test_predict.append(y_test_predict_.detach().cpu().numpy())
                j+=1

        trainScore=math.sqrt(mean_squared_error(y_train,y_train_predict)) 
        print('Train Score: %.5f RMSE' % (trainScore))


        
        y_test_predict=np.concatenate(y_test_predict,axis=0)

        y_test_predict=np.reshape(y_test_predict,(int(y_test_predict.shape[0]*y_test_predict.shape[1]),y_test_predict.shape[2]))
        y_test=np.reshape(y_test,(int(y_test.shape[0]*y_test.shape[1]),y_test.shape[2]))

        testScore=math.sqrt(mean_squared_error(y_test,y_test_predict)) 
        print('Test Score: %.5f RMSE' % (testScore))


        auc_value=roc_auc_score(y_test,y_test_predict)
        pauc_value=roc_auc_score(y_test,y_test_predict,max_fpr=0.1)
        print('auc: ',auc_value)
        print('pauc: ',pauc_value)
        
            
        y_predict=np.concatenate((y_train_predict,y_test_predict),axis=0)
        
        x = np.zeros(shape=(TotalSize, 1))
        x_mean = np.zeros(shape=TotalSize)
        for i in range (TotalSize):
            x[i, :] = y_predict[i][1]
            x_mean[i] = np.average(a=x[i, :])
        
        act_mean = np.zeros(shape=y.shape[0])
        for i in range(y.shape[0]):
            act_mean[i] = np.average(a=(y[i]))
            i+=1
     
        plotnine.options.figure_size = (25, 25)
        plot = ggplot(pd.melt(pd.concat([pd.DataFrame(x_mean, columns=["AutoEncoder CNN Distribution"]).reset_index(drop=True),
                                         pd.DataFrame(act_mean, columns=["Actual Distribution"]).reset_index(drop=True)],
                                        axis=1))) + \
        geom_density(aes(x="value",
                         fill="factor(variable)"), 
                     alpha=0.5,
                     color="black") + \
        geom_point(aes(x="value",
                       y=0,
                       fill="factor(variable)"), 
                   alpha=0.5, 
                   color="black") + \
        xlab("Value") + \
        ylab("Density") + \
        ggtitle("Trained AutoEncoder Convolutional Neural Network (AE-CNN) Predicted Anomaly Values") + \
        theme_matplotlib()
        plot.save(filename='output_'+str(object=n0)+'_'+str(object=n2)+'aecnn_dist.png', path=graph_dir_3)
        
        #history_dict=loss_history.history
        #history_dict.keys()

        y_test_=[]
        for i in range (len(y_test)):
            y_1=y_test[i][1]
            y_test_.append(y_1)

            i+=1


        y_test_predict_=[]
        for i in range (len(y_test_predict)):
            y_2_=y_test_predict[i][1]
            y_test_predict_.append(y_2_)

            i+=1


        fpr,tpr,thresholds=roc_curve(y_test_,y_test_predict_)
        roc_auc=auc(fpr,tpr)
    

        
        print(thresholds)
        
        print('length of fpr: ', len(fpr),len(tpr))
        print('roc_auc: ',roc_auc)
        
        
        
        plt.figure(3)
        lw=2
        plt.plot(fpr,tpr,color='orange', lw=lw,label='ROC Curve (area = %0.2f)' % roc_auc)
        plt.plot([0,1],[0,1],color='blue',lw=lw, linestyle='--')
        plt.xlim([0.0000,1.0000])
        plt.ylim([0.0000,1.0500])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Anomaly Sound Detection')
        plt.legend(loc='best')
        fig=plt.gcf()
        png_name_roc='output_aecnn_'+str(object=random_int)+'_'+str(object=n2)+'_'+timeSequence+'_roc_curve.png'
        plt.savefig(graph_dir_4+png_name_roc)
        plt.close()       

        precision,recall,thresholds=precision_recall_curve(y_test_,y_test_predict_)
        prc_auc=auc(recall,precision)

        
        print('length of precision: ', len(precision),len(recall))
        print('prc_auc: ',prc_auc)
        
        
        
        plt.figure(4)
        lw=2
        plt.plot(recall,precision,color='orange', lw=lw,label='ROC Curve (area = %0.2f)' % prc_auc)
        plt.plot([0,1],[0.5,0.5],color='blue',lw=lw, linestyle='--')
        plt.xlim([0.0000,1.0000])
        plt.ylim([0.0000,1.05000])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Anomaly Sound Detection')
        plt.legend(loc='best')
        fig=plt.gcf()
        png_name_roc='output_aecnn_'+str(object=random_int)+'_'+str(object=n2)+'_'+timeSequence+'_prc_curve.png'
        plt.savefig(graph_dir_5+png_name_roc)
        plt.close()      
        
        return x_test,y_test,x_train,y_test_predict,y_train_predict,y_test_,y_test_predict_
    
    def myVisualize(self):
        x_test,y_test,x_train,y_test_predict,y_train_predict,y_test_,y_test_predict_=MyDCGAN.predict("")
        Diff=[]
        count=0
        totalCount=0


        x_predict_=[]
        for i in range (0,len(y_test_predict_)):
            if y_test_predict[i][0]>=y_test_predict[i][1]:
                x_predict_.append(0)
            else:
                x_predict_.append(1)
            Diff.append(y_test_[i]-x_predict_[i])
            if Diff[i]==0:
                count=count+1
            else:
                count=count
            totalCount=totalCount+1

            i+=1

        x_predict_=np.array(x_predict_)
        x_predict_=np.reshape(x_predict_,(x_predict_.shape[0],1))
        y_=np.reshape(x_predict_,(x_predict_.shape[0]))
        
        print(x_predict_)
        d=np.concatenate((y_test_,y_),axis=0)
        df_output=pd.DataFrame(data=d)
        df_output.to_csv(csv_dir+'nlpcnn_output_'+timeSequence+'.csv')
        
        cm_predict=confusion_matrix(y_test_,x_predict_)
        auc_predict=roc_auc_score(y_test_,y_test_predict_,average='micro')
        pauc_predict=roc_auc_score(y_test_,y_test_predict_,average='micro',max_fpr=0.1)
        ari=adjusted_rand_score(y_test_,y_test_predict_)
        nmi=normalized_mutual_info_score(y_test_,y_test_predict_)
        
        fmeasure=f1_score(y_test_,y_,average='micro')
        ac_score=accuracy_score(y_test_,y_)
        print('auc_predict: ',auc_predict)
        print('pauc_predict: ',pauc_predict)
        print('cm_predict: ',cm_predict)
        print('ARI: ',ari)
        print('NMI: ',nmi)
        print('F Measure: ',fmeasure)
        print('Accuracy: ', ac_score)
        mse_= mean_squared_error(y_test_,y_test_predict_,multioutput='raw_values')
        mse= mean_squared_error(y_test_,x_predict_,multioutput='raw_values')
        print('mse = ',mse,mse_)
        
        f= open(log_dir+'svm_log_100_10_16_20211105.txt','a') 
        f.write('----------------------------------------------------\n')
        f.write('confusion matrix={}\n'.format(cm_predict))
        f.write('auc={}\n'.format(auc_predict))
        f.write('pauc={}\n'.format(pauc_predict))
        f.write('ARI={}\n'.format(ari))
        f.write('NMI={}\n'.format(nmi))
        f.write('F Measure={}\n'.format(fmeasure))
        f.write('Accuracy Score={}\n'.format(ac_score))
        f.write('binary fitness={}\n'.format(count/totalCount))
        f.write('mse={}\n'.format(mse))
        f.close()
                
        df_output = pd.DataFrame.from_records({'Actual':y_test_,'Predict':y_test_predict_,'Binary Prediction': y_},index='Actual')
        df_output.to_csv(csv_dir+'aecnn_output_'+timeSequence+'.csv')
        
        fig,ax=plt.subplots()
        
        ax.plot(y_test_,color='blue',label='actual')
        ax.set_xlabel('Count')
        ax.set_ylabel('Category(Actual)')
        ax2=ax.twinx()
        ax2.plot(y_test_predict_,color='red',label='prediction')
        ax2.set_ylabel('Category(Predict)')
        ax.legend()
        
        fig.set_size_inches(15,7)
        #plt.show()
        png_name_aecnn = 'prediction_aecnn_line_'+str(object=random_int)+'_'+str(object=n2)+'_'+timeSequence+'.png'
        fig.savefig(graph_dir+png_name_aecnn)
        plt.close()

        del x_test
        del y_test
        del y_train_predict
        del y_test_predict
        del x_predict_
        del y_
        del y_test_
        del y_test_predict_

        
if __name__=='__main__':
  for i in range(5):
    x=MyDCGAN()
    x.myVisualize()
    i+=1