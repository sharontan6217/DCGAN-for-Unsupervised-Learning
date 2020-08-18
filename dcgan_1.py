
"""
Created on Sat May 23 11:08:17 2020

@author: sharontan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:07:51 2019

@author: sharontan
"""

from PIL import Image

import struct
import wave
import sys
import pydub
from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub.silence import split_on_silence
import scipy
from scipy.io import wavfile
import pickle

import os
import math
import inspect
import sys
import importlib
import random

import numpy as np
from numpy import log10

import pandas as pd

import datetime
from datetime import timedelta

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import keras
from keras import backend as bkend
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras import layers
from keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten, convolutional, pooling, Reshape, concatenate, ZeroPadding2D, Conv2DTranspose
from keras.layers import LSTM, GRU, Bidirectional, BatchNormalization, TimeDistributed,Deconv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,MaxPooling2D,AveragePooling1D,AveragePooling2D,Conv1D
from keras import metrics
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, Adamax
from keras.layers.recurrent import LSTM
from keras import losses
from keras.utils.generic_utils import Progbar
from keras.layers.pooling import GlobalAveragePooling1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import L1L2



import datetime
from datetime import timedelta

import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize, scale

import tensorflow as tf
from tensorflow.python.client import device_lib

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from plotnine import *
import plotnine


# init data
wave_dir_1='Documents/data/pump/train/'
wave_dir_2='Documents/data/pump/test/'
chunk_dir='/Documents/chunk/'
csv_dir='/Documents/gan/csv/'
log_dir='/Documents/gan/log/'
graph_dir='/Documents/gan/graph/line/'
graph_dir_1='/Documents/gan/graph/loss/'
graph_dir_2='/Documents/gan/graph/error/'

currentTime=datetime.datetime.now()

os.environ["KERAS_BACKEND"] = "tensorflow"
importlib.reload(bkend)
print(device_lib.list_local_devices())

file_format='.wav'

sample_count=10

PopulationSize=16000
PredictSize=1600
GapSize=1600

evaluation_rate=0.05

currentTime=datetime.datetime.now()

timeSequence=str(object=currentTime)[20:26]


class data_preprocessing():

    def _init_(self):
        self.df=df
    
    def file_alias():
        
        file_string_a='normal_id_00_000000'
        file_string_b='anomaly_id_00_000000'
        

        file_name_a=[]
        file_name_b=[]
        for i in range(0,sample_count):
            file_number=random.randint(1,99)
            if file_number<10:
                file_number='0'+str(object=file_number)
            else:
                file_number=str(object=file_number)
            file_name_a.append(file_string_a+file_number)
            file_name_b.append(file_string_b+file_number)
            i+=1
        return file_name_a, file_name_b
    


        
    def populationInit():

        global n1,n2,n3
        
        #file_1=AudioSegment.from_wav(wave_dir_1+file_name_1+file_format)
        file_name_a, file_name_b=data_preprocessing.file_alias()
        i=0
        df_a=[]

        df_b=[]

        for i in range(0,sample_count):
            print(i,file_name_a[i])
            file_alias=wave_dir_1+file_name_a[i]+file_format
            Fs, audioData=wavfile.read(file_alias)
            n=audioData.size
            t=round(Fs/10)
            m=round(n/t)
            #print(m)
            
            wavFile=wave.open(file_alias)
            audioString=wavFile.readframes(wavFile.getnframes())
            audioText=struct.unpack('%ih' % (wavFile.getnframes()*wavFile.getnchannels()),audioString)
            audioText=[float(val)/pow(2,15) for val in audioText]
            print(len(audioText)/m,round(len(audioText)/m))
            audio_rows=round(len(audioText)/m)
            audio_cols=m
            textArray=np.array_split(audioText,round(len(audioText)/m))
            df=pd.DataFrame(data=textArray)
            problem=[]
            for j in range (round(len(audioText)/m)):
                problem.append(0)
                j+=1
            df['IssueOrNot']=problem
            df_a.append(df)

            file_alias_b=wave_dir_2+file_name_b[i]+file_format
            Fs_b, audioData_b=wavfile.read(file_alias_b)
            n_b=audioData_b.size
            t_b=round(Fs_b/10)
            m_b=round(n_b/t_b)
            #print(m)
            
            wavFile_b=wave.open(file_alias_b)
            audioString_b=wavFile_b.readframes(wavFile_b.getnframes())
            audioText_b=struct.unpack('%ih' % (wavFile_b.getnframes()*wavFile_b.getnchannels()),audioString_b)
            audioText_b=[float(val_b)/pow(2,15) for val_b in audioText_b]
            print(len(audioText_b)/m_b,round(len(audioText_b)/m_b))
            textArray_b=np.array_split(audioText_b,round(len(audioText_b)/m_b))
            df_=pd.DataFrame(data=textArray_b)
            problem_b=[]
            for i in range (round(len(audioText_b)/m_b)):
                problem_b.append(1)
                i+=1
            df_['IssueOrNot']=problem_b
            df_b.append(df_)
            #print(i)
            
            i+=1


        df_a=pd.concat(df_a,axis=0)
        df_b=pd.concat(df_b,axis=0)
        df=df_a.append(df_b)
        #print(len(df_a))
        #print(m)

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
        print(df_consolidated)
        return df_consolidated, audio_rows, audio_cols




class DCGANAnalysis():
    def __init__(self,
                 audio_rows=None,
                 audio_cols=None,
                 audio_channels=None,
                 latency_dim=None,
                 epochs=None,
                 batch_size=None):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        


        
        for arg, val in values.items():
            setattr(self, arg, val)
        
        global optimizer_c,optimizer_d,optimizer_g, audio_shape
        
        self.audio_rows=audio_rows
        self.audio_cols=audio_cols
        self.audio_channels=audio_channels
        audio_shape=(self.audio_rows,self.audio_cols,self.audio_channels,1)
        self.latency_dim=latency_dim
        optimizer_c = Adam(0.0002, 0.5)
        optimizer_d = Adam(0.0002, 0.5)
        optimizer_g = Adam(0.0002, 0.5)
        self.epochs=epochs
        self.batch_size=batch_size
        #self.gru_units=gru_units
        #self.X=X
        #self.X_=X_
        #self.y=y
        #self.noise=noise
        #self.valid=valid
        #self.raw=raw
        
        # Build the discriminator.
        

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=optimizer_d,loss='mean_squared_error',metrics=['mae'])


        

        
        self.generator = self.build_generator()
        self.generator.compile(optimizer=optimizer_g,loss='mean_squared_error',metrics=['mae'])

        noise = Input(shape=(self.latency_dim,))
        raw = self.generator(noise)       
        
        self.discriminator.trainable=False
        
        valid = self.discriminator(raw)
        
        # Set up and compile the combined model.
        self.cgan_generator = Model(noise,valid)
        self.cgan_generator.compile(optimizer=optimizer_c,loss='mean_squared_error',metrics=['mae'])
        self.cgan_generator.summary()

 
    def fit(self,
            X,
            y,
            z,
            y_valid,
            input_shape=None,
            batch_size=None,
            epochs=None,
            latency_dim=None):
        global scaler
        
        num_train = X.shape[0]
        start = 0
        
        
        
        # Adversarial ground truths.
        valid = np.ones((self.batch_size,1))  
        fake = np.zeros((self.batch_size,1))
        
        #scaler=MinMaxScaler()
        
        for step in range(self.epochs):
            idx=np.random.randint(low=0,high=X.shape[0],size=self.batch_size)
            raw_data=X[idx]
            # Generate a new batch of noise...
            noise = np.random.uniform(low=-1.0, high=1.0, size=(self.batch_size,self.latency_dim))
            #noise=np.reshape(noise,(self.batch_size,self.latency_dim,1,1))
            # ...and generate a batch of synthetic returns data.
            generated_data = self.generator.predict(noise)
            
            # Get a batch of real returns data...

            stop = start + self.batch_size
            raw_data = X[start:stop]
            print('shape of raw data',raw_data.shape)
            raw_data=np.reshape(raw_data,(raw_data.shape[0],raw_data.shape[1],1,1))


            # Train the discriminator.
            d_loss_real = self.discriminator.train_on_batch(raw_data, valid)
            d_loss_fake = self.discriminator.train_on_batch(generated_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            print('shape of noise',noise.shape)

            # Train the generator.
            
            #X=np.reshape(X,(X.shape[0],X.shape[1],1,1))
            #y=np.reshape(y,(y.shape[0],y.shape[1],1,1))
            #z=np.reshape(z,(z.shape[0],z.shape[1],1,1))
            #y_valid=np.reshape(y_valid,(y_valid.shape[0],y_valid.shape[1],1,1))

            history_callback=self.cgan_generator.fit(X,y,batch_size=batch_size,epochs=epochs,\
                                   verbose=2, validation_data=[z,y_valid],shuffle = True)

            g_loss = self.cgan_generator.train_on_batch(noise, valid)  
            
            start += self.batch_size
            if start > num_train - self.batch_size:
                start = 0
            
            if step % 100 == 0:
                # Plot the progress.
                print("[Discriminator loss: %f, Discriminator mae: %.2f%%] [Generator loss: %f]" % (d_loss[0], 100 * d_loss[1], g_loss[0]))
        return self

    
    def build_generator(self):
        # We will map z, a latent vector, to continuous returns data space (..., 1).

        model = Sequential()


        #print(input.shape[0],input.shape[1])
        model.add(layers.Dense(256*1*25, activation="relu", input_dim=self.latency_dim))
        print('output_shape:',model.output_shape)
        #model.add(layers.Reshape((25,1,128)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        print(model.output_shape)
        model.add(layers.Reshape((25,1,256)))
        assert model.output_shape==(None,25,1,256)
        model.add(layers.Deconv2D(filters = 256, kernel_size =(5,5),strides=(1,1),padding='same',use_bias=False))
        print(model.output_shape)
        assert model.output_shape==(None,25,1,256)
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        #model.add(layers.LeakyReLU())
        #print(model.output_shape)
        model.add(layers.Deconv2D(filters = 128, kernel_size =(5,5),strides=(2,1),activation='relu',padding='same',use_bias=False))
        print(model.output_shape)
        assert model.output_shape==(None,50,1,128)
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        #model.add(layers.LeakyReLU())
        #print(model.output_shape)
        model.add(layers.Deconv2D(filters = 64, kernel_size =(5,5),strides=(2,1),activation='relu',padding='same',use_bias=False))
        print(model.output_shape)
        assert model.output_shape==(None,100,1,64)
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        #model.add(layers.LeakyReLU())
        #print(model.output_shape)
        model.add(layers.Conv2DTranspose(filters = self.audio_channels, kernel_size =(5,5),strides=(1,1),activation='relu',padding='same',use_bias=False))
        print(model.output_shape)
        assert model.output_shape==(None,100,1,1)
        #model.add(Flatten())
        print(model.output_shape)
        model.add(layers.Activation("tanh"))




        #print (model.output_shape)
        '''
        model.add(Dense(units=1))
        model.add(GlobalAveragePooling1D())
        '''
        #model.add(Reshape(self.input_shape))
        model.summary()
        print (model.summary())
        
        noise = Input(shape=(self.latency_dim,))
        raw_data = model(noise)

        print('generator shape',raw_data.shape)

        
        model.compile(loss='mean_squared_error',optimizer=optimizer_g,metrics=['mae'])
        
        '''
        history_callback=model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,\
                                   verbose=2, validation_data=[x_test,y_test],shuffle = True)
        '''
        
        return Model (noise, raw_data)
    
    def build_discriminator(self):

        model = Sequential()

        model.add(layers.Conv2D(filters = 64, kernel_size =(5,5),strides=(2,2),input_shape=[100,1,1],padding='same',kernel_initializer='uniform'))
        print(model.output_shape)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.4))
        print(model.output_shape)
        model.add(Conv2D(filters = 128, kernel_size = (5,5),strides=(2,2),padding='same',kernel_initializer='uniform'))
        model.add(layers.BatchNormalization())
        #print(model.output_shape)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.4))
        print(model.output_shape)
        model.add(Conv2D(filters = 256, kernel_size = (5,5),strides=(2,2),padding='same',kernel_initializer='uniform'))
        model.add(layers.BatchNormalization())
        #print(model.output_shape)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.4))
        print(model.output_shape)
        model.add(Conv2D(filters = 512, kernel_size = (5,5),strides=(2,2),padding='same',kernel_initializer='uniform'))
        model.add(layers.BatchNormalization())
        #print(model.output_shape)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.4))
        print(model.output_shape)
        #model.add(AveragePooling1D(pool_size=1,padding='valid'))
        model.add(layers.Flatten())

        model.add(layers.Dense(512,activation='relu'))
        #model.add(Dense(90,kernel_initializer='uniform'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.4))
        #model.add(keras.layers.core.Reshape([input.shape[2],input.shape[1]]))
        model.add(layers.Dense(1,activation='sigmoid'))



        #print (model.output_shape)
        '''
        model.add(Dense(units=1))
        model.add(GlobalAveragePooling1D())
        '''
        #model.add(Reshape(self.input_shape))

        print('raw_data',model.output_shape)

        


        model.summary()
        print (model.summary())
        
        raw_data = Input(shape=(100,1,1))
        valid = model(raw_data)

        model.compile(loss='mean_squared_error',optimizer=optimizer_d,metrics=['mae'])
        
        return Model(raw_data,valid)


    
    def data_load(self):
        global realSize, log_dir, graph_dir_1, graph_dir_2, fileName,n,n1,n2,n3
        
        df_consolidated, audio_rows, audio_cols=data_preprocessing.populationInit()
        df_=np.array(df_consolidated.values)
        n=0
        #n=random.randint(0,4800)
        n1=n+PopulationSize
        n2=n+PopulationSize+GapSize
        n3=n+PopulationSize+GapSize+PredictSize

        #print(n,n1,n2)
        



        #print('x_train is', x_train)
        #print('y_train is', y_train)
        #print('x_test is', x_test)
        #print('y_test is', y_test)
        #print(y_train.shape[0],y_train.shape[1])
        #print(len(x_train),x_train.shape[0],x_train.shape[1])
        print (len(df_consolidated.columns))
        features=len(df_consolidated.columns)-1

        x_train=df_[n:n1,:][:,-features-1:-1]
        print(x_train)
        y_train=df_[n:n1,:][:,-1:]
        print(y_train)
        x_test=df_[n2:n3,:][:,-features-1:-1]
        y_test=df_[n2:n3,:][:,-1:]
        #print(features)
        #x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1,1))
        y_train=np.reshape(y_train,(y_train.shape[0],y_train.shape[1]))
        #print(x_train.shape[0],x_train.shape[1],features,len(x_train))
        #print(x_train)
        #x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1,1))
        y_test=np.reshape(y_test,(y_test.shape[0],y_test.shape[1]))     
        #print(x_test)
        #y_train=np.reshape(y_train,(y_train.shape[0],1))
        #print(y_train)
        return (x_train,y_train,x_test,y_test,features,df_consolidated,audio_rows, audio_cols)

    
    def predict (self):
        
        x_train,y_train,x_test,y_test,features,df_consolidated,audio_rows, audio_cols= DCGANAnalysis.data_load("")
        

        batch_size=128
        audio_rows=1600
        audio_cols=features
        audio_channels=1
        latency_dim=100
        epochs=1000
        #drop_out=0.2
        #patience=5
        #gru_units=90
        #dense_units=10
        input_shape=(100,)
        
        cgan = DCGANAnalysis(audio_rows=audio_rows,
                      audio_cols=audio_cols,
                      audio_channels=audio_channels,
                      latency_dim=latency_dim,
                      epochs=epochs,
                      batch_size=batch_size)
        
        cgan.fit(X=x_train,y=y_train,z=x_test,y_valid=y_test,\
                     input_shape=input_shape,\
                     batch_size=batch_size,epochs=epochs)
        
        n_sim = len(x_train)
        noise_train = np.random.uniform(low=-1.0, high=1.0, size=(n_sim, features))
        #noise_train = noise_train.reshape(n_sim,latency_dim,1,1)
        y_predict = np.zeros(shape=(n_sim,1))
        print(enumerate(noise_train))
        for i, xi in enumerate(noise_train):  
            print(xi)
            y_predict[i, :] = cgan.generator.predict(x=xi)[0]
            i+=1
      

        n_test = len(x_test)
        noise_test = np.random.uniform(low=-1.0, high=1.0, size=(n_test, features))
        #noise_test = np.reshape(noise_test,(noise_test.shape[0],noise_test.shape[1],1,1))
        x_predict = np.zeros(shape=(n_test,1))
        for i, xi in enumerate(noise_test):  
            x_predict[i, :] = cgan.generator.predict(x=xi)[0]
            i+=1

        print(x_actual,x_predict,np.count_nonzero(x_actual),np.count_nonzero(x_predict))


        #print(np.count_nonzero(x),np.count_nonzero(x))
        #print(z_test,x)


        x_predict=np.asarray(x_predict)
        #print(x_actual,x_predict,d_predict)

        #print(np.count_nonzero(d_predict))
        #print(d_predict)
        #print(np.count_nonzero(x_predict),np.count_nonzero(x_pre))
        #print(x_predict)
     
     
       #generator
        return (x_test,y_test,x_predict,x_train,y_train,y_predict)

class MyDCGAN():      
    def DCGANvisualize(self):
        from sklearn.metrics import mean_squared_error
        
        x_test,y_test,x_predict,x_train,y_train,y_predict = DCGANAnalysis.predict("")
        #print(np.count_nonzero(x_train))


        x=[]
        y=[]
        x_predict_=[]
        for i in range (len(x_predict)):
            if x_predict[i]>0.5:
                x_predict_.append(1)
            else:
                x_predict_.append(0)
            x.append(y_test[i])
            y.append(x_predict[i])
            i+=1
        y_predict_=[]
        for i in range (len(y_predict)):
            if y_predict[i]>0.5:
                y_predict_.append(1)
            else:
                y_predict_.append(0)
            i+=1
        x_predict_=np.array(x_predict_)
        x_predict_=np.reshape(x_predict_,(x_predict_.shape[0],1))
        y_predict_=np.array(y_predict_)
        y_predict_=np.reshape(y_predict_,(y_predict_.shape[0],1))
        #print(x_predict_)
        d=np.concatenate((y_test,x_predict_),axis=1)
        df_output=pd.DataFrame(data=d)
        #df_output = pd.DataFrame.from_records({'Actual':y_test,'Predict':x_predict_},index='Actual')
        df_output.to_csv(csv_dir+'nlpcnn_output_'+timeSequence+'.csv')
        

        c=0
        c_=0
        for i in range(PredictSize): 
            if np.array(np.abs(x[i]-y[i]))<=evaluation_rate:
               c=c+1
            else:
               c=c     
            if np.array(np.abs(x_predict_[i]-y_test[i]))==0:
               c_=c_+1
            else:
               c_=c_
            i+=1
        fitness_total=c/PredictSize
        fitness_sub=c/len(x)
        fitness_simple=c_/len(x)
        mse= mean_squared_error(x,y,multioutput='raw_values')
        avg_diff=np.average(d)
        print('total fitness=',fitness_total)
        print('fitness=',fitness_sub) 
        print('binary fitness=',fitness_simple)
        print('mse=',mse)
        print('average of difference=',avg_diff)


    #generate output log
        f= open(log_dir+'log.txt','a') 
        f.write('----------------------------------------------------\n')
        f.write('total fitness={}\n'.format(fitness_total))
        f.write('fitness={}\n'.format(fitness_sub))
        f.write('binary fitness={}\n'.format(fitness_simple))
        f.write('mse={}\n'.format(mse))
        f.write('average of difference={}\n'.format(avg_diff))
        f.close()
        
        plt.plot(y_predict_,color='red',label='prediction')
        plt.plot(y_train,color='blue',label='actual')
        plt.xlabel('Counts')
        plt.ylabel('Validity')
        plt.legend()
        fig = plt.gcf()
        fig.set_size_inches(15,7)
        #plt.show()
        #print(timeSequence)
        png_name_cnn_1 = 'train_cnn_line_'+str(object=n2)+'_'+timeSequence+'.png'
        plt.savefig(graph_dir+png_name_cnn_1)
        plt.close()
        
        plt.plot(x_predict_,color='red',label='prediction')
        plt.plot(y_test,color='blue',label='actual')
        plt.xlabel('Counts')
        plt.ylabel('Validity')
        plt.legend()
        fig = plt.gcf()
        fig.set_size_inches(15,7)
        #plt.show()
        png_name_cnn_2 = 'prediction_cnn_line_'+str(object=n2)+'_'+timeSequence+'.png'
        plt.savefig(graph_dir+png_name_cnn_2)
        plt.close()

 

        del x_test
        del y_test
        del x_predict
        del x_train
        del y_train
        del y_predict
        del x
        del y

 
        
    




if __name__=='__main__':
    x=MyDCGAN()
    #x.data_load()
    #x.predict()
    x.DCGANvisualize()

    #x.clean() 
        