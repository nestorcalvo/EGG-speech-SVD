#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:47:13 2020

@author: HP
"""
from scipy.io.wavfile import read, write
from scipy.signal import resample
from scipy.stats import kurtosis
from scipy.stats import skew
import pysptk
import matplotlib.pyplot as plt
import numpy as np

from os import listdir
from os.path import isfile, join
import os
import pandas as pd
import parselmouth   #praat
import sys


def norm_fs(x,fs,fs_min):
    x_new = resample(x, round(fs_min*len(x)/fs))
    return x_new

def energy(sig):         # Definir la funcion que encuentra energía
    sig2 = np.square(sig)  # Elevar al cuadrado las muestras de la senal
    sumsig2 = np.sum(sig2) # Sumatoria
    return sumsig2



def voiced(file_name,sig,t_ventana,fs):
    snd = parselmouth.Sound(file_name)
    pitch = snd.to_pitch(time_step = t_ventana)
    pitch_values = pitch.selected_array['frequency']
    
    muestras = int(t_ventana*fs)     #Divididir señal en segmentos de 10ms
    segments = int(len(sig)/muestras)
    newsound = np.empty([0])

    
    for i in range(len(pitch_values)):
        trozo = sig[muestras*i:(i+1)*muestras]

        if pitch_values[i] > 0:
            newsound = np.hstack((newsound,trozo)) #conservar segmento            
    return newsound

def get_mfcc(sig,fs,t_ventana):
    mat = np.zeros([0,20])
    muestras = int(t_ventana*fs)
    half = round(muestras/2)
    
    for i in range(round( 2*len(sig)/muestras - 1)):
        vent = sig[half*i:half*(i+muestras)]
        coeficientes = pysptk.sptk.mfcc(vent, order = 20, num_filterbanks =  21)
        mat = np.vstack((mat,coeficientes))
        
    return mat
  
def mfcc_vector(mfcc_mat):
    forma = mfcc_mat.shape[1]
    vect_mfcc = np.zeros(forma*4)
    
    for i in range(forma):
        vect_mfcc[4*i] = np.mean(mfcc_mat[:,i])
        vect_mfcc[4*i + 1] = np.std(mfcc_mat[:,i])
        vect_mfcc[4*i + 2] = skew(mfcc_mat[:,i])
        vect_mfcc[4*i + 3] = kurtosis(mfcc_mat[:,i])
    
    return vect_mfcc

def MFCC_feature_generate(audio_path, save_path):
    files = [f for f in listdir(audio_path) if isfile(join(audio_path, f))]
    
    fs_list = []
    ex_len_list = []
    dis_len_list = []
    
    for f in files:     #Recorrrer lista de disparos
        f = os.path.join(audio_path,f)
        fs, x = read(f)
        fs_list.append(fs)
        dis_len_list.append(len(x)/fs)
    
    
    #fs_min = min(fs_list)
    fs_min = 20000
    
    mat_dis = np.zeros([0,80])      # Matriz de caracteristicas vacia
    en_tot = np.zeros([0])          # energias
    
    #for file_audio in filesexplosion:
    file_list = []
    for count,file_audio in enumerate(files):  
        file_audio_name = file_audio
        file_audio = os.path.join(audio_path, file_audio)
        fs, x = read(file_audio)
        x = norm_fs(x,fs,fs_min)              # Normalizar frecuencia de muestreo
        t=np.arange(0, len(x)/fs_min, 1.0/fs_min)
        
        #try:
          #  x.shape[1] == 2                 # Normalizar canal
         #   x = (x[:,0])/2+(x[:,1])/2      
        #except: print('Oh no!!!') 
        
        # t=np.arange(0, float(len(x))/fs_min, 1.0/fs_min)
        
        x = x - np.mean(x)                # Restar media
        x = x/float(max(abs(x)))          # escalar la amplitud de la senal
        
        seg_voz = [] #Creamos una nueva variable donde almacenaremos el segmento
        
        # A continuación solo guardaremos un segmento de la señal
        # perteneciente al tiempo transcurrido entre 2.5 seg y 3.5
        # segundos
        
        for i in range(0,len(x)):
            if ((t[i]>=0.2) and (t[i]<=0.8)):
                seg_voz.append(x[i])
        matriz_mfcc = get_mfcc(x,fs_min,0.04)   #Matriz de MFCC
        
        v_carac = mfcc_vector(matriz_mfcc)
        mat_dis = np.vstack((mat_dis,v_carac))  #Matriz de caracteristicas
        file_list.append(file_audio_name)
        if count%10==0:
            print(count)
        
    #----------Prueba guardar datos---------------------------
    col_names = []
    for i in range(20):
        col_names.append('MFCC_'+str(i+1)+'_mean')
        col_names.append('MFCC_'+str(i+1)+'_std')
        col_names.append('MFCC_'+str(i+1)+'_skew')
        col_names.append('MFCC_'+str(i+1)+'_kurt')
    
    df = pd.DataFrame(data=mat_dis.astype(float))
    df.columns = col_names
    
    df["file_name"] = file_list
    cols = list(df)
    cols.insert(0, cols.pop(cols.index('file_name')))
    df = df.loc[:, cols]
    
    df.to_csv(save_path, sep=',', header=True, index=False)


