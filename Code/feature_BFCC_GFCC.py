#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 19:36:34 2022

@author: nestor
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from scipy.io.wavfile import read
from spafe.features.bfcc import bfcc
from spafe.features.gfcc import gfcc
from spafe.utils.vis import show_features
from spafe.utils.preprocessing import SlidingWindow
from scipy.signal import resample
from os import listdir
from os.path import isfile, join
from scipy.stats import skew, kurtosis

def norm_fs(x,fs,fs_min):
    x_new = resample(x, round(fs_min*len(x)/fs))
    return x_new
# =============================================================================
# PATH = '/home/nestor/Documents/Maestria/Avances Maestria/Databases Masters/Database Merge WAV'
# files = [f for f in listdir(PATH) if isfile(join(PATH, f))]
# fs_list = []
# ex_len_list = []
# dis_len_list = []
# 
# for f in files:     #Recorrrer lista de disparos
#     f = os.path.join(PATH,f)
#     fs, x = read(f)
#     fs_list.append(fs)
#     dis_len_list.append(len(x)/fs)
# 
# 
# #fs_min = min(fs_list)
# fs_min = 20000
# =============================================================================


def BFCC_feature_generate(PATH, save_path):
    num_filters = 32
    num_ceps = 13
    fs_min = 20000
    mat_bfcc = np.zeros([0,num_ceps*4]) 
    file_list = []
    files = [f for f in listdir(PATH) if isfile(join(PATH, f))]
    for count,file_audio in enumerate(files):  
        
        file_audio_name = file_audio
        file_audio = os.path.join(PATH, file_audio)
        
        
        fs, x = read(file_audio)
        x = norm_fs(x,fs,fs_min)              # Normalizar frecuencia de muestreo
        t=np.arange(0, len(x)/fs_min, 1.0/fs_min)
        x = x - np.mean(x)                # Restar media
        x = x/float(max(abs(x))) 
    
        matriz_bfcc = bfcc(x,fs_min,window=SlidingWindow(0.025, 0.01, "hamming"),num_ceps =num_ceps ,nfilts= num_filters)
        feat_vect = np.empty(0)        
        for i in range(num_ceps):
            media  = np.mean(matriz_bfcc[:,i])
            stand = np.std(matriz_bfcc[:,i])
            skewn = skew(matriz_bfcc[:,i])
            kurto = kurtosis(matriz_bfcc[:,i])
            
            feat_vect = np.hstack((feat_vect,[media, stand, skewn, kurto]))
        mat_bfcc = np.vstack((mat_bfcc,feat_vect))
        
        file_list.append(file_audio_name)
        
    
    
        if count%100==0:
            print(count)
        
    col_names = []
    for i in range(num_ceps):
        col_names.append('BFCC_'+str(i+1)+'_mean')
        col_names.append('BFCC_'+str(i+1)+'_std')
        col_names.append('BFCC_'+str(i+1)+'_skew')
        col_names.append('BFCC_'+str(i+1)+'_kurt')
    df = pd.DataFrame(data=mat_bfcc.astype(float))
    df.columns = col_names
    df["file_name"] = file_list
    cols = list(df)
    cols.insert(0, cols.pop(cols.index('file_name')))
    df = df.loc[:, cols]
    df.to_csv(save_path, sep=',', header=True, index=False)



def GFCC_feature_generate(PATH, save_path):
    num_filters = 32
    num_ceps = 13
    fs_min = 20000
    mat_gfcc = np.zeros([0,num_ceps*4]) 
    file_list = []
    
    files = [f for f in listdir(PATH) if isfile(join(PATH, f))]
    for count,file_audio in enumerate(files):  
        file_audio_name = file_audio
        file_audio = os.path.join(PATH, file_audio)
        fs, x = read(file_audio)
        x = norm_fs(x,fs,fs_min)              # Normalizar frecuencia de muestreo
        t=np.arange(0, len(x)/fs_min, 1.0/fs_min)
        x = x - np.mean(x)                # Restar media
        x = x/float(max(abs(x))) 
    
        matriz_gfcc = gfcc(x,fs_min,window=SlidingWindow(0.025, 0.01, "hamming"),num_ceps =num_ceps ,nfilts= num_filters)
        feat_vect = np.empty(0)        
        for i in range(num_ceps):
            media  = np.mean(matriz_gfcc[:,i])
            stand = np.std(matriz_gfcc[:,i])
            skewn = skew(matriz_gfcc[:,i])
            kurto = kurtosis(matriz_gfcc[:,i])
            
            feat_vect = np.hstack((feat_vect,[media, stand, skewn, kurto]))
        mat_gfcc = np.vstack((mat_gfcc,feat_vect))
        
        file_list.append(file_audio_name)
        
    
    
        if count%100==0:
            print(count)
    
    col_names = []
    for i in range(num_ceps):
        col_names.append('GFCC_'+str(i+1)+'_mean')
        col_names.append('GFCC_'+str(i+1)+'_std')
        col_names.append('GFCC_'+str(i+1)+'_skew')
        col_names.append('GFCC_'+str(i+1)+'_kurt')
    df = pd.DataFrame(data=mat_gfcc.astype(float))
    df.columns = col_names
    df["file_name"] = file_list
    cols = list(df)
    cols.insert(0, cols.pop(cols.index('file_name')))
    df = df.loc[:, cols]
    
    df.to_csv(save_path, sep=',', header=True, index=False)