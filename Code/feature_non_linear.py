#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 01:43:03 2022

@author: nestor
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample
from os import listdir
from os.path import isfile, join
import os
from scipy.io.wavfile import read, write
import sys
import nolds as ns
def norm_fs(x,fs,fs_min):
    x_new = resample(x, round(fs_min*len(x)/fs))
    return x_new
def get_features(sig,fs,t_ventana):
    mat = np.zeros([0,20])
    muestras = int(t_ventana*fs)
    half = round(muestras/2)
    sample_entropy = []
    LLE = []
    hurst_exponent = []
    
    for i in range(round( 2*len(sig)/muestras - 1)):
        vent = sig[half*i:half*(i+muestras)]
        sample_entropy.append(ns.sampen(vent))
        LLE.append(ns.lyap_r(vent))
        hurst_exponent.append(ns.hurst_rs(vent))
        
    
    return np.mean(sample_entropy), np.std(sample_entropy),np.mean(LLE), np.std(LLE),np.mean(hurst_exponent), np.std(hurst_exponent)

def non_linear_feature_generate(PATH, save_path):
    files = [f for f in listdir(PATH) if isfile(join(PATH, f))]
    
    fs_list = []
    ex_len_list = []
    dis_len_list = []
    
    for f in files:     #Recorrrer lista de disparos
        f = os.path.join(PATH,f)
        fs, x = read(f)
        fs_list.append(fs)
        dis_len_list.append(len(x)/fs)
    
    
    #fs_min = min(fs_list)
    fs_min = 20000
    
    mat_dis = np.zeros([len(files),3])      # Matriz de caracteristicas vacia
    en_tot = np.zeros([0])          # energias
    file_list = []
    for count,file_audio in enumerate(files):  
        file_audio_name = file_audio
        file_audio = os.path.join(PATH, file_audio)
        fs, x = read(file_audio)
        x = norm_fs(x,fs,fs_min)              # Normalizar frecuencia de muestreo
        t=np.arange(0, len(x)/fs_min, 1.0/fs_min)
        
        x = x - np.mean(x)                # Restar media
        x = x/float(max(abs(x)))          # escalar la amplitud de la senal
        print("Extract features from signal")
        mat_dis[count][0] = ns.sampen(x[0:int(0.1*fs)])
        mat_dis[count][1] = ns.lyap_r(x[0:int(0.1*fs)])
        mat_dis[count][2] = ns.hurst_rs(x[0:int(0.1*fs)])
        #mat_dis[count] = get_features(x,fs_min,0.1) 
    
        
    file_list = []
    for count,file_audio in enumerate(files):  
        file_list.append(file_audio)
    
    col_names = []
    col_names.append("sampen")
    col_names.append("LLE")
    col_names.append("hurst_exponent")
    df_non_linear = pd.DataFrame(data=mat_dis.astype(float))
    df_non_linear.columns = col_names
    
    df_non_linear["file_name"] = file_list
    cols = list(df_non_linear)
    cols.insert(0, cols.pop(cols.index('file_name')))
    df_non_linear = df_non_linear.loc[:, cols]
    
    df_non_linear.to_csv(save_path, sep=',', header=True, index=False)