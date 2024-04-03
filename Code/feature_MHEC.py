#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:32:34 2022

@author: nestor
"""
import gammatone.filters as fi
import matplotlib.pyplot as plt
import numpy as np 
from scipy.io.wavfile import read
from scipy.signal import hilbert
from scipy.signal import resample
import scipy.signal as sp
from scipy.fftpack import dct
from scipy.stats import skew, kurtosis
import pandas as pd
import parselmouth   #praat

from os import listdir
from os.path import isfile, join

def butter_lowpass_filter(data,cutoff,fs,order):
    nyq_rate = fs/2
    normal_cutoff = cutoff/nyq_rate
    b,a = sp.butter(order,normal_cutoff,btype='low',analog=False)
    y = sp.filtfilt(b,a,data)
    return y

def norm_fs(x,fs,fs_min):
    x_new = resample(x, round(fs_min*len(x)/fs))
    return x_new

PATH = '/home/nestor/Documents/Maestria/Avances Maestria/Databases Masters/Database Merge'
files = [f for f in listdir(PATH) if isfile(join(PATH, f))]

file_list = []
final_mat = []
df = pd.DataFrame()
mat_dis = np.zeros([0,128]) 
for count,audio in enumerate(files):
    
    file_audio_name = audio
    file_audio = join(PATH, audio)
    fs, senal = read(file_audio)
    
    total = len(senal)
    fs_min = 20000
    frame_size = round(0.025*fs_min)
    skip = round(0.01*fs_min)
    

    senal = senal - np.mean(senal)
    senal = senal/(max(abs(max(senal)),abs(min(senal))))
    
    #senal = voiced(file_audio,senal,0.01,fs) 
    senal = norm_fs(senal,fs,fs_min) 
    
    start=0
    frames = 0
    while start + frame_size <= len(senal):
        frames = frames + 1
        start = start + skip
    
    t = np.arange(len(senal)) / fs
    num_gammatone_filters = 32
    centre_f = fi.erb_space(low_freq=200, high_freq=3400, num=num_gammatone_filters)
    coefs = fi.make_erb_filters(fs, centre_f, width=1.0)
    
    signals = fi.erb_filterbank(senal, coefs)
    Mat_prom_s = np.empty((0,frames))
    for gammatone_filtler_index,_ in enumerate(signals):
        #print(gammatone_filtler_index)
        analytic = hilbert(signals[gammatone_filtler_index,:])
        amplitude_envelope = np.abs(analytic)
        env_s = butter_lowpass_filter(amplitude_envelope,20,fs,3)
        
        MATs_lj = np.empty((0,frame_size))
        start = 0
        
        while start + frame_size <= len(senal):
        
            block = env_s[start:start+frame_size]
            
            window = sp.hamming(frame_size)
            
            s_lj = block*window
            
            # plt.figure()
            # plt.plot(s_lj)
            
            MATs_lj = np.vstack((MATs_lj,s_lj)) 
            
            start = start + skip
        
        Mat_prom_s = np.vstack((Mat_prom_s,list(np.sum(MATs_lj,axis=1)/frame_size)))
    
    Mat_prom_s = Mat_prom_s**(1/15)
    
    Mat_prom_sdct = dct(Mat_prom_s)
    feat_vect = np.empty(0)        
    for i in range(num_gammatone_filters):
        media  = np.mean(Mat_prom_sdct[i,:])
        stand = np.std(Mat_prom_sdct[i,:])
        skewn = skew(Mat_prom_sdct[i,:])
        kurto = kurtosis(Mat_prom_sdct[i,:])
        
        feat_vect = np.hstack((feat_vect,[media, stand, skewn, kurto]))
    #df['file_audio_name'] = feat_vect
    mat_dis = np.vstack((mat_dis,feat_vect))
    
    file_list.append(file_audio_name)
    


    if count%10==0:
        print(count)
    
#%%
col_names = []
for i in range(32):
    col_names.append('MHCC_'+str(i+1)+'_mean')
    col_names.append('MHCC_'+str(i+1)+'_std')
    col_names.append('MHCC_'+str(i+1)+'_skew')
    col_names.append('MHCC_'+str(i+1)+'_kurt')
df = pd.DataFrame(data=mat_dis.astype(float))
df.columns = col_names
df["file_name"] = file_list
cols = list(df)
cols.insert(0, cols.pop(cols.index('file_name')))
df = df.loc[:, cols]
#%%
df.to_csv('MHCC_a_vowel_egg.csv', sep=',', header=True, index=False)