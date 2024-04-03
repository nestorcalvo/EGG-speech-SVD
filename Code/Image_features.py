#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 04:23:13 2023

@author: nestor
"""
import numpy as np
import phase_heat as phh
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from os import walk
import pickle
from tqdm import tqdm
#****************************************************************************

if __name__ == '__main__':
    
    path = '/home/nestor/Documents/Maestria/Avances Maestria/Databases Masters/Database Merge'
    file_path = 'embeddings.pickle'
    

    with open(file_path, 'ab+') as fp:
        pickle.dump([], fp)
        fp.close()

    f = []
    for (dirpath, dirs, files) in walk(path):
        f = files
    subjects = []
    data = []
    for index, i in enumerate(tqdm(f)):
        filename = os.path.join(dirpath, i)
        number = i.split('-')[0]
        #Read signal
        fs,signal = read(filename)
        
        #Signal normalization
        signal = signal-np.mean(signal)
        signal = signal/np.max(np.abs(signal))
        
        
        #Phase plot features
        img = phh.get_phase_plots(signal,fs)
        
        if index == 0:
            np_img = np.array(img)    
            subjects = [number]*np_img.shape[0]
            data1 = {"embeddings": np_img}
        else:
            temp = np.array(img)    
            np_img = temp
            #np_img = np.vstack((np_img, temp))
            temp_2 = [number]*temp.shape[0]
            subjects = np.hstack((subjects,temp_2))
            file = open(file_path, 'rb')
            old_data = pickle.load(file)
            new_embeddings = old_data['embeddings']
            new_embeddings = np.vstack((new_embeddings,np_img))
            data1 = {"embeddings": new_embeddings}
        with open(file_path, 'wb') as fp:
            pickle.dump(data1, fp)
            fp.close()   

        
#%%
file = open(file_path, 'rb')
old_data = pickle.load(file)
