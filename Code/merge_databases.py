#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:49:06 2022
@author: nestor
"""
import pandas as pd
import os
import numpy as np
import os.path
from os import listdir
from os.path import isfile, join
import shutil
import seaborn as sns
from os.path import exists
import librosa 
import soundfile as sf
from scipy.signal import resample

#%%
databases = ['SVD', 'Keele', 'KED', 'FDA']
save_path = '/home/nestor/Documents/Maestria/Avances Maestria/Databases Masters/Database Merge WAV'
databases_path = ['../Databases Masters/SVD',
                  '../Databases Masters/KEELE Pitch Database',
                  '../Databases Masters/KED Database/cmu_us_ked_timit',
                  '../Databases Masters/FDA Database']

SVD = databases_path[0]
list_pathological = [join(SVD,x) for x in listdir(SVD)]
for folder in list_pathological:
    files = [join(folder,x) for x in listdir(folder)]
    for file in files:
# =============================================================================
#         # Copy EGG for vowel A
#         if ("egg.wav" in file.split('-') and "a_n" in file.split('-')):
#             shutil.copy(file,save_path)
# =============================================================================
        # Copy WAV for vowel A
        if ("a_n.wav" in file.split('-')):
            print(file)
            shutil.copy(file,save_path)

#%%
info_subjects = '/home/nestor/Documents/Maestria/Avances Maestria/Databases Masters/Database_info.ods'
save_path = '/home/nestor/Documents/Maestria/Avances Maestria/Databases Masters/Database Merge WAV'
metadata_SVD = pd.read_excel(info_subjects, sheet_name="SVD", engine='odf')
metadata_SVD = metadata_SVD.drop(columns=(['i wav','u wav','phrase wav','i egg','u egg','phrase egg']))
#%%
metadata_SVD =  metadata_SVD[metadata_SVD['a wav'].notna()]
list_files = listdir(save_path)
id_in_files = []
for file in list_files:
    id_in_files.append(int(file.split('-')[0]))
metadata_SVD_no_duplicates = metadata_SVD[metadata_SVD['ID'].isin(id_in_files)].drop_duplicates(subset = ["ID"], keep = 'first')
#%%
PD = metadata_SVD_no_duplicates[metadata_SVD_no_duplicates['T']=='p'].copy()
HC = metadata_SVD_no_duplicates[metadata_SVD_no_duplicates['T']=='n'].copy()
#%% Change frequency audios
audios = "/home/nestor/Documents/Maestria/Avances Maestria/Databases Masters/Database Merge WAV"
for item in listdir(audios):
    data, samplerate = sf.read(join(audios,item))
    
#%%
audios = "/home/nestor/Documents/Maestria/Avances Maestria/Databases Masters/Saarbruecken Voice Database/HC/WAV/a vowel neutral"
for item in listdir(audios):
    data, samplerate = sf.read(join(audios,item))
data = resample(data, 20000)
sf.write("Test4.wav", data,20000)
d, sa = sf.read("Test4.wav")