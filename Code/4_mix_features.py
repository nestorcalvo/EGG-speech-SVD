#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 19:35:29 2022

@author: nestor

Codigo que toma archivos de caracteristicas y genera un archivo csv con las caracteristicas encontradas,
permite mezclar caracteristicas y se almacenan en un csv con las caracteristicas por cada tipo de señal
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir

#%%

audio_path = '/home/nestor/Documents/Maestria/Avances Maestria/Databases Masters/Database Merge WAV'
save_path = '/home/nestor/Documents/Maestria/Avances Maestria/Features Disvoice/Features/MFCC_a_vowel_speech.csv'
feature_path = '/home/nestor/Documents/Maestria/Avances Maestria/Features Disvoice/Features'
signal_type = 'speech'
def extract_features(feature_path, store_path):
    feature_list = ['BFCC', 'MFCC', 'phonation', 'GFCC', 'non_linear']
    signal_type = ['egg', 'speech']
    tasks = ['a_vowel']
    features_found = os.listdir(feature_path)
    for feature in feature_list:
        for signal in signal_type:
            for task in tasks:
                
# =============================================================================
#     for feature in features:
#         for signal in signal_type:
#             for task in tasks:
#         if "MFCC" in feature and signal_type in feature:
#             df = pd.read_csv(feature)
#         elif "ROC" in feature:
#             continue
#         elif "BFCC" in feature and signal_type in feature:
#             df = pd.read_csv(feature)
#         elif "GFCC" in feature and signal_type in feature:
#             df = pd.read_csv(feature)
#         elif "phonation" in feature and signal_type in feature:
#             df = pd.read_csv(feature)
#             
#         elif 'non_linear' in feature and signal_type in feature:
#             df = pd.read_csv(feature)
#         else:
#             continue
# =============================================================================
        
    df.rename(columns={'id':'file_name'}, inplace = True)
    df['ID'] = df['ID'].str.split('-',expand = True)[0] 
    file_name = '/home/nestor/Documents/Maestria/Avances Maestria/Features Disvoice/Features/phonation_features_'+signal_type+'.csv'
    df4.to_csv(file_name)
#%%

# Se descomenta si se quiere almacenar ya sea las de phonacion, noise, o mezclar
# las cepstral

#df4 = pd.merge(pd.merge(df_mfcc,df_bfcc,on='file_name'),df_gfcc,on='file_name')
df4 = df_phonation
#df4 = df_nonlinear
df4.rename(columns={'file_name':'ID'}, inplace = True)
df4['ID'] = df4['ID'].str.split('-',expand = True)[0] 

