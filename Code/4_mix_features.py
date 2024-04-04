#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 19:35:29 2022

@author: nestor

Codigo que toma archivos de caracteristicas y genera un archivo csv con las caracteristicas encontradas,
permite mezclar caracteristicas y se almacenan en un csv con las caracteristicas por cada tipo de señal
"""
import os
import sys
import pathlib
if './Code' not in sys.path:
    sys.path.append('./Code')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from Code.feature_MFCC import *
from Code.feature_BFCC_GFCC import *
from Code.feature_non_linear import *

#%%

data_path = './Database'
feature_path = './Features'
#IDs_path = './Metadata/IDs_Balanced.csv'
tasks = 'phrase'
signal_type = ['speech','egg']
feature='all'

#def extract_features(data_path, feature_path, IDs_path, tasks, signal_type, feature='all'):
    
if feature == 'all':
    feature_list = ['BFCC', 'MFCC', 'phonation', 'GFCC', 'non_linear']
else:
    if isinstance(feature, str):
        feature_list = [feature]
    else:
        feature_list = feature
if isinstance(signal_type, str):
    signal_type = [signal_type]
    
    
if isinstance(tasks, str):
    tasks = [tasks]
    
#IDs = pd.read_csv(IDs_path, index_col=[0])
#ID_list = list(IDs['ID'].astype(int))

feature_path = pathlib.Path(feature_path)
feature_path.mkdir(parents=True, exist_ok=True)
features_found = os.listdir(feature_path)

for feature in feature_list:
    for signal in signal_type:
        for task in tasks:
            path_to_store = f'{feature}_{task}_{signal}.csv'
            path_to_store = path_to_store.replace(' ', '_')
            path_to_store = os.path.join(feature_path, path_to_store)
            path_to_store = os.path.abspath(path_to_store)
            
            path_audio = f'{signal}/{task}'
            path_audio = os.path.join(data_path, path_audio)
            print(f'Creating {feature} file for the {task} and the {signal} signal')
            #if feature == 'MFCC':
                #MFCC_feature_generate(path_audio, path_to_store)
                #print(f'Creating {feature} file for the {task} and the {signal} signal')
                #print(path_to_store)
            if feature == 'BFCC':
                BFCC_feature_generate(path_audio, path_to_store)
            if feature == 'GFCC':
                GFCC_feature_generate(path_audio, path_to_store)
            if feature == 'non_linear':
                non_linear_feature_generate(path_audio, path_to_store)

        
# =============================================================================
#     df.rename(columns={'id':'file_name'}, inplace = True)
#     df['ID'] = df['ID'].str.split('-',expand = True)[0] 
#     file_name = '/home/nestor/Documents/Maestria/Avances Maestria/Features Disvoice/Features/phonation_features_'+signal_type+'.csv'
#     df4.to_csv(file_name)
# =============================================================================
#%%

# Se descomenta si se quiere almacenar ya sea las de phonacion, noise, o mezclar
# las cepstral

#df4 = pd.merge(pd.merge(df_mfcc,df_bfcc,on='file_name'),df_gfcc,on='file_name')
df4 = df_phonation
#df4 = df_nonlinear
df4.rename(columns={'file_name':'ID'}, inplace = True)
df4['ID'] = df4['ID'].str.split('-',expand = True)[0] 

