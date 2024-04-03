#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:31:20 2023

@author: nestor
"""
#%% Librerias

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold,GridSearchCV
import random
from time import sleep
from tqdm import tqdm
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import roc_curve,auc,accuracy_score,confusion_matrix
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from scipy import interp
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from os import listdir
from os.path import isfile, join
import re
import shutil
#%%
audio_path = '/home/nestor/Documents/Maestria/Avances Maestria/Databases Masters/SVD'
IDs_path = '/home/nestor/Documents/Maestria/Avances Maestria/Dataframe For Models/IDs_Balanced.csv'
HC_EGG = '/home/nestor/Documents/Maestria/Avances Maestria/Databases Masters/Phrase SVD EGG files/HC'
P_EGG = '/home/nestor/Documents/Maestria/Avances Maestria/Databases Masters/Phrase SVD EGG files/P'
HC_WAV = '/home/nestor/Documents/Maestria/Avances Maestria/Databases Masters/Phrase SVD WAV files/HC'
P_WAV = '/home/nestor/Documents/Maestria/Avances Maestria/Databases Masters/Phrase SVD WAV files/P'
ids_df = pd.read_csv(IDs_path, index_col=0)
class_type = "P"
signal_type = "WAV"

if class_type == "HC":
    target = 0
else:
    target = 1

if signal_type == "EGG":
    regex = ".*phrase-egg.wav"
    new_path = HC_EGG if target == 0 else P_EGG
else:

    regex = ".*phrase.wav"
    new_path = HC_WAV if target == 0 else P_WAV
    
IDs_to_copy = list(ids_df[ids_df.target == target]['ID'])

folders = list(os.walk(audio_path))

for dirpath, subdirs, files in os.walk(audio_path):
    if subdirs!=[]:
        continue
    r = re.compile(regex)
    
    files_first_filter = [f for f in files if int(f.split('-')[0]) in IDs_to_copy]
    
    if files_first_filter == []:
        continue
    file_list = [os.path.join(dirpath, file) for file in files_first_filter]
    newlist = list(filter(r.match, file_list)) # Read Note below
    for index,item in enumerate(newlist):
        file_name = item.split('/')[-1]
        if os.path.exists(os.path.join(new_path,file_name)):
            print('File already exist')
            continue
        else:
            print('File doesnt exist in folder, copying...')
            shutil.copy(item,new_path)
