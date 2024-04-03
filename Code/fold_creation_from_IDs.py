#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:20:27 2023

@author: nestor
"""
"""
Librerias
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
#%% Load IDs to create folds
ids_balanced = "/home/nestor/Documents/Maestria/Avances Maestria/Dataframe For Models/IDs_Balanced.csv"
df = pd.read_csv(ids_balanced)

skf = StratifiedKFold(n_splits=10,shuffle = True, random_state=26)

target = df.loc[:,'target']

# for each fold split the data into train and validation 
# sets and save the fold splits to csv
fold_no = 1
for train_index, test_index in skf.split(df, target):
    train = df.loc[train_index,['ID','target']]
    test = df.loc[test_index,['ID','target']]
    train.to_csv('/home/nestor/Documents/Maestria/Avances Maestria/Dataframe For Models/Folds/' + 'train_fold_' + str(fold_no) + '.csv')
    test.to_csv('/home/nestor/Documents/Maestria/Avances Maestria/Dataframe For Models/Folds/' + 'test_fold_' + str(fold_no) + '.csv')
    fold_no += 1
