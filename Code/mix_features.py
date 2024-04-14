#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 19:35:29 2022

@author: nestor

Codigo que toma archivos de caracteristicas y genera un archivo csv con las caracteristicas encontradas,
permite mezclar caracteristicas y se almacenan en un csv con las caracteristicas por cada tipo de señal

#4 in execution order
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
tasks = ['vowel a','vowel i', 'vowel u', 'phrase']#'phrase'
signal_type = ['egg']
feature='non_linear'

def extract_features(data_path, feature_path, tasks, signal_type, feature='all'):

  if feature == 'all':
      feature_list = ['MFCC','phonation', 'BFCC', 'GFCC', 'non_linear']
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
              if feature == 'MFCC':
                  MFCC_feature_generate(path_audio, path_to_store)
                  #print(f'Creating {feature} file for the {task} and the {signal} signal')
                  #print(path_to_store)
              if feature == 'BFCC':
                  BFCC_feature_generate(path_audio, path_to_store)
              if feature == 'GFCC':
                  GFCC_feature_generate(path_audio, path_to_store)
              if feature == 'non_linear':
                  non_linear_feature_generate(path_audio, path_to_store)

extract_features(data_path, feature_path, tasks, signal_type, feature)
