"""
Created on Thu Apr  4 19:51:12 2024

@author: nestor
"""

import os
import pandas as pd
import numpy as np
import sys
if './Code' not in sys.path:
    sys.path.append('./Code')
from Code.utils import *

data_path = './Database'
feature_path = './Features'
IDs_path = './Metadata/IDs_Balanced.csv'
task = 'phrase'
signal_type = 'speech'
signal = '' if signal_type == 'speech' else '-egg'
feature='non_linear'

#IDs = pd.read_csv(IDs_path, index_col=[0])
# Read audios
#def read_database_audio(data_path, task, signal, IDs_path):
IDs_csv = pd.read_csv(IDs_path, index_col=[0])
IDs = [int(x) for x in list(IDs_csv['ID'])]
feature_name = f'{feature}_{task}_{signal_type}.csv'
feature_file = os.path.join(feature_path, feature_name)
feature = pd.read_csv(feature_file, index_col=[0])

X = []
y = []
for i, ID in enumerate(IDs):
    
    ID_selected_features = list(feature.loc[f'{ID}-{task}{signal}.wav'])
    
        
    X.append(ID_selected_features)    
    y.append(IDs_csv[IDs_csv['ID']==ID].target.item())

    
# Fold creation 