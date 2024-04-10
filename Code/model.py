"""
Created on Thu Apr  4 19:51:12 2024

@author: nestor

#5 in excecution order
"""


import os
import pandas as pd
import numpy as np
import sys
if './Code' not in sys.path:
    sys.path.append('./Code')


data_path = './Database'
feature_path = './Features'
IDs_path = './Metadata/IDs_Balanced.csv'
task = 'vowel_a'
signal_type = 'speech'
feature='non_linear'

#IDs = pd.read_csv(IDs_path, index_col=[0])
# Read audios
def get_features_X_Y(feature_path, feature, task, signal_type, IDs_path):

    IDs_csv = pd.read_csv(IDs_path)

    IDs = [int(x) for x in list(IDs_csv['ID'])]
    
    try:
      feature_name = f'{feature}_{task}_{signal_type}.csv'
    
      feature_file = os.path.join(feature_path, feature_name)
      feature = pd.read_csv(feature_file, index_col=[0])
    except:
      print(f'Feature {feature} not found')
      return None, None
    
    X = []
    y = []
    IDs_array = []
    for i, ID in enumerate(IDs):
        if 'vowel' in task:
          task = task.replace('vowel_','')
          # IF THE AUDIO IS NOT NORMAL PITCH (CHANGE THIS)
          task = task + '_n'
        signal = '' if signal_type == 'speech' else '-egg'
        try:
            ID_selected_features = list(feature.loc[f'{ID}-{task}{signal}.wav'])
        except:
            print(f'Audio {ID}-{task}{signal}.wav not found')
    
    
        X.append(ID_selected_features)
        y.append(IDs_csv[IDs_csv['ID']==ID].target.item())
        IDs_array.append(ID)
    X = np.array(X)
    y = np.array(y)
    return X, y, IDs_array