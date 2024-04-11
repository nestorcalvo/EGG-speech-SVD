#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 09:00:53 2024

@author: nestor

#7 in execution order
"""
import sys

if './Code' not in sys.path:
    sys.path.append('./Code')

from utils import *
from model import *
import ast

data_path = './Database'
feature_path = './Features'
results_path = './Results'
folds_path = './Folds'
#IDs_path = './Metadata/all_IDs.csv'
IDs_path = './Metadata/IDs_Balanced.csv'
tasks = ['vowel_a','vowel_i','vowel_u','phrase'] #'phrase' 'vowel_a'
models = ['SVM','DT','RF'] #'DT' or 'RF' or 'XGBoost' or ['DT', 'RF']
signal_type =  ['speech']# or just 'egg' or 'speech'
feature=['BFCC', 'GFCC', 'non_linear'] #'BFCC' or 'GFCC' or 'non_linear' or 'all'

def models_optimization(folds_path, IDs_path, results_path, tasks, models, signal_type, feature):
    if feature == 'all':
        feature_list = [ 'BFCC', 'GFCC', 'non_linear']
    else:
        if isinstance(feature, str):
            feature_list = [feature]
        else:
            feature_list = feature
    if isinstance(signal_type, str):
        signal_type = [signal_type]
    
    if isinstance(models, str):
        models = [models]
    
    if isinstance(tasks, str):
        tasks = [tasks]
    
    results_path = pathlib.Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    folds_name = os.path.join(folds_path,'folds.pickle')
    with open(folds_name, 'rb') as output:
        cv_folds = pickle.load(output)
   
    
    for model in models:
      for feature in feature_list:
        for task in tasks:
          for signal in signal_type:
    
            print(' Model:',model,' Feature:',feature,' Task:',task,' Signal:',signal)
            X,y,ID_array = get_features_X_Y(feature_path, feature, task, signal, IDs_path)
            X = pd.DataFrame(X,index=ID_array)

            y = pd.DataFrame(y,index=ID_array)
            save_path_name_folds = f'{results_path}/folds_{model}_{feature}_{task}_{signal}.pickle'
            save_path_name_fixed = f'{results_path}/fixed_{model}_{feature}_{task}_{signal}.pickle'
            if model == 'SVM':
              print('SVM model')
              SVM_Optimization(X,y,cv_folds,save_path_name_folds,save_path_name_fixed)
            elif model == 'DT':
              print('DT model')
              DT_Optimization(X,y,cv_folds,save_path_name_folds,save_path_name_fixed)
            elif model == 'RF':
              print('RF model')
              RF_Optimization(X,y,cv_folds,save_path_name_folds,save_path_name_fixed)
            elif model == 'XGBoost':
              print('XGBoost model')
            else:
              print('Model not supported')
              break
#%%
models_optimization(folds_path, IDs_path, results_path, tasks, models, signal_type, feature)
#%%


if feature == 'all':
    feature_list = [ 'BFCC', 'GFCC', 'non_linear']
else:
    if isinstance(feature, str):
        feature_list = [feature]
    else:
        feature_list = feature
if isinstance(signal_type, str):
    signal_type = [signal_type]

if isinstance(models, str):
    models = [models]

if isinstance(tasks, str):
    tasks = [tasks]

folds_name = os.path.join(folds_path,'folds.pickle')
with open(folds_name, 'rb') as output:
    cv_folds = pickle.load(output)

for model in models:
  for feature in feature_list:
    for task in tasks:
      for signal in signal_type:
        file_name = f'{results_path}/fixed_{model}_{feature}_{task}_{signal}.pickle'
        print(' Model:',model,' Feature:',feature,' Task:',task,' Signal:',signal)
        with open(file_name, 'rb') as output:
            results = pickle.load(output)
        accuracy_toal = []
        f1_score_total = []

        for key in results.keys():
          accuracy_toal.append(results[key]["accuracy_best_model"])
          f1_score_total.append(results[key]["f1_score"])

        mean_accuracy = np.mean(accuracy_toal)
        mean_f1_score = np.mean(f1_score_total)

        std_accuracy = np.std(accuracy_toal)
        std_f1_score = np.std(f1_score_total)

        print(mean_accuracy, "+-", std_accuracy)
        print(mean_f1_score, "+-", std_f1_score)
        #print(results[key]["param_used"])
