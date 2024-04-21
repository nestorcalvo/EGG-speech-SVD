#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 23:08:04 2024

@author: nestor
"""

import sys
import os
from os import listdir
from os.path import isfile, join

disvoice_path = '/home/nestor/Documents/Maestria/DisVoice/disvoice/'
path_store = '/home/nestor/Documents/Maestria/Masters Degree Thesis/Features/Features_disvoice'
if disvoice_path not in sys.path:
    sys.path.append(disvoice_path)
from articulation.articulation import Articulation
from phonation import Phonation
signal_type = ['speech', 'egg']
tasks = ['vowel u', 'phrase']
for task in tasks:
    for signal in signal_type:
        
        if ' ' in task:
            task_name_file = task.replace(' ','_')
        else:
            task_name_file = task
        path_folder = f'/home/nestor/Documents/Maestria/Masters Degree Thesis/Database/{signal}/{task}/'
        articulation=Articulation()
        phonation=Phonation()
        
        features1=phonation.extract_features_path(path_folder, static=True, plots=False, fmt="csv")
        features1 = features1.rename({'id':'file_name'})
        filename = f'phonation_{task_name_file}_{signal}.csv'
        filename_path = os.path.join(path_store,filename)
        features1.to_csv(filename_path, sep=',', index=False, encoding='utf-8')
        
        features2 = phonation.extract_features_path(path_folder, static=True, plots=False, fmt="csv")
        features2 = features2.rename({'id':'file_name'})
        filename = f'articulation_{task_name_file}_{signal}.csv'
        filename_path = os.path.join(path_store,filename)
        features2.to_csv(filename_path, sep=',', index=False, encoding='utf-8')
        


#%%


