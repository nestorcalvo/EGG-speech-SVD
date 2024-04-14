# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:25:21 2022

@author: ariasvts
"""

import os,sys
sys.path.append(os.path.join(sys.path[0],'Measurements'))
import pprint
from tqdm import tqdm
pprint.pprint(sys.path)
import praat.praat_functions as praat
#-
import sigproc as sg
import prosody as pr
#-
import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
from scipy import stats
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import librosa
import json

step_time = 0.01
path_files = '/home/nestor/Documents/Maestria/Avances Maestria/Databases Masters/Database Merge'
X_phon = {
    "ID":[],
    "jit":[],
    "shimm":[],
    "apq3":[],
    "apq5":[],
}

base_path, _, files_name = list(os.walk(path_files))[0]
print("antes de entrar al for")
for file in tqdm(files_name):
    
    id_number = int(file.split('-')[0])

    
    #wavfile = '/home/nestor/Documents/Maestria/Avances Maestria/Databases Masters/Database Merge WAV/1-a_n.wav'
    wavfile = os.path.join(base_path, file)
    new_sr = 44100  # new sampling rate
    
    sig, sr = librosa.load(wavfile, sr=None)
    
    sig = librosa.resample(sig, sr, new_sr)
    fs = new_sr
    #-
    sig = sig-np.mean(sig)
    sig = sig/np.max(sig)
    
    f0 = pr.f0_contour_pr(sig,fs)
    uf0 = np.mean(f0[f0>0])
    sf0 = np.std(f0[f0>0])
    #Loudness
    spl = pr.sound_pressure_level(sig,fs)
    uspl = np.mean(spl[spl>0])
    sspl = np.std(spl[spl>0])
    #Perturbation
    abs_signal = np.abs(sig)
    

    diff_signal = np.diff(abs_signal)

    jitter = str(np.std(diff_signal))
    
    jit = pr.ppq(f0,2)
    ppq3 = pr.ppq(f0,3)
    ppq5 = pr.ppq(f0,5)
    shimm = pr.apq(spl,2)
    apq3 = pr.apq(spl,3)
    apq5 = pr.apq(spl,5)       
      
    X_phon['ID'].append(id_number)

    X_phon['jit'].append(jitter)
    X_phon['shimm'].append(shimm)
    X_phon['apq3'].append(apq3)
    X_phon['apq5'].append(apq5)

df = pd.DataFrame.from_dict(X_phon)
df.to_csv('phonation_features_tomas_EGG.csv')
    
# =============================================================================
# with open("phonation_features_tomas_EGG.json", "w") as write_file:
#     json.dump(X_phon, write_file, indent=4)
# =============================================================================
