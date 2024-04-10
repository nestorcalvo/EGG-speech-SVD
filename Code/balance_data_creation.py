#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 19:15:18 2022

@author: nestor

Codigo utilizado para balancer las bases de datos entre genero 
y edad, utiliza el archivo de caracteristicas completas y extrae los IDs y caracteristicas
de allí que cumplan un balance en edad y genero.

Para el caso de speech se usan los IDs extraidos en EGG, dado que speech tiene mas
samples

Es importante mencionar que con este codigo se generan los IDs balanceados, pero las lineas de comando se comentan
para utilizar siempre los mismos IDs en todo momento los cuales se encuentran en un archivo csv

#3 in execution order
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import python_speech_features as psf
from scipy.stats import chi2_contingency
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from scipy.stats import chisquare
from statsmodels.stats import weightstats as stests

#%% Read metadata files
#info_subjects = '/home/nestor/Documents/Maestria/Avances Maestria/Databases Masters/Database_info.ods'
info_subjects = './Metadata/SVD_all_pathologies.csv'
save_path = './CSV_organized'
path_balanced_IDs = './Metadata/IDs_Balanced.csv'
def balance_dataset(info_subjects,save_path,path_balanced_IDs):
    """
    Generates a .csv file with the IDs of the subjects that generates a balanced database, this searchs for the best
    combination of users that fulfill the condition of removing imbalances in gender and age
    
    Parameters:
        info_subjects: Path to .csv with all the metadata or user data
        save_path: Path that has all the files of a single task in one folder
        path_balanced_IDs: Path to store the balanced IDs
    
    """
    metadata_SVD = pd.read_csv(info_subjects)
    metadata_SVD = metadata_SVD.dropna()
    metadata_SVD = metadata_SVD.drop(columns=(['i wav','u wav','phrase wav','i egg','a egg','u egg','phrase egg']))
    #metadata_SVD = metadata_SVD[metadata_SVD['a wav'].notna()]
    #metadata_SVD = metadata_SVD.drop_duplicates(subset = ['S'])
    mask = ['Laryngitis','Dysphonie', 'Funktionelle Dysphonie','Dysodie','Kontaktpachydermie', 'Balbuties','Intubationsgranulom', 'Gesangsstimme','Leukoplakie']
    metadata_SVD = metadata_SVD[~metadata_SVD['Pathologies'].isin(mask)]
    list_files = listdir(save_path)
    id_in_files = []
    
    for file in list_files:
        file_path = os.path.join(save_path, file)
        df = pd.read_csv(file_path, on_bad_lines='skip')
    
        id_in_files += list(df['ID'].map(lambda x:int(x)))
    
    
    metadata_SVD_no_duplicates = metadata_SVD[metadata_SVD['ID'].isin(id_in_files)].drop_duplicates(subset = ["ID"], keep = 'first')
    metadata_SVD_no_duplicates["ID"] = metadata_SVD_no_duplicates["ID"].astype(str)
    PD = metadata_SVD_no_duplicates[metadata_SVD_no_duplicates['T']=='p'].copy()
    HC = metadata_SVD_no_duplicates[metadata_SVD_no_duplicates['T']=='n'].copy()
    print(PD.shape)
    print(HC.shape)
    alpha = 0.05
    flag_thresholdAge = False
    Gender = ['w', 'm']
    
    HC = HC[HC['A']>22]
    print(stats.ttest_ind(PD['A'], HC['A']))
    for threshold in reversed(range(0,10)):
        Ids_toADD = {}
        if not(flag_thresholdAge):
            threshold_desv_age = threshold
            for i, gender in enumerate(Gender):
    
                dataGenderHC = HC[HC['G'] == gender]
                dataGenderPD = PD[PD['G'] == gender]
                for i,age in enumerate(dataGenderHC['A']):
                    ageReferenceHC = int(age)
                    dataToCheck = dataGenderPD[['ID','G','A']]
                    ID_HC = str(dataGenderHC['ID'].iloc[i])
                    for id_GITA, gender, agePD in dataToCheck.iloc:
                        distanceAge = abs(ageReferenceHC-int(agePD))
                        if distanceAge <= threshold_desv_age:
                            #id_new = id_GITA+'_'+session
                            if not(id_GITA in Ids_toADD.keys()):
                                Ids_toADD[id_GITA] = [agePD, gender]
                                break
            dataToAdd = pd.DataFrame.from_dict(Ids_toADD, orient='index',columns=['A', 'G'])
            t_testAge = stats.ttest_ind(dataToAdd['A'], HC['A'])
    
            if t_testAge[1] >= alpha:
                flag_thresholdAge = True
                break
    
    mask = PD['ID'].isin(list(dataToAdd.index))
    PD = PD.loc[mask]
    
    print(PD.shape)
    #PD['ID'] = PD['ID'].str.split('.',expand = True)[0]
    #HC['ID'] = HC['ID'].str.split('.',expand = True)[0]
    print(HC.shape)
    metadata_SVD_no_duplicates['ID'] = metadata_SVD_no_duplicates['ID'].astype(str)
    print(metadata_SVD_no_duplicates)
    mask_PD = metadata_SVD_no_duplicates['ID'].isin(list(PD['ID']))
    mask_HC = metadata_SVD_no_duplicates['ID'].isin(list(HC['ID']))
    
    PD_balanced = metadata_SVD_no_duplicates.loc[mask_PD]
    HC_balanced = metadata_SVD_no_duplicates.loc[mask_HC]
    print(PD_balanced.shape)
    print(HC_balanced.shape)
    PD_balanced['target']=1
    HC_balanced['target']=0
    df_final = pd.concat([PD_balanced, HC_balanced])
    
    dict_gender = {'PD':{'m':PD.groupby('G').count()['ID']['m'],'w':PD.groupby('G').count()['ID']['w']},
                   'HC':{'m':HC.groupby('G').count()['ID']['m'],'w':HC.groupby('G').count()['ID']['w']}}
    print(dict_gender)
    df_gender = pd.DataFrame.from_dict(dict_gender)
    print(df_gender)
    t_testAge = stats.ttest_ind(PD['A'], HC['A'])
    if t_testAge[1] >= alpha:
        print("Dataset valid in terms of age, p_value: ",t_testAge[1])
    else:
        print("Dataset invalid in terms of age, p_value: ",t_testAge[1])
    
    c, p, dof, expected = chi2_contingency(df_gender)
    if p >= alpha:
        print("Dataset valid in terms of gender, p_value: ",p)
    else:
        print("Dataset invalid in terms of gender, p_value: ",p)
    
    # Estas lineas son para guardar los IDs encontrados en EGG para usarlos con speech
    IDs_df_balanced = df_final[["ID","target", "G", "A", "Pathologies"]]
    IDs_df_balanced.to_csv(path_balanced_IDs)
    return IDs_df_balanced
#%%
IDs_df_balanced = balance_dataset(info_subjects,save_path,path_balanced_IDs)
#%%
#'/home/nestor/Documentos/Maestria/Metadata/IDs_Balanced.csv'
csv_readed = pd.read_csv('/home/nestor/Documentos/Maestria/Metadata/IDs_Balanced.csv')
PD = csv_readed[csv_readed['target']==1].copy()
HC = csv_readed[csv_readed['target']==0].copy()
t_testAge = stats.ttest_ind(PD['A'], HC['A'])
print(t_testAge)