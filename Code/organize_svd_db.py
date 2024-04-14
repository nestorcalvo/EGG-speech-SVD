#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 12:40:44 2022

@author: nestor

Script que recibe información obtenida del web scrapping, la carpeta de los audios sin organizar (consiste en descomprimir 
los archivos descargados y dejarlos en sus carpetas, el script realiza la respectiva organización) y la carpeta donde 
se almacenarán los audios finales organizados.

El script tambien verifica si un audio que esta dentro del csv con información se encuentra descargado, en caso de que no
simplemente se ignora, el script organiza cada audio dentro de la carpeta con el nombre de la enfermedad

Una vez organizado se crea un nuevo csv que contiene la información de los audios que si fueron descargados y organizados
llamado "SVD_all_pathologies.csv" este nuevo csv tiene información como edad, genero, enfermedad, frecuencia de muestreo e ID 

#2 in execution order
"""
import pandas as pd
import os
import numpy as np
import os.path
from os import listdir
from os.path import isfile, join
import shutil
import seaborn as sns
from os.path import exists
import pathlib
import itertools

#%% Store audio files in folder with the name of the pathology
folder_with_csv = "./Web_scraping_csv"
folder_with_audios = "./svd_no_organize"
folder_to_save_audios = "./SVD"

def store_files_in_folder_name(folder_with_csv, folder_with_audios,folder_to_save_audios):
    """
    Copy fiiles from the folders where the DB is unorganized and check in the .csv files for each pathology
    to check where the file belongs
    
    Parameters:
        folder_with_csv: Path to folder that contains the .csv of each patholgy
        folder_with_audios: Path to the folder with the audios after being extracted from the downloaded .zip files
        folder_to_save_audios: Path to save the organized database, inside this path a folder for each pathology will
                                be created and the audios will be stored there.
    
    """
    onlyfiles = [f for f in listdir(folder_with_csv) if isfile(join(folder_with_csv, f))]
    for file in onlyfiles:
        if (file!='geckodriver.log'):
            path = os.path.join(folder_with_csv,file)
            df = pd.read_csv(path, on_bad_lines='skip')
            file_name = file.split('.')[0]
            paths = []
            files = []
            for path_, subdirs, f in os.walk(folder_with_audios):
                paths.append(path_)
                files.append(f)
            ids = list(df['ID'])
            final_files = []
            final_path = []
            
            for ID in ids:
                for index,i in enumerate(files):
                    for f in i:
                        try:
                            if os.path.isfile(os.path.join(paths[index],f)) and str(ID) == f.split('-')[0]:
                                final_path.append(paths[index])
                                final_files.append(f)
                        except:
                            continue
            new_path = os.path.join(folder_to_save_audios,file_name)
            
            if not os.path.exists(os.path.join(folder_to_save_audios,file_name)):
                os.makedirs(os.path.join(folder_to_save_audios,file_name))
            for index,item in enumerate(final_path):
                if exists(os.path.join(new_path,final_files[index])):
                    print('File already exist')
                    continue
                else:
                    print('File doesnt exist in folder, copying...')
                    shutil.copy(os.path.join(item,final_files[index]),new_path)
    
#%% Check audio files and store 1 if is donwloaded
folder_with_csv = "./Web_scraping_csv"
folder_to_check_audios = "./SVD"
folder_to_save_csv = "./CSV_organized"
def check_file_stored(folder_with_csv, folder_to_check_audios,folder_to_save_csv):
    """
    Creates new .csv with the information of the files downloaded, sometimes audios aren't downloaded or they are missing
    so this function check what files we have and create new .csv for each pathology
    
    Parameters:
        folder_with_csv: Path to folder that contains the .csv of each patholgy
        folder_to_check_audios: Path to the folder with the audios organized with store_files_in_folder_name()
        folder_to_save_csv: Path to save the organized updated .csv
    
    """
    onlyfiles = [f for f in listdir(folder_with_csv) if isfile(join(folder_with_csv, f))]
    columns_to_add =['a wav','i wav','u wav','phrase wav','a egg','i egg','u egg','phrase egg']
    for file in onlyfiles:
        
        if (file!='geckodriver.log'):
            path = os.path.join(folder_with_csv,file)
            
            df = pd.read_csv(path, error_bad_lines='skip')
            df_new = pd.concat([df,pd.DataFrame(columns=columns_to_add)])
            
            file_name = file.split('.')[0]
            
            path_pathology = os.path.join(folder_to_check_audios, file_name)
            
            for file_audio in os.listdir(path_pathology):
                
                text_splitted = file_audio.split('-')
                
                id_file = text_splitted[0]
                audio_letter = text_splitted[1].split('_')[0]
                #audio_letter = audio_letter.split('.')[0]
                
                #EGG file
                if(len(text_splitted)==3):
                    file_extension = text_splitted[2].split('.')[1]
                    if (file_extension=='wav'):
                        column_to_save = audio_letter + " egg"
                else:
                    audio_letter = audio_letter.split('.')[0]
                    
                    column_to_save = audio_letter + " wav"
                print(column_to_save)
                df_new.loc[df_new["ID"] == int(id_file),column_to_save] = 1
                df_new.to_csv(os.path.join(folder_to_save_csv, file), index=False)
                #print(audio_letter)
            
                
check_file_stored(folder_with_csv, folder_to_check_audios,folder_to_save_csv)
#%% Concatenate all the csv into one csv
folder_with_csv = "./CSV_organized"
path_to_save = "./Metadata/SVD_all_pathologies.csv"

def concatenate_all_csv(folder_with_csv, path_to_save):
    """
    Concatenate all the .csv organized into one final .csv
    
    Parameters:
        folder_with_csv: Path to folder that contains the .csv of each patholgy
        path_to_save: Path to save the final .csv
    """
    onlyfiles = [f for f in listdir(folder_with_csv) if isfile(join(folder_with_csv, f))]
    array_of_dataframes = []
    for file in onlyfiles:
        
        path = os.path.join(folder_with_csv,file)
        df_temp = pd.read_csv(path, error_bad_lines='skip')
        if (df_temp.shape[1] == 18):
            df_temp = df_temp.drop(columns=('remarks wav'))
        array_of_dataframes.append(df_temp)
    
    df_final = pd.DataFrame(np.concatenate([x.values for x in array_of_dataframes]), columns=array_of_dataframes[1].columns)
    
    df_final = df_final.drop(columns=("Remarks w.r.t. diagnosis"))
    df_final["Pathologies"][pd.isna(df_final["Pathologies"])] = '-'
    df_final = df_final.dropna(thresh=7)
    df_final = df_final.drop(columns=("B"))
    df_final[['Pathologies', 'B']] = df_final['Pathologies'].str.split(',', n=1, expand=True)
    df_final["label"] = df_final["Pathologies"]
    df_final["label"][df_final["Pathologies"]!="-"] = "P"
    df_final["label"][df_final["Pathologies"]=="-"] = "H"
    df_final = df_final.drop(columns=("B"))
    df_final.to_csv(path_to_save, index=False)
    
concatenate_all_csv(folder_with_csv, path_to_save)

#%%
audio_folder = './SVD'
general_path = './Database'
def copy_files_folder(audio_folder, general_path):
    '''
    Copy all the audios that are in each pathology folder and copies it to a folder
    where it has the task and the signal as name
    
    Parameters:
        audio_folder: Path with audios inside pathology folder
        general_path: Path to store the audios (the folders for the task and the signal are created)
    '''
    signal_type = ['egg', 'speech']
    tasks = ['vowel a', 'vowel i', 'vowel u', 'phrase']
    # Create folders per signal per task if folder is missing
    for signal, task in itertools.product(signal_type, tasks):
        path_name = f'{general_path}/{signal}/{task}'
        path = pathlib.Path(path_name)
        path.mkdir(parents=True, exist_ok=True)
        
    folder_names = [name for name in os.listdir(audio_folder) if os.path.isdir(os.path.join(audio_folder, name))]
    for folder_name in folder_names:
        print('Copying files to folders...')
        path_audio = os.path.join(audio_folder, folder_name)
        path_audio = pathlib.Path(path_audio)
        onlyfiles = [f for f in listdir(path_audio) if isfile(join(path_audio, f))]
        for file in onlyfiles:
            task_name = ''
            signal_name = ''
            # Drop all files that are not .wav
            if '.wav' not in file:
                continue
            # Check which task and signal type is (they need to be the ones in the variables
            # task and signal_type above)
            if 'a_n' in file:
                task_name = tasks[0]
            elif 'i_n' in file:
                task_name = tasks[1]
            elif 'u_n' in file:
                task_name = tasks[2]
            elif 'phrase' in file:
                task_name = tasks[3]
            else:
                print(f'File {file} does not contain a proper task name')
            # Check the signals
            if '-egg.' in file:
                signal_name = signal_type[0]
            else:
                signal_name = signal_type[1]
            
            
            store_path = f'{general_path}/{signal_name}/{task_name}/{file}'
            shutil.copyfile(os.path.join(path_audio, file), store_path)
    print('Files copied')     
#def split_audios_per_task(audio_folder, task, signal):
copy_files_folder(audio_folder, general_path)

    