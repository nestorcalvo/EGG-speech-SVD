#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:05:29 2022

@author: nestor

Archivo utilizad para acceder automaticamente a la base de datos SVD y realizar la descarga de los
archivos, este script permite hacer webscarping para no solo descargar sino almacenar la información en un .csv
de los datos descargados
"""
#%% Librerias a importar
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import os.path

from selenium.webdriver.support import expected_conditions as EC
#%%

def read_ID_missing(file_path, audio):
    df = pd.read_csv(file_path)
    id_missing = df['ID'][pd.isna(df[audio])]
    id_missing = list(id_missing.astype(int))
    return id_missing


#%%
def get_pages(pathology_name,pages):
    #Get pages 
    
    select_pages = Select(pages)
    list_pages = select_pages.options

    results_array= []
    for i, page in enumerate(list_pages):
        # Check page selector after page update
        pages = driver.find_element(By.XPATH, value = '/html/body/form/div[6]/table/tbody/tr[2]/td[2]/span[2]/select')
        select_pages = Select(pages)
        # Select a page
        select_pages.select_by_visible_text(str(i+1))
        driver.implicitly_wait(2)
        # Extract values from table in this page
        table = driver.find_element(By.XPATH, value = '/html/body/form/div[6]/table/tbody')
        rows = table.find_elements(By.TAG_NAME,'tr')
        #Remove title row, use it only once
        if i ==0 :
            lower = 2
        else:
            lower = 3
        # For each row, append the data to an array
        for index_row, row in enumerate(rows):

            if (index_row>=lower and index_row<=len(rows)-3):
                cells = row.find_elements(By.TAG_NAME,'td')
                temp_array = []
                for index_cell, cell in enumerate(cells): 
                     temp_array.append(cell.text)
                results_array.append(temp_array)
            
            
        driver.implicitly_wait(1)
    df = pd.DataFrame(results_array[1:], columns = results_array[0])
    df = df.drop(columns = ['E'])
    df.to_csv(pathology_name+'.csv', index=False)
   
    
#%%
driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()))

# Find the DB page
driver.get("http://stimmdb.coli.uni-saarland.de/index.php4#target")
driver.implicitly_wait(1)
# Change lenguage to english
english_button = driver.find_element(by=By.NAME, value="sb_lang")
english_button.click()
driver.implicitly_wait(1)
# Search for button to entry the database
search_button = driver.find_element(by=By.NAME, value="sb_search")

assert search_button.get_attribute('value') == "Database request", "Database expected to be in english"

# Click the button and wait for the page to load
search_button.click()
driver.implicitly_wait(1)

#Select healthy 
HC_checkbox = driver.find_element(by=By.ID, value="n")
HC_checkbox.click()
#Check descending order speaker number
speaker_number_order = driver.find_element(by=By.ID, value="sort1inv")
speaker_number_order.click()

#Accept button 
accept_button = driver.find_element(by=By.NAME, value="sb_sent")
accept_button.click()

driver.implicitly_wait(1)
#Get pages 
pages = driver.find_element(By.XPATH, value = '/html/body/form/div[6]/table/tbody/tr[2]/td[2]/span[2]/select')
select_pages = Select(pages)
list_pages = select_pages.options

results_array= []
for i, page in enumerate(list_pages):
    # Check page selector after page update
    pages = driver.find_element(By.XPATH, value = '/html/body/form/div[6]/table/tbody/tr[2]/td[2]/span[2]/select')
    select_pages = Select(pages)
    # Select a page
    select_pages.select_by_visible_text(str(i+1))
    driver.implicitly_wait(2)
    # Extract values from table in this page
    table = driver.find_element(By.XPATH, value = '/html/body/form/div[6]/table/tbody')
    rows = table.find_elements(By.TAG_NAME,'tr')
    #Remove title row, use it only once
    if i ==0 :
        lower = 2
    else:
        lower = 3
    # For each row, append the data to an array
    for index_row, row in enumerate(rows):

        if (index_row>=lower and index_row<=len(rows)-3):
            cells = row.find_elements(By.TAG_NAME,'td')
            temp_array = []
            for index_cell, cell in enumerate(cells): 
                 temp_array.append(cell.text)
            results_array.append(temp_array)
        
        
    driver.implicitly_wait(1)
driver.quit()

#%%

df = pd.DataFrame(results_array[1:], columns = results_array[0])
df = df.drop(columns = ['E'])
df.to_csv('HC_metadata.csv', index=False)


#%%
driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()))

# Find the DB page
driver.get("http://stimmdb.coli.uni-saarland.de/index.php4#target")
driver.implicitly_wait(1)

# Change lenguage to english
english_button = driver.find_element(by=By.NAME, value="sb_lang")
english_button.click()
driver.implicitly_wait(1)
# Search for button to entry the database
search_button = driver.find_element(by=By.NAME, value="sb_search")

assert search_button.get_attribute('value') == "Database request", "Database expected to be in english"

# Click the button and wait for the page to load
search_button.click()
driver.implicitly_wait(1)

#Check descending order speaker number
speaker_number_order = driver.find_element(by=By.ID, value="sort1inv")
speaker_number_order.click()


select_element = driver.find_element(By.ID,'s_pat_1')
select_pathology = Select(select_element)
list_pathology = select_pathology.options
for index, pathology in enumerate(list_pathology):
    select_element = driver.find_element(By.ID,'s_pat_1')
    select_pathology = Select(select_element)
    list_pathology = select_pathology.options
    select_pathology.select_by_visible_text(list_pathology[index].text)
    
    pathology_name = list_pathology[index].text
    if os.path.isfile(pathology_name+'.csv'):
        print ("File "+pathology_name+'.csv'+ " exist")
        driver.implicitly_wait(2)
        
        reset_button = driver.find_element(by = By.XPATH, value = "/html/body/form/div[4]/span[4]/input")
        reset_button.click()
        driver.implicitly_wait(10)
        
        speaker_number_order = driver.find_element(by=By.ID, value="sort1inv")
        speaker_number_order.click()
    else:
        print ("File "+pathology_name+'.csv'+ " not exist")

    
        add_button = driver.find_element(By.XPATH,"/html/body/form/div[3]/table/tbody/tr[3]/td[2]/table/tbody/tr[2]/td[2]/div[1]/input")
        add_button.click()
        accept_button = driver.find_element(by=By.NAME, value="sb_sent")
        accept_button.click()
        driver.implicitly_wait(1)
        
    
            
        pages = driver.find_element(By.XPATH, value = '/html/body/form/div[6]/table/tbody/tr[2]/td[2]/span[2]/select')
        #get_pages(pathology.text,pages)
        select_pages = Select(pages)
        list_pages = select_pages.options
    
        results_array= []
        for i, page in enumerate(list_pages):
            # Check page selector after page update
            pages = driver.find_element(By.XPATH, value = '/html/body/form/div[6]/table/tbody/tr[2]/td[2]/span[2]/select')
            select_pages = Select(pages)
            # Select a page
            select_pages.select_by_visible_text(str(i+1))
            driver.implicitly_wait(2)
            # Extract values from table in this page
            table = driver.find_element(By.XPATH, value = '/html/body/form/div[6]/table/tbody')
            rows = table.find_elements(By.TAG_NAME,'tr')
            #Remove title row, use it only once
            if i ==0 :
                lower = 2
            else:
                lower = 3
            # For each row, append the data to an array
            for index_row, row in enumerate(rows):
    
                if (index_row>=lower and index_row<=len(rows)-3):
                    cells = row.find_elements(By.TAG_NAME,'td')
                    temp_array = []
                    for index_cell, cell in enumerate(cells): 
                         temp_array.append(cell.text)
                    results_array.append(temp_array)
                
                
            driver.implicitly_wait(1)
        df = pd.DataFrame(results_array[1:], columns = results_array[0])
        df = df.drop(columns = ['E'])
        df.to_csv(pathology_name+'.csv', index=False)
        
        export_button = driver.find_element(by=By.XPATH,value = '/html/body/form/div[7]/span[5]/input')
        export_button.click()
        driver.implicitly_wait(1)
        i_neutral_check = driver.find_element(by=By.XPATH,value = '/html/body/form/div[3]/table/tbody/tr[1]/td[2]/table/tbody/tr[2]/td[2]/table/tbody/tr[2]/td[1]/input').click()
        a_neutral_check = driver.find_element(by=By.XPATH,value = '/html/body/form/div[3]/table/tbody/tr[1]/td[2]/table/tbody/tr[2]/td[2]/table/tbody/tr[3]/td[1]/input').click()
        u_neutral_check = driver.find_element(by=By.XPATH,value = '/html/body/form/div[3]/table/tbody/tr[1]/td[2]/table/tbody/tr[2]/td[2]/table/tbody/tr[4]/td[1]/input').click()
        sentence_check = driver.find_element(by=By.ID,value = 's_export_phrase').click()
        
        speech_wav_check = driver.find_element(by=By.ID,value = 's_exp_nspwav').click()
        egg_wav_check = driver.find_element(by=By.ID,value = 's_exp_eggegg').click()
        egg_egg_check = driver.find_element(by=By.ID,value = 's_exp_eggwav').click()
        
        accept_button = driver.find_element(by=By.XPATH,value ='/html/body/form/div[4]/span[3]/input').click()
        driver.implicitly_wait(4)
        #download_button = driver.find_element(by=By.XPATH,value ='/html/body/form/a/div[2]/table/tbody/tr/td[4]/a')
        element = WebDriverWait(driver, 3000).until(
        EC.element_to_be_clickable((By.XPATH, '/html/body/form/a/div[2]/table/tbody/tr/td[4]/a')))
        element.click()
        driver.implicitly_wait(2)
        
        back_button =driver.find_element(by=By.XPATH,value ='/html/body/form/a/div[1]/span[5]/input').click() 
        driver.implicitly_wait(2)
        
        reset_button = driver.find_element(by = By.XPATH, value = "/html/body/form/div[4]/span[4]/input")
        reset_button.click()
        driver.implicitly_wait(10)
        
        speaker_number_order = driver.find_element(by=By.ID, value="sort1inv")
        speaker_number_order.click()
driver.implicitly_wait(10)
driver.quit()
    
# =============================================================================
# #Accept button 
# accept_button = driver.find_element(by=By.NAME, value="sb_sent")

# accept_button.click()
# 
# driver.implicitly_wait(1)
# #Get pages 
# pages = driver.find_element(By.XPATH, value = '/html/body/form/div[6]/table/tbody/tr[2]/td[2]/span[2]/select')
# select_pages = Select(pages)
# list_pages = select_pages.options
# 
# results_array= []
# =============================================================================

#%% Get databse from ID session
driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()))
ID_missing = read_ID_missing("/home/nestor/Documents/Maestria/Avances Maestria/Database Codes/SVD_all_pathologies.csv",'a wav')
# Find the DB page
driver.get("http://stimmdb.coli.uni-saarland.de/index.php4#target")
driver.implicitly_wait(1)

# Change lenguage to english
english_button = driver.find_element(by=By.NAME, value="sb_lang")
english_button.click()
driver.implicitly_wait(1)
# Search for button to entry the database
search_button = driver.find_element(by=By.NAME, value="sb_search")

assert search_button.get_attribute('value') == "Database request", "Database expected to be in english"

# Click the button and wait for the page to load
search_button.click()
driver.implicitly_wait(1)

for ID in ID_missing:
    try:
        box_ID = driver.find_element(By.ID,"s_sess_id")
        box_ID.send_keys(ID)
        accept_button = driver.find_element(by=By.NAME, value="sb_sent")
        accept_button.click()
        driver.implicitly_wait(1)
        export_button = driver.find_element(by=By.XPATH,value = '/html/body/form/div[7]/span[5]/input')
        export_button.click()
        driver.implicitly_wait(1)
        i_neutral_check = driver.find_element(by=By.XPATH,value = '/html/body/form/div[3]/table/tbody/tr[1]/td[2]/table/tbody/tr[2]/td[2]/table/tbody/tr[2]/td[1]/input').click()
        a_neutral_check = driver.find_element(by=By.XPATH,value = '/html/body/form/div[3]/table/tbody/tr[1]/td[2]/table/tbody/tr[2]/td[2]/table/tbody/tr[3]/td[1]/input').click()
        u_neutral_check = driver.find_element(by=By.XPATH,value = '/html/body/form/div[3]/table/tbody/tr[1]/td[2]/table/tbody/tr[2]/td[2]/table/tbody/tr[4]/td[1]/input').click()
        sentence_check = driver.find_element(by=By.ID,value = 's_export_phrase').click()
        
        speech_wav_check = driver.find_element(by=By.ID,value = 's_exp_nspwav').click()
        egg_wav_check = driver.find_element(by=By.ID,value = 's_exp_eggegg').click()
        egg_egg_check = driver.find_element(by=By.ID,value = 's_exp_eggwav').click()
        
        accept_button = driver.find_element(by=By.XPATH,value ='/html/body/form/div[4]/span[3]/input').click()
        driver.implicitly_wait(4)
        #download_button = driver.find_element(by=By.XPATH,value ='/html/body/form/a/div[2]/table/tbody/tr/td[4]/a')
        element = WebDriverWait(driver, 3000).until(
        EC.element_to_be_clickable((By.XPATH, '/html/body/form/a/div[2]/table/tbody/tr/td[4]/a')))
        element.click()
        driver.implicitly_wait(2)
        print(str(ID)+" Donwloaded")
        back_button =driver.find_element(by=By.XPATH,value ='/html/body/form/a/div[1]/span[5]/input').click() 
        driver.implicitly_wait(2)
        
        reset_button = driver.find_element(by = By.XPATH, value = "/html/body/form/div[4]/span[4]/input")
        reset_button.click()
        driver.implicitly_wait(10)
        
    except:
        print(str(ID)+" has no audio files")
        continue