#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 12:40:33 2022

@author: nestor
"""
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
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.model_selection import train_test_split
#%%
def impute_outliers_IQR(df):
   df = df.dropna(thresh = 5)
   q1=df.quantile(0.25)

   q3=df.quantile(0.75)

   IQR=q3-q1

   upper = df[~(df>(q3+1.5*IQR))].max()

   lower = df[~(df<(q1-1.5*IQR))].min()

   df = np.where(df > upper,

       df.mean(),

       np.where(

           df < lower,

           df.mean(),

           df

           )

       )

   return df
# %% Extract list of balance dataset of each feature set
DATASETS_PATH = "/home/nestor/Documents/Maestria/Avances Maestria/Dataframe For Models/"

# list to store files
dict_features = {}
signal_type = 'speech'
# Iterate directory
for path in os.listdir(DATASETS_PATH):
    # check if current path is a file
    if os.path.isfile(os.path.join(DATASETS_PATH, path)):
        if signal_type in path:
            print(path)
            key = path.split('.')[0].split('_')[-2]
            
            dict_features[key] = os.path.join(DATASETS_PATH, path)

#%%
all_path = "/home/nestor/Documents/Maestria/Avances Maestria/Features Disvoice/SVD_all_pathologies_and_HC.csv"
get_id = pd.read_csv(all_path)
get_id = get_id.drop_duplicates(subset=['ID'])
df_path = "/home/nestor/Documents/Maestria/Avances Maestria/Features Disvoice/Features/egg_non_linear_features.csv"
df = pd.read_csv(df_path, index_col=0)
df_3 = pd.merge(df, get_id[["ID","T"]], on="ID")
df_3["T"] = np.where(df_3["T"] == "p", 1, 0)
# %% Complete training
fig1 = plt.figure(figsize=[12,12])
ax1 = fig1.add_subplot(111,aspect = 'equal')  

key = "periodicity"
print(key)
if key == "periodicity":
    #df = pd.read_csv(df_path, index_col=0)
    df = df_3.dropna()
    #df = df_3
    #df = df.reset_index()
    df = df.drop(columns = ["ID"])
    #df_new = impute_outliers_IQR(df)
    #df = pd.DataFrame(df_new)
    # Get dataset in X and Y
    
    y = df.iloc[:, -1]
    X = df.iloc[:,0:-1]
    #X = X.drop(columns = ["ID","Unnamed: 0"])
    total_index = list(y.index)
    # Fold

    ACC_array = []
    AUC_array = []
    SPE_array = []
    SEN_array = []
    probab_array = []
    y_t_array = []
    tprs = []
    aucs = []
    y0_preds=np.array([])
    y1_preds=np.array([])
    mean_fpr = np.linspace(0,1,100)
    i = 0 
    #for i in tqdm(range(10),desc='test loop'):
    #test_index = y.sample(n=150, random_state = 42).index
    
    #t_index = np.setdiff1d(total_index,test_index)
    #for t_index, test_index in tqdm(skf.split(X, y), total=skf.get_n_splits(), desc='1st loop'):
    #print("TRAIN:", train_index, "TEST:", test_index)
    # Data splitting
    scaler = StandardScaler()
# =============================================================================
#         X_train, X_test = X.iloc[t_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[t_index], y.iloc[test_index]
# =============================================================================
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.2, random_state=42)
    #print(y_test)
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)
    
    pca = PCA(n_components=3)
    scaler = StandardScaler()
    #X_train = pca.fit_transform(X_train)
    #X_test = pca.transform(X_test)
       
    param_grid = [
      {'C': [0.01,0.1, 1,10,10], 'kernel': ['linear']},
      {'C': [0.01,0.1, 1,10,10], 'gamma': [0.001,0.01,0.1,1], 'kernel': ['rbf']},
      ]
    SVM = svm.SVC(probability=True)
    SVM_new_model = GridSearchCV(SVM, param_grid=param_grid, n_jobs=-1,refit=True,cv = 3,verbose=5)
    SVM_new_model.fit(X_train, y_train)
    print("Best Hyper Parameters:\n",SVM_new_model.best_params_)
    #Prediction
    
    #SVM_new_model = svm.SVC(**model_SVM.best_params_, probability=True) 
    #SVM_new_model.fit(X_train,y_train)
    prediction=SVM_new_model.predict(X_test)
    
    prediction_probability = SVM_new_model.predict_proba(X_test)[:, 1]
    probab_array=SVM_new_model.predict_proba(X_test)
    y0_preds= np.hstack([y0_preds,probab_array[np.where(y_test == 0), 1]]) if y0_preds.size else probab_array[np.where(y_test == 0), 1]
    y1_preds= np.hstack([y1_preds,probab_array[np.where(y_test == 1), 1]]) if y1_preds.size else probab_array[np.where(y_test == 1), 1]
    #evaluation(Accuracy)
    print("Accuracy:",accuracy_score(prediction,y_test))
    #evaluation(Confusion Metrix)
    cm1 = confusion_matrix(prediction,y_test)
    print("Confusion Metrix:\n",confusion_matrix(prediction,y_test))
    
    total1=sum(sum(cm1))
    Accuracy = (cm1[0,0]+cm1[1,1])/total1
    Specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    Sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    
    ACC_array.append(Accuracy)
    SPE_array.append(Specificity)
    SEN_array.append(Sensitivity)
    
    fpr, tpr, t = roc_curve(y_test, prediction_probability)
    roc_auc = auc(fpr, tpr)
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    AUC_array.append(roc_auc)
    i= i+1
       
    
    # ROC curve
    mean_tpr_sta = np.mean(tprs, axis=0)
    mean_auc_sta = auc(mean_fpr, mean_tpr_sta)
    plt.plot(mean_fpr, mean_tpr_sta,
             label=r'ROC curve of %s features (AUC = %0.2f )' % (key,mean_auc_sta),lw=2, alpha=1)  
    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(r'ROC curve for %s features' % (key))
    plt.legend(loc="lower right")
    plt.show()
    
    fig2 = plt.figure(figsize=[12,12])
    sns.distplot(y0_preds, color='blue', kde_kws={'linestyle':'--'})
    sns.distplot(y1_preds, color='red')
    plt.title(r'Score distribution of %s features' % (key))
    plt.legend(['Healthy','Pathologycal'])
    plt.show()
    #plt.show()
    
#%%

print("Accuracy: ", np.mean(ACC_array), "+-", np.std(ACC_array))
print("Specificity: ", np.mean(SPE_array), "+-", np.std(SPE_array))
print("Sensitivity: ", np.mean(SEN_array), "+-", np.std(SEN_array))

#%%
mean_tpr_linear = np.mean(tprs, axis=0)
mean_auc_linear = auc(mean_fpr, mean_tpr_linear)
plt.plot(mean_fpr, mean_tpr_linear,
         label=r'ROC curve of %s features (AUC = %0.2f )' % (key,mean_auc_linear),lw=2, alpha=1)  
plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(r'ROC curve for %s features' % (key))
plt.legend(loc="lower right")
plt.show()

fig2 = plt.figure(figsize=[12,12])
sns.distplot(y0_preds, color='blue', kde_kws={'linestyle':'--'})
sns.distplot(y1_preds, color='red')
plt.title(r'Score distribution of %s features' % (key))
plt.legend(['Healthy','Pathologycal'])
plt.show()