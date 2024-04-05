#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:51:12 2024

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
from sklearn.metrics import classification_report,f1_score, confusion_matrix,accuracy_score
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from scipy import interp
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle

#%%
def statistical_information(y_true, y_predicted, classes):
    f1 = f1_score(y_true, y_predicted, average='weighted')
    
    print("Weighted F1 Score:", f1)
    
    conf_matrix = confusion_matrix(y_true, y_predicted, labels = classes)
    print("Class order: ", classes)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    num_classes = len(classes)  # Number of unique classes in the true labels
    
    sensitivity = []
    specificity = []
    
    for i, class_name in enumerate(classes):
        true_positive = conf_matrix[i, i]
        false_positive = sum(conf_matrix[:, i]) - true_positive
        false_negative = sum(conf_matrix[i, :]) - true_positive
        true_negative = sum(sum(conf_matrix)) - true_positive - false_positive - false_negative
    
        sensitivity_i = true_positive / (true_positive + false_negative)
        specificity_i = true_negative / (true_negative + false_positive)
    
        sensitivity.append(sensitivity_i)
        specificity.append(specificity_i)
    
    print("Sensitivity (True Positive Rate) for each class:", sensitivity)
    print("Specificity (True Negative Rate) for each class:", specificity)
    return f1, sensitivity, specificity,conf_matrix

#%%
def DecisionTree(X,y,folds, optimizer=True,*args, **kwargs):
    result_dict = {}
    if optimizer:
        param_grid = {'model__criterion':['gini','entropy'],
              'model__max_depth':np.arange(3,19).tolist()[0::2],
              'model__min_samples_split':np.arange(2,11).tolist()[0::2],
              'model__max_leaf_nodes':np.arange(3,20).tolist()[0::2]}
    acc_array = []
    for i, folder in enumerate(folds):
        #Get info of each folder
        train_set_folder = folds[folder]['train']
        test_set_folder = folds[folder]['test']

        X_train = X.iloc[train_set_folder]
        y_train = y.iloc[train_set_folder]

        X_test = X.iloc[test_set_folder]
        y_test = y.iloc[test_set_folder]
        print(f"FOLD #{i+1}")
        print(f"X train shape: {X_train.shape} and y train shape: {y_train.shape}")
        print(f"X test shape: {X_test.shape} and y test shape: {y_test.shape}")
        if optimizer:
            dt_classifier = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('model', DecisionTreeClassifier())
            ])
            grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
            classes_order = grid_search.classes_
            accuracy = best_model.score(X_test, y_test)
            y_predicted = best_model.predict(X_test)
            #decision_scores = best_model.decision_function(X_test)
            f1, sensitivity, specificity,c_matrix = statistical_information(list(y_test),list(y_predicted),classes_order)
            print(accuracy)

            result_dict[f'Fold {i+1}'] = {'best_params':best_params,
                                            'accuracy_best_model':accuracy,
                                            'f1_score':f1,
                                            'sensitivity':sensitivity,
                                            'specificity':specificity,
                                            'confusion_matrix':c_matrix,
                                            'class_order':classes_order}

        else:
            best_criterion = kwargs.get('criterion')
            best_depth = kwargs.get('max_depth')
            best_samples_split = kwargs.get('min_samples_split')
            best_leaf_nodes = kwargs.get('max_leaf_nodes')
            clf = make_pipeline(StandardScaler(), DecisionTreeClassifier(criterion = best_criterion, max_depth = best_depth, min_samples_split = best_samples_split,max_leaf_nodes = best_leaf_nodes))

            clf.fit(X_train, y_train)
            classes_order = clf.classes_
            y_predicted = clf.predict(X_test)
            prob_predicted = clf.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_predicted)

            #decision_scores = clf.decision_function(X_test)
            f1, sensitivity, specificity,c_matrix = statistical_information(list(y_test),list(y_predicted),classes_order)
            print(accuracy)

            result_dict[f'Fold {i+1}'] = {'param_used':{'criterion':best_criterion,
                                                        'max_depth':best_depth,
                                                        'min_samples_split':best_samples_split,
                                                        'max_leaf_nodes':best_leaf_nodes},
                                            'accuracy_best_model':accuracy,
                                            'f1_score':f1,
                                            'sensitivity':sensitivity,
                                            'specificity':specificity,
                                            'confusion_matrix':c_matrix,
                                            'class_order':classes_order,
                                            'prediction_label':y_predicted,
                                            'original_label':y_test,
                                            'probability':prob_predicted}

    return result_dict
def RandomForest(X,y,folds, optimizer=True,*args, **kwargs):
    result_dict = {}
    if optimizer:
        param_grid = {'model__max_features':['log2','sqrt'],
              'model__max_depth':np.arange(10,100).tolist()[0::10],
              'model__min_samples_split':np.arange(2,11).tolist()[0::2],
              'model__n_estimators':np.arange(200,2100).tolist()[0::2000]}
    acc_array = []
    for i, folder in enumerate(folds):
        train_set_folder = folds[folder]['train']
        test_set_folder = folds[folder]['test']

        X_train = X.iloc[train_set_folder]
        y_train = y.iloc[train_set_folder]

        X_test = X.iloc[test_set_folder]
        y_test = y.iloc[test_set_folder]
        print(f"FOLD #{i+1}")
        print(f"X train shape: {X_train.shape} and y train shape: {y_train.shape}")
        print(f"X test shape: {X_test.shape} and y test shape: {y_test.shape}")
        if optimizer:
            dt_classifier = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier())
            ])
            grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
            classes_order = grid_search.classes_
            accuracy = best_model.score(X_test, y_test)
            y_predicted = best_model.predict(X_test)
            #decision_scores = best_model.decision_function(X_test)
            f1, sensitivity, specificity,c_matrix = statistical_information(list(y_test),list(y_predicted),classes_order)
            print(accuracy)

            result_dict[f'Fold {i+1}'] = {'best_params':best_params,
                                            'accuracy_best_model':accuracy,
                                            'f1_score':f1,
                                            'sensitivity':sensitivity,
                                            'specificity':specificity,
                                            'confusion_matrix':c_matrix,
                                            'class_order':classes_order}

        else:
            best_max_features = kwargs.get('max_features')
            best_depth = kwargs.get('max_depth')
            best_samples_split = kwargs.get('min_samples_split')
            best_n_estimators = kwargs.get('n_estimators')
            clf = make_pipeline(StandardScaler(), RandomForestClassifier(max_features = best_max_features, max_depth = best_depth, min_samples_split = best_samples_split,n_estimators = best_n_estimators))

            clf.fit(X_train, y_train)
            classes_order = clf.classes_
            y_predicted = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_predicted)
            prob_predicted = clf.predict_proba(X_test)
            #decision_scores = clf.decision_function(X_test)
            f1, sensitivity, specificity,c_matrix = statistical_information(list(y_test),list(y_predicted),classes_order)
            print(accuracy)

            result_dict[f'Fold {i+1}'] = {'param_used':{'max_features':best_max_features,
                                                        'max_depth':best_depth,
                                                        'min_samples_split':best_samples_split,
                                                        'n_estimators':best_n_estimators},
                                            'accuracy_best_model':accuracy,
                                            'f1_score':f1,
                                            'sensitivity':sensitivity,
                                            'specificity':specificity,
                                            'confusion_matrix':c_matrix,
                                            'class_order':classes_order,
                                            'prediction_label':y_predicted,
                                            'original_label':y_test,
                                            'probability':prob_predicted}

    return result_dict


def SVM_Classifier(X,y,folds,optimizer = True,*args, **kwargs):
    result_dict = {}
    if optimizer:
        param_grid = {
            'model__C': [0.00001,0.0005,0.0001,0.005,0.001, 0.01, 0.1, 1],
            'model__kernel': ['linear', 'rbf'],
            'model__gamma': [0.00001,0.0005,0.0001,0.005,0.001,0.001, 0.01, 0.1, 1],
        }
    feature_name = list(X.columns)

    for i, folder in enumerate(folds):
        #Get info of each folder
        train_set_folder = folds[folder]['train']
        test_set_folder = folds[folder]['test']

        X_train = X.iloc[train_set_folder]
        y_train = y.iloc[train_set_folder]

        X_test = X.iloc[test_set_folder]
        y_test = y.iloc[test_set_folder]


        print(f"FOLD #{i+1}")
        print(f"X train shape: {X_train.shape} and y train shape: {y_train.shape}")
        print(f"X test shape: {X_test.shape} and y test shape: {y_test.shape}")
        if optimizer:
            svm_classifier = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('model', SVC(probability=True))
            ])
            grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, scoring='accuracy')

            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
            classes_order = grid_search.classes_
            accuracy = best_model.score(X_test, y_test)
            y_predicted = best_model.predict(X_test)
            decision_scores = best_model.decision_function(X_test)
            f1, sensitivity, specificity,c_matrix = statistical_information(list(y_test),list(y_predicted),classes_order)
            print(accuracy)

            result_dict[f'Fold {i+1}'] = {'best_params':best_params,
                                            'accuracy_best_model':accuracy,
                                            'f1_score':f1,
                                            'sensitivity':sensitivity,
                                            'specificity':specificity,
                                            'confusion_matrix':c_matrix,
                                            'scores':decision_scores,
                                            'class_order':classes_order}
        else:
            best_C = kwargs.get('C')
            best_gamma = kwargs.get('gamma')
            best_kernel = kwargs.get('kernel')
            clf = make_pipeline(StandardScaler(), SVC(C = best_C,gamma = best_gamma,kernel = best_kernel,probability=True))

            clf.fit(X_train, y_train)
            classes_order = clf.classes_
            y_predicted = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_predicted)
            prob_predicted = clf.predict_proba(X_test)
            decision_scores = clf.decision_function(X_test)
            f1, sensitivity, specificity,c_matrix = statistical_information(list(y_test),list(y_predicted),classes_order)
            print(accuracy)

            result_dict[f'Fold {i+1}'] = {'param_used':{'C':best_C,
                                                        'gamma':best_gamma,
                                                        'kernel':best_kernel},
                                            'accuracy_best_model':accuracy,
                                            'f1_score':f1,
                                            'sensitivity':sensitivity,
                                            'specificity':specificity,
                                            'confusion_matrix':c_matrix,
                                            'scores':decision_scores,
                                            'class_order':classes_order,
                                            'prediction_label':y_predicted,
                                            'original_label':y_test,
                                            'probability':prob_predicted}

    return result_dict

def SVM_Optimization(X,y, cv_folds, PATH_BEST_RESULTS_FOLDS,PATH_BEST_RESULTS_FIXED_PARAMS):
    result = SVM_Classifier(X,y,cv_folds)
    with open(PATH_BEST_RESULTS_FOLDS, 'wb') as handle:
        pickle.dump(result, handle)
    C_array = []
    gamma_array = []
    kernel_array = []
    for key in result:
        C_array.append(result[key]['best_params']['model__C'])
        gamma_array.append(result[key]['best_params']['model__gamma'])
        kernel_array.append(result[key]['best_params']['model__kernel'])


    best_C = max(set(C_array), key=C_array.count)
    best_gamma = max(set(gamma_array), key=gamma_array.count)
    best_kernel = max(set(kernel_array), key=kernel_array.count)
    best_params_dict = {'C':best_C,'gamma':best_gamma,'kernel':best_kernel}
    result_best_param = SVM_Classifier(X,y,cv_folds,False,**best_params_dict)
    with open(PATH_BEST_RESULTS_FIXED_PARAMS, 'wb') as handle:
        pickle.dump(result_best_param, handle)
    return result_best_param
def DT_Optimization(X,y, cv_folds, PATH_BEST_RESULTS_FOLDS,PATH_BEST_RESULTS_FIXED_PARAMS):
    result = DecisionTree(X,y,cv_folds)
    with open(PATH_BEST_RESULTS_FOLDS, 'wb') as handle:
        pickle.dump(result, handle)
    criterion_array = []
    depth_array = []
    samples_split_array = []
    leaf_nodes_array = []
    for key in result:
        criterion_array.append(result[key]['best_params']['model__criterion'])
        depth_array.append(result[key]['best_params']['model__max_depth'])
        samples_split_array.append(result[key]['best_params']['model__min_samples_split'])
        leaf_nodes_array.append(result[key]['best_params']['model__max_leaf_nodes'])


    best_criterion = max(set(criterion_array), key=criterion_array.count)
    best_depth = max(set(depth_array), key=depth_array.count)
    best_samples_split = max(set(samples_split_array), key=samples_split_array.count)
    best_leaf_nodes = max(set(leaf_nodes_array), key=leaf_nodes_array.count)
    best_params_dict = {'criterion':best_criterion,'max_depth':best_depth,'min_samples_split':best_samples_split, 'max_leaf_nodes':best_leaf_nodes}
    result_best_param = DecisionTree(X,y,cv_folds,False,**best_params_dict)
    with open(PATH_BEST_RESULTS_FIXED_PARAMS, 'wb') as handle:
        pickle.dump(result_best_param, handle)
    return result_best_param

def RF_Optimization(X,y, cv_folds, PATH_BEST_RESULTS_FOLDS,PATH_BEST_RESULTS_FIXED_PARAMS):
    result = RandomForest(X,y,cv_folds)
    with open(PATH_BEST_RESULTS_FOLDS, 'wb') as handle:
        pickle.dump(result, handle)
    max_features_array = []
    depth_array = []
    samples_split_array = []
    n_estimators_array = []

    for key in result:
        max_features_array.append(result[key]['best_params']['model__max_features'])
        depth_array.append(result[key]['best_params']['model__max_depth'])
        samples_split_array.append(result[key]['best_params']['model__min_samples_split'])
        n_estimators_array.append(result[key]['best_params']['model__n_estimators'])


    best_max_features = max(set(max_features_array), key=max_features_array.count)
    best_depth = max(set(depth_array), key=depth_array.count)
    best_samples_split = max(set(samples_split_array), key=samples_split_array.count)
    best_n_estimators = max(set(n_estimators_array), key=n_estimators_array.count)
    best_params_dict = {'max_features':best_max_features,'max_depth':best_depth,'min_samples_split':best_samples_split, 'n_estimators':best_n_estimators}
    result_best_param = RandomForest(X,y,cv_folds,False,**best_params_dict)
    with open(PATH_BEST_RESULTS_FIXED_PARAMS, 'wb') as handle:
        pickle.dump(result_best_param, handle)
    return result_best_param