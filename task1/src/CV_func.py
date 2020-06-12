import pandas as pd
import numpy as np
import matplotlib as plt
from plotnine import *
#import preprocessing_funcs as pre_funcs
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso,LinearRegression,Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,BaggingRegressor, RandomForestClassifier
from xgboost import XGBRegressor
from tqdm import tqdm
import pickle


def my_cross(X_train, y_train, model):
 """""
 :param X_train: Feature matrix
 :param y_train: Response vector 
 :param model: Model to use, e.g. model = linear_model.Lasso()
 add tuning parameter if needed
 :return 5 fold cross validation test score 
 """""""""
 cv = cross_validate(model, X_train, y_train, scoring='neg_mean_squared_error')
 return cv


def best_tune(X_train, y_train, model, params):
 """""
 :param X_train: Feature matrix
 :param y_train: Response vector 
 :param model: Model to use, e.g. model = linear_model.Lasso()
 :param params: 
 :return best tuning parameter of 5 fold cross validation 
 """""""""
 cv = GridSearchCV(model, params, cv=5).fit(X_train, y_train)
 return cv


# Example
# best_tune(trainX, trainY, KNeighborsClassifier(), param_grid)

# prep model_list
Las = Lasso()
Rfc = RandomForestRegressor(random_state=2)
Xgb = XGBRegressor(random_state=2)

zipped = zip([Rfc, Las, Xgb],
             ['Random Forest', 'Lasso', 'XGB'])


def run_model(trainX, trainY, model_list=zipped):
 """""
 :param X_train: Feature matrix
 :param y_train: Response vector 
 :param model_list: 2 zipped lists, one with model objects, one with model name. 
 e.g. zip([Las],['Lasso'])
 :return Table of errors (estimated by cross validation) of each model
 """""""""

 # Tune parameters
 model_params = {'Random Forest': {'max_depth': [80, 90, 100, 110],
                                   'min_samples_split': [8, 10, 12],
                                   'n_estimators': [150]},
                 'Lasso': {'alpha': [3, 6, 8]},
                 'XGB': {'max_depth': [10, 30, 50],
                         'min_child_weight': [1, 3, 6],
                         'n_estimators': [200],
                         'learning_rate': [0.05, 0.1, 0.16]}}


 #####
 scores = []
 for model, name in tqdm(model_list):
  if name in model_params.keys():
   scores.append(best_tune(trainX, trainY, model, model_params[name]))
  else:
   scores.append(my_cross(trainX, trainY, model))

 return scores

def run_classification(X_train, y_train):
    params = {
        'max_depth': [80, 90, 100, 110],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [150]
    }
    model = RandomForestClassifier(random_state=2).fit(X_train, y_train)
    return best_tune(X_train, y_train, model, params)

def run_regression(X_train, y_train):
    model = RandomForestRegressor(random_state=2,
                                  max_depth=80,
                                  min_samples_split=10,
                                  n_estimators=150).fit(X_train, y_train)
    filename = 'final_reg_model_RandForest.sav'
    pickle.dump(model, open(filename, 'wb'))
    return model


def run_classify(X_train, y_train):
    model = RandomForestClassifier(random_state=2,
                                   max_depth=80,
                                   min_samples_split=12,
                                   n_estimators=150).fit(X_train, y_train)
    filename = 'final_class_model_RandForest.sav'
    pickle.dump(model, open(filename, 'wb'))
    return model
