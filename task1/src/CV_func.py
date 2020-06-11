import pandas as pd
import numpy as np
import matplotlib as plt
from plotnine import *
#import preprocessing_funcs as pre_funcs
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

# importing Various regression algorithms
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso,LinearRegression,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,BaggingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from tqdm import tqdm


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
LinR = LinearRegression()
Rid = Ridge()
Rfc = RandomForestRegressor(random_state=2)
Dtc = DecisionTreeRegressor(random_state=2)
# Las, LinR, Rid, Dtc, Rfc, Boost_Lin, Boost_las, Boost_rid, Bg_Lin, Bg_las, Bg_rid
# 'Lasso', 'Linear Regression', 'Ridge', 'Random forest Regressor', 'Decision Tree Regressor',
#               'Boosted Linear',
#               'Boosted Lasso', 'Boosted Ridge', 'Bagged Linear', 'Bagged Lasso', 'Bagged Ridge'


zipped = zip([Las, LinR, Rid, Rfc, Dtc],
             ['Lasso', 'Linear', 'Ridge', 'Random Forest', 'Decision Tree'])


def run_model(trainX, trainY, model_list=zipped):
 """""
 :param X_train: Feature matrix
 :param y_train: Response vector 
 :param model_list: 2 zipped lists, one with model objects, one with model name. 
 e.g. zip([Las],['Lasso'])
 :return Table of errors (estimated by cross validation) of each model
 """""""""

 # Tune parameters
 model_params = {'Random Forest': {'base_estimator__max_depth': [80, 90, 100, 110],
                                   'base_estimator__min_samples_split': [8, 10, 12],
                                   'n_estimators': [50, 100, 150]},
                 'Lasso': {'base_estimator__alpha': range(1, 10), 'n_estimator': [50, 100, 150]},
                 'Ridge': {'base_estimator__alpha': range(1, 10), 'n_estimator': [50, 100, 150]},
                 'Decision Tree': {'base_estimator__max_depth': [80, 90, 100, 110],
                                   'base_estimator__min_samples_split': [8, 10, 12],
                                   'n_estimators': [50, 100, 150]}}

 #####
 scores = []
 scores_boost = []
 scores_bag = []
 for model, name in tqdm(model_list):
  boost = AdaBoostRegressor(base_estimator=model)
  bag = BaggingRegressor(base_estimator=model)

  if name in model_params.keys():

   scores_boost.append(best_tune(trainX, trainY, boost, model_params[name]))
   scores_bag.append(best_tune(trainX, trainY, bag, model_params[name]))

   scores.append(best_tune(trainX, trainY, model, model_params[name]))
  else:
   scores_boost.append(my_cross(trainX, trainY, boost))
   scores_bag.append(my_cross(trainX, trainY, bag))

   scores.append(my_cross(trainX, trainY, model))

 return scores_boost, scores_bag, scores