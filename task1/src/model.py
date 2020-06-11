"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Author(s):

===================================================
"""

import pandas as pd
import numpy as np
import matplotlib as plt
from plotnine import *
import re
import feather
from sklearn.model_selection import train_test_split
import preprocessing_funcs as pre_funcs
from sklearn.preprocessing import LabelBinarizer

class FlightPredictor:
    def __init__(self, path_to_weather='all_weather_data.csv'):
        """
        Initialize an object from this class.
        @param path_to_weather: The path to a csv file containing weather data.
        """
        np.random.seed(0)
        train_data = pd.read_csv('train_data.csv')
        weather_data = pd.read_csv(path_to_weather).rename(columns={'day': 'FlightDate'})
        # Fix date format
        train_data['FlightDate'] = train_data['FlightDate'].apply(lambda x: re.sub(r'(\d\d)(\d\d)(-\d+-)(\d+)',
                                                                                   r'\4\3\2',
                                                                                   x))
        merged_table = pre_funcs.merge_tables(train_data, weather_data)
        # merged_table.to_pickle('merged_table.pkl')
        # merged_table = pd.read_pickle('merged_table.pkl')
        y = merged_table['ArrDelay']
        X = merged_table.drop(columns=['ArrDelay', 'DelayFactor'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)
        X_train = pre_funcs.fix_weather_data(X_train)
        X_train = pre_funcs.all_categoricals(X_train)
        
        X_val= pre_funcs.fix_weather_test_data(X_val)
        X_val= pre_funcs.all_categoricals(X_val)
        
        X_test = pre_funcs.fix_weather_test_data(X_test)
        X_test = pre_funcs.all_categoricals(X_test)
        
        ## remove na's
        y_train = y_train[-X_train.isnull().any(axis=1)]
        X_train = X_train[-X_train.isnull().any(axis=1)]
        
        raise NotImplementedError

    def predict(self, x):
        """
        Recieves a pandas DataFrame of shape (m, 15) with m flight features, and predicts their
        delay at arrival and the main factor for the delay.
        @param x: A pandas DataFrame with shape (m, 15)
        @return: A pandas DataFrame with shape (m, 2) with your prediction
        """
        raise NotImplementedError

data = FlightPredictor()
