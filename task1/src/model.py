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
from sklearn.preprocessing import data
from sklearn.metrics import mean_squared_error
import preprocessing_funcs as pre_funcs
import CV_func as models_exec

class FlightPredictor:
    def __init__(self, path_to_weather='all_weather_data.csv'):
        """
        Initialize an object from this class.
        @param path_to_weather: The path to a csv file containing weather data.
        """
        np.random.seed(0)
        np.set_printoptions(precision=5)
        train_data = pd.read_csv('train_data.csv')
        weather_data = pd.read_csv(path_to_weather).rename(columns={'day': 'FlightDate'})
        # Fix date format
        train_data['FlightDate'] = train_data['FlightDate'].apply(lambda x: re.sub(r'(\d\d)(\d\d)(-\d+-)(\d+)',
                                                                                   r'\4\3\2',
                                                                                   x))
        merged_table = pre_funcs.merge_tables(train_data, weather_data)
        merged_table.to_pickle('merged_table.pkl')
        # merged_table = pd.read_pickle('merged_table.pkl')
        y = merged_table['ArrDelay']
        X = merged_table.drop(columns=['ArrDelay',
                                       'DelayFactor',
                                       'Tail_Number',
                                       'Flight_Number_Reporting_Airline'])
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

        X_train = pre_funcs.date_to_timestamp(X_train)
        # X_train.to_pickle('x_train.pkl')
        # y_train.to_pickle('y_train.pkl')
        # X_train = pd.read_pickle('x_train.pkl')
        # y_train = pd.read_pickle('y_train.pkl')
        X_train_short = X_train.iloc[:150000, :]
        y_train_short = y_train.iloc[:150000]
        full_df = pd.concat([X_train_short, y_train_short], axis=1)
        full_df = full_df.dropna()
        y_train_short = full_df['ArrDelay']
        X_train_short = full_df.drop(columns=['ArrDelay'])
        boost, bag, mod = models_exec.run_model(X_train_short, y_train_short)
        # print("The boost scores is: ")
        # print(scores_boost)
        # print("The bag scores is: ")
        # print(scores_bag)
        # print("The scores is: ")
        # print(scores)
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
