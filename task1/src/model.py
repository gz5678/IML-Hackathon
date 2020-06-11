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
        weather_data = pre_funcs.fix_weather_data(weather_data)
        merged_table = pre_funcs.merge_tables(train_data, weather_data)
        merged_table = pre_funcs.all_categoricals(merged_table)
        y = merged_table['ArrDelay']
        X = merged_table.drop(columns=['ArrDelay', 'DelayFactor'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)
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
