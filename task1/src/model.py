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
import preprocessing_funcs as pre_funcs


class FlightPredictor:
    def __init__(self, path_to_weather='all_weather_data.csv'):
        """
        Initialize an object from this class.
        @param path_to_weather: The path to a csv file containing weather data.
        """
        train_data = pd.read_csv('train_data.csv')
        weather_data = pd.read_csv(path_to_weather).rename(columns={'day': 'FlightDate'})
        # Fix date format
        train_data['FlightDate'] = train_data['FlightDate'].apply(lambda x: re.sub(r'(\d\d)(\d\d)(-\d+-)(\d+)',
                                                                                   r'\4\3\2',
                                                                                   x))
        weather_data = pre_funcs.fix_weather_data(weather_data)
        merged_table = pre_funcs.merge_tables(train_data, weather_data)
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
