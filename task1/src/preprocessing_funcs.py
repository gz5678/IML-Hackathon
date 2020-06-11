import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import LabelBinarizer
import re
from plotnine import *
import inflect


def fix_weather_data(weather_data):
    """
    * replace "None" string and -100,-99 with nan
    * change all columns except date and station to numeric
    * replace temperatures over 130 farenheit with nan
    """
    weather_data = weather_data.drop(columns=['min_feel', 'avg_feel', 'max_feel', 'climo_high_f', 'climo_low_f', 'climo_precip_in'])
    weather_data[['snow_in', 'snowd_in', 'precip_in']] = weather_data[['snow_in', 'snowd_in', 'precip_in']].replace(to_replace=["None","-100","-99"], value=0)
    weather_data.replace(to_replace=["None","-100","-99"], value=np.nan, inplace=True)
    weather_data.iloc[:,2:] = weather_data.iloc[:,2:].apply(pd.to_numeric)    
    weather_data['max_temp_f'][weather_data['max_temp_f']>130] = np.nan
    return weather_data



def merge_tables(flight_data, weather_data):
    # Create origin and dest data
    weather_origin = weather_data.drop(columns=['station', 'FlightDate'])
    weather_dest = weather_origin.copy()
    weather_origin = weather_origin.add_suffix('_origin')
    weather_dest = weather_dest.add_suffix('_dest')
    weather_origin = pd.concat([weather_data[['station', 'FlightDate']], weather_origin],
                               axis=1).rename(columns={'station': 'Origin'})
    weather_dest = pd.concat([weather_data[['station', 'FlightDate']], weather_dest],
                             axis=1).rename(columns={'station': 'Dest'})

    """
    * Merge into flight_data: We merge the origin data and then the dest.
    * If for some flight data no weather is found for origin, it's origin weather
    * data columns will be na. Same for dest
    """
    temp = flight_data.merge(weather_origin, on=['Origin', 'FlightDate'], how='left')
    merged = temp.merge(weather_dest, on=['Dest', 'FlightDate'], how='left')
    return merged


def crs_to_time(df):
    p = inflect.engine()
    for col in ['CRSDepTime', 'CRSArrTime']:
        dummy_pred = col.split('Time')[0]
        col_name = col + "_Code"
        df[col] = df[col].apply(lambda x: x if x <= 2359 else 0)
        df[col_name] = df[col].apply(lambda x: np.floor(x / 100))
        df['temp'] = df[col_name].apply(lambda x: p.number_to_words(int(x / 6)))
        print(df['temp'].head())
        df = pd.get_dummies(df, prefix=dummy_pred, columns=['temp'], drop_first=True)
    return df


def make_categorical(name,df):
    dummies = pd.get_dummies(pd.Series(df[name]), drop_first=True, prefix = name)
    return dummies


def all_categoricals(merged_data):

    Reporting_Airline_bin = make_categorical('Reporting_Airline',merged_data)

    Origin_bin = make_categorical(r'Origin',merged_data)
    OriginState_bin = make_categorical(r'OriginState',merged_data)
    
    Dest_bin = make_categorical(r'Dest',merged_data)
    DestState_bin = make_categorical(r'DestState',merged_data)
    
    flights_binarized = pd.concat(([Reporting_Airline_bin,Origin_bin,OriginState_bin,Dest_bin,
                                        DestState_bin]),axis=1)
    
    merged_data = merged_data.drop(['Reporting_Airline', 'Origin','OriginCityName','OriginState','Dest','DestCityName','DestState'], axis=1)
    merged_data = pd.concat([merged_data, flights_binarized], axis=1)
    return merged_data
