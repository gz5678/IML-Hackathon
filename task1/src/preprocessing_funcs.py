import pandas as pd
import numpy as np
import matplotlib as plt
import re
from plotnine import *


def fix_weather_data(weather_data):
    """
    * replace "None" string and -100,-99 with nan
    * change all columns except date and station to numeric
    * replace temperatures over 130 farenheit with nan
    """
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
    for col in ['CRSDepTime', 'CRSArrTime']:
        time_col = df[col].astype(int).astype(str)
        time_col = time_col.apply(lambda x: x.zfill(4))
        time_col = time_col.apply(lambda x: re.sub(r'24(\d\d)',
                                                   r'00\1',
                                                   x))
        time_col = pd.to_datetime(time_col,
                                  format="%H%M")
        time_col = time_col.apply(lambda x: x.time())
        df[col] = time_col
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