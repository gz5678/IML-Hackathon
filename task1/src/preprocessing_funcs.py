import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
import re
from plotnine import *
import inflect


def fix_weather_data(weather_data):
    """
    * replace "None" string and -100,-99 with nan
    * change all columns except date and station to numeric
    * replace temperatures over 130 farenheit with nan
    """
    for prefix, col_start in zip(['_origin', '_dest'],[13, 27]):
        varlist = ['max_temp_f', 'min_temp_f', 'max_dewpoint_f', 'min_dewpoint_f',
                   'avg_wind_speed_kts','avg_wind_drct', 'min_rh', 'avg_rh', 'max_rh',
                   'max_wind_speed_kts', 'max_wind_gust_kts']
        varlist = list(map(lambda org_string: org_string + prefix, varlist))
        drop_columns = ['min_feel', 'avg_feel', 'max_feel', 'climo_high_f', 'climo_low_f', 'climo_precip_in']
        drop_columns = list(map(lambda org_string: org_string + prefix, drop_columns))

        none_columns = ['snow_in', 'snowd_in', 'precip_in']
        none_columns = list(map(lambda org_string: org_string + prefix, none_columns))

        weather_data = weather_data.drop(columns=drop_columns)
        weather_data[none_columns] = weather_data[none_columns].replace(to_replace=["None","-100","-99", np.nan], value=0)
        weather_data[varlist] = weather_data[varlist].replace(to_replace=["None","-100","-99", np.nan], value=-1000)
        weather_data.iloc[:, col_start:col_start+14] = weather_data.iloc[:, col_start:col_start+14].apply(pd.to_numeric)
        max_temp_pref = 'max_temp_f' + prefix
        weather_data[max_temp_pref][weather_data[max_temp_pref] > 130] = -1000
        weather_data[varlist] = weather_data[varlist].replace(to_replace=[-1000], value=np.nan)
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


def date_to_timestamp(df):
    df['FlightDate'] = pd.to_datetime(df['FlightDate'], format="%d-%m-%y").apply(lambda x: x.timestamp())
    return df


def make_categorical(name,df):
    dummies = pd.get_dummies(pd.Series(df[name]), drop_first=True, prefix = name)
    return dummies


def all_categoricals(merged_data):

    Reporting_Airline_cat = pd.DataFrame({'Reporting_Airline': LabelEncoder().fit_transform(merged_data['Reporting_Airline'])})

    Origin_cat = pd.DataFrame({'Origin': LabelEncoder().fit_transform(merged_data['Origin'])})
    OriginState_cat = pd.DataFrame({'OriginState': LabelEncoder().fit_transform(merged_data['OriginState'])})
    
    Dest_cat = pd.DataFrame({'Dest': LabelEncoder().fit_transform(merged_data['Dest'])})
    DestState_cat = pd.DataFrame({'DestState': LabelEncoder().fit_transform(merged_data['DestState'])})
    Day_cat = pd.DataFrame({'DayOfWeek': LabelEncoder().fit_transform(merged_data['DayOfWeek'])})
    
    flights_binarized = pd.concat(([Reporting_Airline_cat,Origin_cat,OriginState_cat,Dest_cat,
                                        DestState_cat, Day_cat]),axis=1)
    
    merged_data = merged_data.drop(['Reporting_Airline',
                                    'Origin',
                                    'OriginCityName',
                                    'OriginState',
                                    'Dest',
                                    'DestCityName',
                                    'DestState',
                                    'DayOfWeek'], axis=1)
    merged_data = pd.concat([merged_data, flights_binarized], axis=1)
    return merged_data


def fix_weather_test_data(weather_data):
    """
    * replace "None" string and -100,-99 with nan
    * change all columns except date and station to numeric
    * replace temperatures over 130 farenheit with nan
    """
    weather_data['month'] = [item[1] for item in weather_data['FlightDate'].str.split(pat="-")]
    for prefix in ['_origin', '_dest']:
        
        if prefix == '_origin':
            state = 'OriginState'
        else:
            if prefix == '_dest':
                state = 'DestState'
            else:
                    pass
        
        
        varlist = ['max_temp_f', 'min_temp_f', 'max_dewpoint_f', 'min_dewpoint_f',
                   'avg_wind_speed_kts','avg_wind_drct', 'min_rh', 'avg_rh', 'max_rh',
                   'max_wind_speed_kts', 'max_wind_gust_kts']
        varlist = list(map(lambda org_string: org_string + prefix, varlist))
        drop_columns = ['min_feel', 'avg_feel', 'max_feel', 'climo_high_f', 'climo_low_f', 'climo_precip_in']
        drop_columns = list(map(lambda org_string: org_string + prefix, drop_columns))

        none_columns = ['snow_in', 'snowd_in', 'precip_in']
        none_columns = list(map(lambda org_string: org_string + prefix, none_columns))

        weather_data = weather_data.drop(columns=drop_columns)
        weather_data[none_columns] = weather_data[none_columns].replace(to_replace=["None","-100","-99",np.nan], value=0)
        weather_data[none_columns] = weather_data[none_columns].apply(pd.to_numeric)
        weather_data[varlist] = weather_data[varlist].replace(to_replace=["None","-100","-99", np.nan], value=-1000)
        weather_data[varlist] = weather_data[varlist].apply(pd.to_numeric)
        max_temp_pref = 'max_temp_f' + prefix
        weather_data[max_temp_pref][weather_data[max_temp_pref] > 130] = -1000
        weather_data[varlist] = weather_data[varlist].replace(to_replace=[-1000], value=np.nan)
        weather_data[varlist] = weather_data[varlist].fillna(weather_data.groupby([state,'month'])[varlist].transform('mean'))
        weather_data[varlist] = weather_data[varlist].fillna(weather_data.groupby(['month'])[varlist].transform('mean'))
    weather_data = weather_data.drop(columns='month', axis = 1)
    return weather_data

def prepare_classify(df):
    le_classify = LabelEncoder()
    df['DelayFactor'] = df['DelayFactor'].replace(to_replace=[np.nan], value="NotDelayed")
    encoding = le_classify.fit(df['DelayFactor'])
    encoding_mapping = dict(zip(encoding.transform(encoding.classes_), encoding.classes_))
    df['DelayFactor'] = le_classify.fit_transform(df['DelayFactor'])
    return df, encoding_mapping
