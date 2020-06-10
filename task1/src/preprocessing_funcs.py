import pandas as pd
import numpy as np
import matplotlib as plt
from plotnine import *


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