import pandas as pd
import pyarrow.parquet as pq
import pickle
import warnings
import math
from datetime import datetime, timedelta
import os
import geopandas as gpd
from shapely.geometry import Point


# Suppress specific UserWarning
warnings.filterwarnings('ignore', category=UserWarning, message='The CRS of your data is not defined.')
warnings.filterwarnings('ignore', message='The positionfixes with ids [.*] lead to invalid tripleg geometries.')

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
pd.options.display.max_rows = 2000

# Function to determine the day with a custom boundary at 5:00 AM
def custom_day(dt):
    # Adjust the date if the time is before 5:00 AM
    if dt.time() < datetime.strptime("05:00", "%H:%M").time():
        dt -= timedelta(days=1)
    return dt.date()

#folder_path = '../data/raw gps points/google-oauth2_109244383376429682325'

def data_for_trackintel(folder_name):
    
    folder_path = '../data/input/' + str(folder_name)
    
    # List all Parquet files in the folder
    parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]

    # Initialize an empty DataFrame to hold all the data
    all_data = pd.DataFrame()

    # Loop through the files and read each one into a DataFrame
    for file in parquet_files:
        file_path = os.path.join(folder_path, file)
        table = pq.read_table(file_path)
        df = table.to_pandas()

        # Convert the 't' column from Unix to datetime
        df['t'] = pd.to_datetime(df['t'])

        # Append this DataFrame to the main DataFrame
        all_data = pd.concat([all_data, df], ignore_index=True)

    all_data = all_data[all_data['accuracy'] < 100]
    all_data['user_id'] = 1
    all_data = all_data[['user_id','t','latitude', 'longitude', 'altitude','accuracy']]
    all_data.columns = ['user_id', 'tracked_at', 'latitude', 'longitude', 'elevation', 'accuracy']
    all_data['tracked_at'] = all_data['tracked_at'].dt.tz_localize('UTC')

    # Assuming 'latitude' and 'longitude' are your coordinate columns
    all_data['geometry'] = all_data.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    gall_data = gpd.GeoDataFrame(all_data, geometry='geometry')

    all_data.drop_duplicates(inplace=True)

    all_data['tracked_at'] = pd.to_datetime(all_data['tracked_at'], utc=True)


    # Apply the function to create a new column
    all_data['day_till_5am'] = all_data['tracked_at'].apply(custom_day)
    all_data['day_till_5am'] = pd.to_datetime(all_data['day_till_5am'])


    
    return all_data
    # all_data now contains all the data from the Parquet files in the folder