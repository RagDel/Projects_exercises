import pandas as pd
import numpy as np
import requests
import json 
from sklearn.cluster import DBSCAN
import math
from datetime import datetime, timedelta
import time
import random
import trackintel as ti
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import ConvexHull
import warnings
from scipy.spatial import QhullError
import pyproj

# Suppress specific UserWarning
warnings.filterwarnings('ignore', category=UserWarning, message='The CRS of your data is not defined.')
warnings.filterwarnings('ignore', message='The positionfixes with ids [.*] lead to invalid tripleg geometries.')
warnings.filterwarnings('ignore', message="The positionfixes with ids \[.*\] lead to invalid tripleg geometries.*")
warnings.filterwarnings('ignore', message="The positionfixes with ids \[1088 1089\] lead to invalid tripleg geometries.*")
warnings.filterwarnings('ignore', message="The positionfixes with ids .* lead to invalid tripleg geometries.*")

place_types = pd.read_pickle("../data/processed/place_types_df.pkl")
qol_countries = pd.read_pickle("../data/external/Numbeo_Countries.pkl")
api_key = 'AIzaSyDCL0QT2X4-JMar3AxMbDaFrHrDChTTmeo' # Mario's API

def list_of_tuples_gps(df):
    points = []
    for index,row in df.iterrows():
        points.append((row['latitude'], row['longitude']))
    return points



def calculate_convex_hull_area_and_perimeter2(lat_lon_points):
    if len(lat_lon_points) < 3:
        return None, None

    try:
        # Convert lat/lon to meters
        meters_points = []
        for lat, lon in lat_lon_points:
            x = lon * 111320 * math.cos(math.radians(lat))
            y = lat * 111320
            meters_points.append((x, y))

        # Calculate convex hull
        hull = ConvexHull(np.array(meters_points))

        # Area in square meters and perimeter in meters
        area = hull.area
        perimeter = hull.length

        return area, perimeter
    except Exception as e:
        # Handle the error
        print(f"Error: {e}")
        return None, None

def calculate_convex_hull_area_and_perimeter(points):
    # First, check if there are enough points
    if len(points) < 3:
        return None, None

    try:
        hull = ConvexHull(points)
        return hull.area, hull.volume
        # Your code to calculate area and perimeter
    except QhullError:
        # Handle the error, for example by returning None or logging an error message
        return None, None

def time_away_and_average_distance(gps_data, home_location, distance_threshold=15):
    total_distance = 0
    time_away_minutes = 0

    for point in gps_data:
        #distance = calculate_distance(home_location, point)
        distance = haversine(home_location[0], home_location[1], point[0], point[1])
        if  (distance < 30000):
            #print(distance)
            total_distance += distance
        if (distance > distance_threshold) & (distance < 30000):
            #print(distance)
            #total_distance += distance
            time_away_minutes += 1  # Increment by 1 minute for each data point away from home

    average_distance = total_distance / len(gps_data)
    percentage_time_away = (time_away_minutes / len(gps_data)) * 100
    return percentage_time_away, average_distance

def places_api(lat, lng, home, radius=15, api_key='AIzaSyDCL0QT2X4-JMar3AxMbDaFrHrDChTTmeo'):
    """
    This function interacts with the Google Places API to find places of interest within a specified radius of a given latitude and longitude.

    Args:
    lat (float): Latitude of the location around which to search.
    long (float): Longitude of the location around which to search.
    radius (int, optional): The radius (in meters) within which to conduct the search. Default is 5 meters.

    Returns:
    dict: A dictionary containing the JSON response from the API with details of nearby places.
    """

    # Construct the URL for the API request using the provided latitude, longitude, and radius.
    # It includes the API key for authentication.
    if haversine(home[0], home[1], lat, lng) < 35:
      return None
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius={radius}&key={api_key}"

    # Make a GET request to the Google Places API and store the response.
    response = requests.get(url)

    # Return the JSON content of the response.
    return response.json()

def find_nearest_point_of_interest(response, lat, long):
    if response is None:
        return None
    results = response['results']
    businesses = []
    distance = 9999999999999

    for result in results:
        if ('business_status' in result.keys()):
            if (result['business_status'] == 'OPERATIONAL'):
                businesses.append(result)
    if not businesses:
        return None
    point_of_interest_types = []
    number = 0
    for n, business in enumerate(businesses):
        temp_distance = haversine(lat, long, business['geometry']['location']['lat'], business['geometry']['location']['lng'])
        if distance > temp_distance:
            distance = temp_distance
            point_of_interest_types = business['types']
            number = n
    return point_of_interest_types

def define_type_of_point_of_interest(types_list, place_types_df = place_types):
    all_types_of_places = list(place_types['Place_Type'])
    if types_list is None:
        return None
    type_of_poi = None
    for type in types_list:
        if type == 'food':
            type_of_poi = 'Entertainment and Food'
        if type in all_types_of_places:
            type_of_poi = place_types[place_types['Place_Type'] == type]['Category'].iloc[0]
    return type_of_poi

def track_intel(df):
    df.drop_duplicates(inplace=True)

    pfs = ti.Positionfixes(df)
    # Now pfs is a Trackintel Positionfixes object
    pfs, sp = pfs.as_positionfixes.generate_staypoints(method='sliding')

    total_walking_time = timedelta(0)
    if len(sp) > 2:
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp, method='between_staypoints')
        tpls = tpls.predict_transport_mode()
        walking = tpls.copy()
        walking = walking[walking['mode'] == 'slow_mobility']
        walking['walk_duration'] = walking['finished_at'] - walking['started_at']
        total_walking_time = walking['walk_duration'].sum()
    
    sp_map = sp.copy()
    latitude_sp = []
    longitude_sp = []

    for point in sp_map['geometry']:
        latitude_sp.append(point.y)
        longitude_sp.append(point.x)

    sp_map['latitude'] = latitude_sp
    sp_map['longitude'] = longitude_sp
    return sp_map, total_walking_time

def iterating_staypoints_df(sp, home, api_key='AIzaSyDCL0QT2X4-JMar3AxMbDaFrHrDChTTmeo'):
    types = {}
    for index, row in sp.iterrows():
        lat = row['latitude']
        long = row['longitude']
        type = define_type_of_point_of_interest(find_nearest_point_of_interest(places_api(lat, long, home), lat, long))
        if type in types.keys():
            types[type] += 1
        else:
            types[type] = 1
    return types

def find_home(data):
    """
    This function identifies the most likely 'home' location from a dataset of GPS coordinates based on the frequency and time of visits.

    Args:
    data (DataFrame): A pandas DataFrame containing columns 'tracked_at', 'latitude', and 'longitude' of the tracking data.

    Returns:
    numpy.ndarray or None: Returns the median coordinates of the most frequently visited location during night hours, assumed to be 'home', or None if no such location is identified.
    """

    # Filter data to only include coordinates tracked between 22:00 - 04:59 (night hours)
    data = data[data['tracked_at'].dt.hour.isin(list(range(22, 24)) + list(range(0, 5)))]

    # If there are no data points in this time range, return None
    if len(data) == 0:
        return None

    # Convert latitude and longitude to a numpy array for clustering
    coords = data[['latitude', 'longitude']].to_numpy()

    # Apply DBSCAN clustering algorithm to group nearby points.
    # eps (maximum distance between two samples for one to be considered as in the neighborhood of the other) and min_samples (number of samples in a neighborhood for a point to be considered as a core point)
    # need to be adjusted based on the density and distribution of data.
    db = DBSCAN(eps=0.001, min_samples=3).fit(coords)

    # Extract cluster labels for each point in the dataset
    cluster_labels = db.labels_

    # Identify the most frequent cluster (excluding noise points, which have label -1)
    try:
        most_common_cluster = np.argmax(np.bincount(cluster_labels[cluster_labels >= 0]))
    except ValueError:
        # In case bincount fails (e.g., all points are noise), return None
        return None

    # Compute the median of the coordinates in the most common cluster, assumed to be 'home'
    home_location = np.median(coords[cluster_labels == most_common_cluster], axis=0)

    return home_location

def define_None_home(home_locations):
    """
    This function fills in missing 'home' locations in a series of daily locations.
    If the 'home' location for a particular day is None, it is replaced with the 'home' location of the previous day.

    Args:
    home_locations (dict): A dictionary where keys are dates (or any sequential identifiers) and values are home location coordinates (or None).

    Returns:
    dict: The updated dictionary with missing 'home' locations filled in based on the previous day's location.
    """

    # Convert the values and keys of the dictionary to lists for easier processing
    homes_lst = list(home_locations.values())
    days_lst = list(home_locations.keys())

    # Iterate through the list of home locations
    for n, home in enumerate(homes_lst):
        # Check if the current home location is None and it's not the first item in the list
        if (home is None) and (n != 0):
            # If True, replace the None value with the home location of the previous day
            home_locations[days_lst[n]] = home_locations[days_lst[n-1]]

    # Return the updated dictionary
    return home_locations

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface given their latitude and longitude.

    Args:
    lat1 (float): Latitude of the first point in decimal degrees.
    lon1 (float): Longitude of the first point in decimal degrees.
    lat2 (float): Latitude of the second point in decimal degrees.
    lon2 (float): Longitude of the second point in decimal degrees.

    Returns:
    float: The distance between the two points in meters.

    The Haversine formula is used to calculate the distance. It is an approximation but works well for most practical purposes.
    """

    # Radius of the Earth in kilometers. The average radius is used here.
    R = 6371.0

    # Convert latitudes and longitudes from degrees to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon1_rad = math.radians(lon1)
    lon2_rad = math.radians(lon2)

    # Difference in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula: It calculates the circular distance between two points on a sphere.
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in meters. The Earth's radius is multiplied by the central angle.
    distance = R * c * 1000

    return distance

def format_duration(td):
    """
    Format a timedelta object into a string showing days, hours, and minutes.

    Args:
    td (timedelta): A timedelta object representing a duration of time.

    Returns:
    str: A string representation of the duration in the format "{days} days {hours} hours {minutes} min".

    This function extracts days, hours, and minutes from a timedelta object and formats them into a readable string.
    """

    # Extract the number of whole days from the timedelta
    days = td.days

    # Extract the number of whole hours remaining after the days are accounted for
    # 'seconds' attribute of timedelta holds the total number of seconds in the duration
    # 3600 seconds in an hour
    hours = td.seconds // 3600

    # Extract the number of whole minutes remaining after days and hours are accounted for
    # 60 seconds in a minute
    minutes = (td.seconds // 60) % 60

    # Format and return the duration as a string
    return f"{days} days {hours:02d} hours {minutes:02d} min"

def time_away_from_home(data_original, distances):
    """
    Calculate the total time spent away from home based on location data and distances.

    Args:
    data_original (DataFrame): A pandas DataFrame containing location data with 'tracked_at' and 'day_till_5am' columns.
    distances (array-like): An array-like object containing distances from home for each data point in 'data_original'.

    Returns:
    timedelta: Total duration spent away from home.
    """

    # Create a copy of the original DataFrame to avoid modifying the original data
    data = data_original.copy()

    # Add a new column 'away_from_home' indicating whether each location is more than 100 meters from home
    data['away_from_home'] = (distances > 25) & (distances < 30000)

    # Get unique days from the data to iterate over
    days = data['day_till_5am'].unique()

    # List to store durations of time spent away from home
    time_away_durations = []

    # Tracking variable for the start of a period away from home
    current_start = None

    # Iterate over each day
    for day in days:
        # Filter data for the current day
        temp_data = data[data['day_till_5am'] == day]

        # Iterate over rows in the day's data
        for index, row in temp_data.iterrows():
            if row['away_from_home']:
                # Mark the start of a new period away from home
                if current_start is None:
                    current_start = row['tracked_at']
            else:
                # End of a period away from home; calculate duration
                if current_start is not None:
                    duration = row['tracked_at'] - current_start
                    time_away_durations.append(duration)
                    current_start = None

        # Handle case where the last recorded point in a day is still away from home
        if current_start is not None:
            last_duration = temp_data['tracked_at'].iloc[-1] - current_start
            time_away_durations.append(last_duration)

    # Normalize all durations to timedelta objects
    time_away_durations = [timedelta(seconds=x.total_seconds()) if isinstance(x, pd.Timedelta) else timedelta(minutes=x) for x in time_away_durations]

    # Calculate the total time spent away from home across all days
    total_time_away = sum(time_away_durations, timedelta())

    return total_time_away

def calculate_time_away_from_home(data, home_coordinates, distance_threshold=25):
    time_away = 0
    away_from_home = False
    last_time_stamp = None

    for index, row in data[['tracked_at', 'latitude', 'longitude']].iterrows():
        current_location = (row['latitude'], row['longitude'])
        distance = haversine(current_location[0], current_location[1], home_coordinates[0], home_coordinates[1])

        if distance > distance_threshold:
            if not away_from_home:
                away_from_home = True
                last_time_stamp = row['tracked_at']
        else:
            if away_from_home:
                away_from_home = False
                time_away += (row['tracked_at'] - last_time_stamp).total_seconds()
                last_time_stamp = None

    return time_away

def get_country_from_coordinates(coords, api_key):
    """
    Function to get the country name from a specific set of latitude and longitude coordinates.

    Parameters:
    coords (list): A list containing two elements: latitude and longitude.
    api_key (str): Google Geocoding API key.

    Returns:
    str: The name of the country corresponding to the given coordinates.
    """

    lat, lon = coords
    url = f'https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={api_key}'
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json().get('results', [])
        for component in results[0]['address_components']:
            if 'country' in component['types']:
                return component['long_name']
    return None

def get_country_indices(country_name):
    """
    Function to return various indices for a given country in the DataFrame.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the data.
    country_name (str): The name of the country to search for.

    Returns:
    Tuple of floats: Quality of Life Index, Purchasing Power Index, Safety Index,
                     Health Care Index, Cost of Living Index, Property Price to Income Ratio,
                     Traffic Commute Time Index, Pollution Index, Climate Index
    """


    # Filter DataFrame for rows where 'Country' matches the country_name
    filtered_df = qol_countries[qol_countries['Country'] == country_name]

    # Check if any rows were found
    if not filtered_df.empty:
        # Extract the first row (assuming one row per country)
        row = filtered_df.iloc[0]

        # Abbreviations for the indices
        QoL = row['Quality of Life Index']
        PPI = row['Purchasing Power Index']
        SI = row['Safety Index']
        HCI = row['Health Care Index']
        CoLI = row['Cost of Living Index']
        PPtIR = row['Property Price to Income Ratio']
        TCTI = row['Traffic Commute Time Index']
        PI = row['Pollution Index']
        CI = row['Climate Index']

        return QoL, PPI, SI, HCI, CoLI, PPtIR, TCTI, PI, CI
    else:
        return None

def calculate_area_perimeter(df):
    """
    Calculate the area and perimeter of a shape defined by a series of latitude and longitude points.

    Args:
    df (DataFrame): A pandas DataFrame containing 'latitude' and 'longitude' columns.

    Returns:
    tuple: A tuple containing the area (in square kilometers) and the perimeter (in meters) of the shape.

    The perimeter is calculated using the haversine formula, which gives a great-circle distance between points.
    The area is calculated using a planar approximation, suitable for small areas (not accurate for large areas or near the poles).
    """

    n = len(df)
    # Check if there are enough points to form a shape
    if n < 3:
        return 0, 0  # Not enough points to form a shape

    # Perimeter Calculation
    # Sum the distances between successive points, wrapping around to the start
    perimeter = sum(haversine(df.iloc[i]['latitude'], df.iloc[i]['longitude'],
                              df.iloc[(i + 1) % n]['latitude'], df.iloc[(i + 1) % n]['longitude']) for i in range(n))

    # Area Calculation using the Shoelace formula (planar approximation)
    # This loop calculates the area based on latitude and longitude coordinates
    # The formula is a simple planar calculation and does not account for the Earth's curvature
    # Hence, it's not accurate for large areas or areas that are near the poles
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += df.iloc[i]['longitude'] * df.iloc[j]['latitude']
        area -= df.iloc[j]['longitude'] * df.iloc[i]['latitude']
    area = abs(area) * 111.32 / 2.0  # Conversion factor to approximate square kilometers
    perimeter = int(perimeter/1000)
    return area, perimeter

# Function to convert lat/long to UTM coordinates
def to_utm(lon, lat):
    proj_latlong = Proj(proj='latlong', datum='WGS84')
    proj_utm = Proj(proj="utm", zone=33, datum='WGS84')
    utm_x, utm_y = transform(proj_latlong, proj_utm, lon, lat)
    return utm_x, utm_y

def calculate_are_utm(df):
  utm_points = []
  for index, row in df.iterrows():
    utm_points.append(to_utm(row['latitude'], row['longitude']))
  polygon = Polygon(utm_points)
  area = polygon.area
  return area

def changes_in_place_api_dict(dict_1, divider):
    # Remove the key that is None
    if None in dict_1:
        del dict_1[None]
    if "Other" in dict_1:
        del dict_1["Other"]
    # Iterate over key-value pairs, handle NaN values and divide by divider
    categories = ['Transportation and Travel',	'Retail and Shopping',	'Entertainment and Food',	'Athletics',	'Professional and Public Services',	'Health and Wellness']
    for cat in categories:
      if cat in dict_1:
        dict_1[cat] = (dict_1[cat] / divider)*100
      else:
        dict_1[cat] = 0
    return dict_1

def randomize_location(lat, lon, radius=200):
    """
    Randomize location within a circle of specified radius.

    Parameters:
    lat (float): Latitude of the original location
    lon (float): Longitude of the original location
    radius (float): Radius in meters (default is 50 meters)

    Returns:
    (float, float): A tuple containing the randomized latitude and longitude
    """

    # Convert radius from meters to degrees
    radius_in_degrees = radius / 111320

    # Random angle
    angle = random.uniform(0, 2 * math.pi)

    # Random radius
    r = radius_in_degrees * math.sqrt(random.uniform(0, 1))

    # Calculate new coordinates
    new_lat = lat + r * math.cos(angle)
    new_lon = lon + r * math.sin(angle)

    return new_lat, new_lon