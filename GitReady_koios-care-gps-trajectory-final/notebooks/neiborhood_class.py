import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import json

def green_persentage_in_area(home, API_KEY = 'AIzaSyAZrllCfkVCImS3m2MwbdXOlH4ddU42H24'):
    
    latitude, longitude = home
    # Determine ZOOM for ~500m coverage based on latitude
    if abs(latitude) <= 30:
        zoom = 15
    elif abs(latitude) <= 60:
        zoom = 16
    elif abs(latitude) <= 80:
        zoom = 17
    else:
        zoom = 18  # For latitudes closer to the poles
    
    size = '640x640'  # Image size (max for scale = 2, capped detail-quality for our API capabilities)
    
    # Construct URL
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&size={size}&scale=2&maptype=satellite&key={API_KEY}"
    
    # Fetch the image
    response = requests.get(url)
    
    # Save the image
    if response.status_code == 200:
    # CHANGE SAVE PATH TO DESIRED LOCATION
        with open(r'../data/images/satellite_image.png', 'wb') as file:
            file.write(response.content)

    image_path = r'../data/images/satellite_image.png'
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define a wider range of green color in HSV
    lower_green = np.array([30, 50, 50])  # Adjust these values for a wider range
    upper_green = np.array([90, 255, 255])  # Adjust these values for a wider range
    
    # Define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])  # Adjust these values
    upper_blue = np.array([130, 255, 255])  # Adjust these values
    
    # Create masks for green and blue colors
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    # Calculate the green and blue areas
    green_area = np.sum(green_mask == 255)
    blue_area = np.sum(blue_mask == 255)
    
    # Calculate the total area
    total_area = image.shape[0] * image.shape[1]
    
    # Calculate the percentages of green and blue areas
    green_percentage = (green_area / total_area) * 100
    blue_percentage = (blue_area / total_area) * 100
    other_percentage = 100 - green_percentage - blue_percentage

    return green_percentage, blue_percentage
    

    
    
    #PIE CHART
# labels = 'Environment', 'Water', 'Others'
# sizes = [green_percentage, blue_percentage, other_percentage]
# colors = ['green', 'blue', 'gray']
# explode = (0.1, 0.1, 0)  # explode the first two slices

# plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.show()

# # HISTOGRAM
# plt.bar(['Environment', 'Water', 'Others'], [green_percentage, blue_percentage, other_percentage], color=['green', 'blue', 'gray'])
# plt.ylabel('Percentage (%)')
# plt.title('Area Distribution')
# plt.show()

# # HISTOGRAM RGB
# color = ('b', 'g', 'r')
# for i, col in enumerate(color):
#     hist = cv2.calcHist([image], [i], None, [256], [0, 256])
#     plt.plot(hist, color=col)
#     plt.xlim([0, 256])

# plt.title('RGB Color Distribution')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')
# plt.show()