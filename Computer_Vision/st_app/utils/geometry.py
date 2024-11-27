import numpy as np
from utils.constants import zoom_to_area

# Function to generate grid coordinates within a specified area
def generate_grid(center_lat, center_lon, area_km, zoom):
    """
    Generates a 2d list of central coordinates for the maps api to take images from.
    Input:
        center_lat: Float, central latitude of the area over which we want to grid search
        center_lon: Float, central longitude of the area over which we want to grid search
        zoom: Int, Zoom parameter for maps static api. Typically 18 (corresponding to roughly 0.36 squarred kilometers)
        img_size: The area estimate for each individual image we will take from google maps. Computer as an estimate from zoom.

    Output:
        grid_ccords: List: 2D list where each element is a tuple of size 2 like (latitude, longitude).
    
    """
    img_size = zoom_to_area[zoom]
    
    lat_range = area_km / 111.32
    lon_range = area_km / (111.32 * np.cos(center_lat * np.pi / 180))

    lat_img = img_size / 111.32
    lon_img = img_size / (111.32 * np.cos(center_lat * np.pi / 180))

    latitudes = np.arange(center_lat - lat_range/2, center_lat + lat_range/2, lat_img)
    longitudes = np.arange(center_lon - lon_range/2, center_lon + lon_range/2, lon_img)

    grid_coords = [(lat, lon) for lat in latitudes for lon in longitudes]
    return grid_coords


def bbox_to_coords(bbox, grid_coord, image_size=(640, 640), zoom=18):
    """
    Function which takes in a bbox dictionary from a YOLO prediction, along with the central grid coordinates for
    the image from which the bounding box(s) was detected, and estimates the exact coordinate centered on the detected
    church through linear interpolation.

    Inputs:
        bbox: Dictionary, 
    """
    lat, lon = grid_coord
    area = zoom_to_area[zoom] # get corresponding area in sq km from zoom

    lat_per_pixel = area / 111.32 / image_size[0]
    lon_per_pixel = area / (111.32 * np.cos(lat * np.pi / 180)) / image_size[1]

    left, top, right, bottom = bbox.xyxy[0].tolist()

    church_lat = lat + (top + bottom) / 2 * lat_per_pixel - area/2 / 111.32
    church_lon = lon + (left + right) / 2 * lon_per_pixel - area/2 / (111.32 * np.cos(lat * np.pi / 180))

    return float(church_lat), float(church_lon)
