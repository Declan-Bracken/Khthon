import numpy as np
from utils.constants import zoom_to_length, kilometers_per_latitude

# Function to generate grid coordinates within a specified area
def generate_grid(center_lat, center_lon, area_km, zoom):
    """
    Generates a list of central coordinates for the maps api to take images from.
    Input:
        center_lat: Float, central latitude of the area over which we want to grid search
        center_lon: Float, central longitude of the area over which we want to grid search
        zoom: Int, Zoom parameter for maps static api. Typically 18 (corresponding to roughly 0.36 squarred kilometers)
        img_size: The area estimate for each individual image we will take from google maps. Computer as an estimate from zoom.

    Output:
        grid_ccords: List: List where each element is a tuple of size 2 like (latitude, longitude).
    
    """
    img_size = zoom_to_length[zoom]

    lat_range = area_km / kilometers_per_latitude
    lon_range = area_km / (kilometers_per_latitude * np.cos(center_lat * np.pi / 180))

    lat_img = img_size / kilometers_per_latitude
    lon_img = img_size / (kilometers_per_latitude * np.cos(center_lat * np.pi / 180))

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
    kilometer_per_image = zoom_to_length[zoom] # get corresponding image dimension in kilometers from zoom

    lat_per_pixel = kilometer_per_image / kilometers_per_latitude / image_size[0]
    lon_per_pixel = kilometer_per_image / (kilometers_per_latitude * np.cos(lat * np.pi / 180)) / image_size[1]

    left, top, right, bottom = bbox.xyxy[0].tolist()

    box_center_height = (top + bottom) / 2
    box_center_width = (left + right) / 2

    lat_relative_to_image_center = -(box_center_height * lat_per_pixel - kilometer_per_image/2 / kilometers_per_latitude)
    lon_relative_to_image_center = box_center_width * lon_per_pixel - kilometer_per_image/2 / (kilometers_per_latitude * np.cos(lat * np.pi / 180))

    church_lat = lat + lat_relative_to_image_center
    church_lon = lon + lon_relative_to_image_center

    return (float(church_lat), float(church_lon))
