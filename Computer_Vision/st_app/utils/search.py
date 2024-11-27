from utils.geometry import generate_grid, bbox_to_coords
from utils.model_handler import YOLOModel
from utils.fetch import fetch_image
from utils.constants import MODEL_PATH
import streamlit as st

@st.cache_resource
def load_model(model_path):
    return YOLOModel(model_path)

# Main function to perform grid search for churches
def grid_search_churches(center_lat, center_lon, total_area_km=1, zoom=18, api_key=None, confidence_threshold = 0.78):
    """
    Create a grid and call the static api from google maps to serve images at locations in the groid_coords list.
    For each image, pass through the model and collect detections in the church_info list. Returns the total number
    of detections along with the detection info, including the latitude, longitude, class, name, image, and confidence
    of each detection.
    """
    assert api_key is not None, 'The Google Maps Static API key cannot be None.'
    grid_coords = generate_grid(center_lat, center_lon, total_area_km, zoom)
    total_detections = 0
    church_info = []

    # Load in the cached model
    model = load_model(MODEL_PATH)
    
    # Loop through all coordinates in the grid and for each, pull the image and then run through model.
    for coord in grid_coords:
        image = fetch_image(coord[0], coord[1], zoom=zoom, api_key=api_key)
        boxes, class_names, annotated_image = model.detect_objects(image, confidence_threshold=confidence_threshold)
        total_detections += len(boxes)

        # If we find that the boxes aren't empty, it means we have at least one detection. For each detection, store it's properties.
        for bbox in boxes:
            church_lat, church_lon = bbox_to_coords(bbox, coord)
            confidence = bbox.conf.item()
            church_class = class_names[int(bbox.cls)]
            church_name = bbox.cls_name if 'cls_name' in bbox else "Unknown Church"
            church_info.append((church_lat, church_lon, church_class, church_name, annotated_image, confidence))

    return total_detections, church_info
