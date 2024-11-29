from utils.geometry import bbox_to_coords
from utils.model_handler import YOLOModel
from utils.fetch import fetch_image_batch
from utils.constants import MODEL_PATH
import asyncio
import streamlit as st
import time

@st.cache_resource
def load_model(model_path):
    return YOLOModel(model_path)

# Main function to perform grid search for churches
def grid_search_churches(grid_coords, zoom=18, size = (640, 640), api_key=None, confidence_threshold = 0.78):
    """
    Create a grid and call the static api from google maps to serve images at locations in the groid_coords list.
    For each image, pass through the model and collect detections in the church_info list. Returns the total number
    of detections along with the detection info, including the latitude, longitude, class, name, image, and confidence
    of each detection.
    """
    assert api_key is not None, 'The Google Maps Static API key cannot be None.'

    total_detections = 0
    church_info = {}

    # Load in the cached model
    model = load_model(MODEL_PATH)
    start = time.time()
    # Call Maps API to fetch images at specific grid coordinates. Return list of PIL Image objects.
    images = asyncio.run(fetch_image_batch(grid_coords, zoom = zoom, size = size, api_key = api_key))
    end = time.time()
    print(f"Time to pull all images = {end - start}")
    # Run inference over all images using pretrained model
    results = model.detect_batch_objects(images, confidence_threshold=confidence_threshold)

    # Populate detections dictionary
    for i in range(len(results)):
        # Get inference results and coordinates for current image
        result = results[i]
        image_coords = grid_coords[i]
        boxes, classes = result.boxes, result.names

        # If there were detections, add data to output dictionary
        if boxes:
            num_detections = len(boxes)
            total_detections += num_detections
            # Initialize arrays to hold data on individual detections
            detection_coords, confidences, class_names, class_indices = [], [], [], []

            # Loop through detections from current image and populate arrays.
            for bbox in boxes:
                detection_coords.append(bbox_to_coords(bbox, image_coords))
                confidences.append(bbox.conf.item())
                class_indices.append(classes[int(bbox.cls)])
                class_names.append(bbox.cls_name if 'cls_name' in bbox else "Church")
                # church_info.append((church_lat, church_lon, class_idx, class_name, annotated_image, confidence))
            
            annotated_image = result.plot()
            image_data = [annotated_image, num_detections, detection_coords, confidences, class_indices, class_names]
            church_info[image_coords] = image_data

    return total_detections, church_info
