import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# Dictionary to map zoom levels to area sizes (in kilometers) covered by the image
zoom_to_area = {
    18: 0.33,  # Example: Zoom level 18 covers ~0.36 square km
    19: 0.18,  # Zoom level 19 covers ~0.18 square km
    # Add more zoom levels if needed
}

# Function to fetch a satellite image from Google Maps API
def fetch_image(latitude, longitude, zoom=18, size=(640, 640), maptype='satellite', api_key=None):
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&size={size[0]}x{size[1]}&maptype={maptype}&key={api_key}"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

# Function to generate grid coordinates within a specified area
def generate_grid(center_lat, center_lon, area_km, img_size=0.36):
    lat_range = area_km / 111.32
    lon_range = area_km / (111.32 * np.cos(center_lat * np.pi / 180))

    lat_img = img_size / 111.32
    lon_img = img_size / (111.32 * np.cos(center_lat * np.pi / 180))

    latitudes = np.arange(center_lat - lat_range/2, center_lat + lat_range/2, lat_img)
    longitudes = np.arange(center_lon - lon_range/2, center_lon + lon_range/2, lon_img)

    grid_coords = [(lat, lon) for lat in latitudes for lon in longitudes]
    return grid_coords

def bbox_to_coords(bbox, grid_coord, image_size=(640, 640), zoom=18):
    lat, lon = grid_coord
    area = zoom_to_area[zoom] # get corresponding area in sq km from zoom

    lat_per_pixel = area / 111.32 / image_size[0]
    lon_per_pixel = area / (111.32 * np.cos(lat * np.pi / 180)) / image_size[1]

    left, top, right, bottom = bbox.xyxy[0].tolist()

    church_lat = lat + (top + bottom) / 2 * lat_per_pixel - area/2 / 111.32
    church_lon = lon + (left + right) / 2 * lon_per_pixel - area/2 / (111.32 * np.cos(lat * np.pi / 180))

    return float(church_lat), float(church_lon)

# Function to preprocess images and perform inference using YOLO model
def detect_churches(image, model, confidence_threshold = 0.78):
    results = model.predict(image, conf = confidence_threshold)
    annotated_image = results[0].plot()  # YOLO method to get image with bounding boxes
    return results[0].boxes, results[0].names, annotated_image  # Return bounding boxes, names, and annotated image

# Main function to perform grid search for churches
def grid_search_churches(center_lat, center_lon, total_area_km=1, zoom=18, api_key=None, model_path='code/Computer_Vision/models/v2/best.pt', confidence_threshold = 0.78):
    assert api_key is not None, 'The Google Maps Static API key cannot be None.'

    img_size = zoom_to_area[zoom]
    grid_coords = generate_grid(center_lat, center_lon, total_area_km, img_size=img_size)

    model = YOLO(model_path)
    total_detections = 0
    church_info = []

    for coord in grid_coords:
        image = fetch_image(coord[0], coord[1], zoom=zoom, api_key=api_key)
        boxes, class_names, annotated_image = detect_churches(image, model, confidence_threshold=confidence_threshold)
        total_detections += len(boxes)

        for bbox in boxes:
            church_lat, church_lon = bbox_to_coords(bbox, coord)
            confidence = bbox.conf.item()
            church_class = class_names[int(bbox.cls)]
            church_name = bbox.cls_name if 'cls_name' in bbox else "Unknown Church"
            church_info.append((church_lat, church_lon, church_class, church_name, annotated_image, confidence))

    return total_detections, church_info

# Streamlit App
st.title('Ethiopian Church Detection with YOLO and Google Maps')

api_key = st.text_input('Enter your Google Maps API Key', type='password')

center_lat = st.number_input('Center Latitude', value=14.16107, format="%.5f")
center_lon = st.number_input('Center Longitude', value=38.95375, format="%.5f")
total_area_km = st.number_input('Total Area (sq km)', value=1.0)
# zoom = st.slider('Zoom Level (18 recommended)', min_value=18, max_value=19, value=18)
confidence_threshold = st.slider('Confidence Threshold', min_value=0.0, max_value=1.0, value=0.78, step=0.01)

model_path = st.text_input('Enter YOLO Model Path', value='code/Computer_Vision/models/v2/best.pt')

# Store results in session state to avoid rerunning the search
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

if st.button('Start Grid Search'):
    with st.spinner('Performing grid search...'):
        total_churches_detected, church_info = grid_search_churches(center_lat, center_lon, 
                                                                    total_area_km=total_area_km, api_key=api_key, 
                                                                    model_path=model_path, 
                                                                    confidence_threshold = confidence_threshold)
        st.session_state.search_results = (total_churches_detected, church_info)
        st.success(f"Total number of churches detected in the {total_area_km} sq km area: {total_churches_detected}")

# Display results only if they are available
if st.session_state.search_results:
    total_churches_detected, church_info = st.session_state.search_results

    if total_churches_detected > 0:
        # Create a dropdown menu to select a detection
        selected_church = st.selectbox(
            'Select a detection to view',
            options=[f"Detection {i+1} - {info[2]} at ({info[0]}, {info[1]})" for i, info in enumerate(church_info)]
        )

        # Get the selected detection's index
        selected_index = [f"Detection {i+1} - {info[2]} at ({info[0]}, {info[1]})" for i, info in enumerate(church_info)].index(selected_church)
        # Extract the confidence score for the selected detection
        confidence_score = church_info[selected_index][5]
        # Display the selected detection's image
        st.image(church_info[selected_index][4], caption=f"Detection at ({church_info[selected_index][0]}, {church_info[selected_index][1]}), Confidence = {confidence_score}", use_column_width=True)
    else:
        st.info("No churches were detected in the specified area.")