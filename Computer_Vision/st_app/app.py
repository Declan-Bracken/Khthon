import streamlit as st
from utils.geometry import generate_grid
from utils.search import grid_search_churches
from utils.constants import inference_time_lower, inference_time_upper
import json
import time


# Streamlit App
def main():
    st.title('Ethiopian Church Detection with YOLO and Google Maps')
    api_key = st.text_input('Enter your Google Maps API Key', type='password')
    # Merged latitude and longitude input
    location_input = st.text_input('Enter center latitude and longitude (comma-separated)', '9.743649316686131, 38.96981250757799')
    try:
        center_lat, center_lon = map(float, location_input.split(','))
    except ValueError:
        st.warning("Please enter valid latitude and longitude values separated by a comma.")

    total_area_km = st.number_input('Total Area (sq km)', value=1.0)
    # Generate grid of coordinates
    grid_coords = generate_grid(center_lat, center_lon, total_area_km, zoom = 18)
    num_images = len(grid_coords)
    time_eta_lowerbound = num_images * inference_time_lower + 0.5
    time_eta_upperbound = num_images * inference_time_upper + 1

    # Create columns to display metrics side by side
    col1, col2 = st.columns(2)
    # Displaying the image and time estimates side by side
    with col1:
        st.metric("Images to Process", num_images)
    with col2:
        st.metric("Estimated Time", f"{time_eta_lowerbound:.1f} - {time_eta_upperbound:.1f} seconds")

    confidence_threshold = st.slider('Confidence Threshold', 
                                    min_value=0.0, max_value=1.0, value=0.78, step=0.01)
    

    # Store results in session state to avoid rerunning the search
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None

    if st.button('Start Grid Search'):
        with st.spinner('Performing grid search...'):
            start = time.time()
            total_churches_detected, church_info = grid_search_churches(grid_coords, api_key=api_key,
                                                                        confidence_threshold = confidence_threshold)
            end = time.time()
            st.session_state.search_results = (total_churches_detected, church_info)
            st.success(f"Found {total_churches_detected} church over {end - start} seconds.")
    # Display results only if they are available
    if st.session_state.search_results:
        total_churches_detected, church_info = st.session_state.search_results

        if total_churches_detected > 0:
            image_keys = church_info.keys()
            # Create a dropdown menu to select a detection
            selected_image = st.selectbox(
                'Select an image with detections:',
                options=[f"{key[0]}, {key[1]}" for key in image_keys]
            )

            # Get the selected detection's index
            delimited_coord = selected_image.split(",")
            image_coords = (float(delimited_coord[0].strip()), float(delimited_coord[1].strip()))
            image_data = church_info[image_coords]

            # Display the selected detection's image
            st.image(image_data[0], caption=f"Detection(s) at {selected_image}", use_column_width=True)
            # Use an accordion for detection details
            with st.expander("Detection Details"):
                for i in range(len(church_info[image_coords][3])):
                    st.markdown(f"**Detection {i}:**")
                    st.write(f"Estimated Coordinates: `{image_data[2][i]}`")
                    st.write(f"Confidence: `{image_data[3][i]:.2f}`")
                    st.write(f"Class: `{image_data[4][i]}`")

            # json_dict = {", ".join(str(val) for val in key): church_info[key] for key in church_info}

            # st.download_button(
            #     label = "Download JSON file of all detections.",
            #     data = json.dumps(json_dict),
            #     file_name = "Church_detections.json"
            # )

        else:
            st.info("No churches were detected in the specified area.")

if __name__ == "__main__":
    main()
