import streamlit as st
from utils.search import grid_search_churches

# Streamlit App
st.title('Ethiopian Church Detection with YOLO and Google Maps')
api_key = st.text_input('Enter your Google Maps API Key', type='password')
center_lat = st.number_input('Center Latitude', value=14.16107, format="%.5f") #    Example latitude and longitude
center_lon = st.number_input('Center Longitude', value=38.95375, format="%.5f")
total_area_km = st.number_input('Total Area (sq km)', value=1.0)
confidence_threshold = st.slider('Confidence Threshold', 
                                 min_value=0.0, max_value=1.0, value=0.14, step=0.01)
model_path = st.text_input('Enter YOLO Model Path', value='Computer_Vision/models/v2/best.pt')

# Store results in session state to avoid rerunning the search
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

if st.button('Start Grid Search'):
    with st.spinner('Performing grid search...'):
        total_churches_detected, church_info = grid_search_churches(center_lat, center_lon, 
                                                                    total_area_km=total_area_km, api_key=api_key,
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
