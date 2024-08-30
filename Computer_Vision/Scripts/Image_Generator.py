import csv
import requests
from PIL import Image
from io import BytesIO
import random
import math
import os
import shutil

class SatelliteImageRetriever:
    def __init__(self, csv_file_path, api_key):
        self.csv_file_path = csv_file_path
        self.api_key = api_key
        self.coordinates = self.load_coordinates()

    def load_coordinates(self):
        coordinates = []
        with open(self.csv_file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                name, lat, lon = row
                coordinates.append((name, float(lat), float(lon)))
        return coordinates

    def fetch_image(self, latitude, longitude, zoom=18, size=(640, 640), maptype='satellite'):
        url = f"https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&size={size[0]}x{size[1]}&maptype={maptype}&key={self.api_key}"
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        return image
    
    def compute_translation_range(self, latitude, area_km2 = 0.36): # Where area is in squared kilometers.
        # Approximate side length of the square area in kilometers
        side_length_km = math.sqrt(area_km2)
        
        # Convert side length to meters (1 km = 1000 m)
        side_length_m = side_length_km * 1000
        
        # Half the side length to get the translation distance from the center to the edge
        half_side_length_m = side_length_m / 2
        
        # Constants for converting degrees to kilometers
        km_per_degree_lat = 111.0
        km_per_degree_lon = 111.0 * math.cos(math.radians(latitude))
        
        # Calculate the translation in degrees
        lat_translation = half_side_length_m / 1000 / km_per_degree_lat
        lon_translation = half_side_length_m / 1000 / km_per_degree_lon

        return lat_translation, lon_translation


    def translate_coordinates(self, latitude, longitude, min_factor = 0.1, max_factor = 0.6):
        """ 
        Max factor corresponds to the maximum distance from the central coordinate that the translated image will be taken. A factor of 1 means that at most the new image will be translated so that the central coordinates are roughly at it's edges. A factor of 0 means there will be no translation at all.
        Min factor corresponds to the minimum distance from the central coordinate that the translated image will be taken.
        """
        lat_translation, lon_translation = self.compute_translation_range(latitude)

        # Take random smaple within the range
        translated_latitude = latitude + random.uniform(lat_translation*min_factor, lat_translation*max_factor) * random.choice([1,-1])
        translated_longitude = longitude + random.uniform(lon_translation*min_factor, lon_translation*max_factor) * random.choice([1,-1])
        
        return translated_latitude, translated_longitude

    def retrieve_images(self, output_dir, translations=3):
        for name, lat, lon in self.coordinates:
            # Strip name of '/':
            name = name.replace('/','')
            # Fetch the centered image
            centered_image = self.fetch_image(lat, lon)
            centered_image.save(f"{output_dir}/{name}_centered.png")

            # Fetch translated images
            for i in range(translations):
                translated_lat, translated_lon = self.translate_coordinates(lat, lon)
                translated_image = self.fetch_image(translated_lat, translated_lon)
                translated_image.save(f"{output_dir}/{name}_translated_{i}.png")
    
    def split_data(self, source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        # Ensure the ratios sum to 1
        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"

        # Create output directories if they don't exist
        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'validation')
        test_dir = os.path.join(output_dir, 'test')
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Group images by name (without suffixes)
        image_groups = {}
        
        for filename in os.listdir(source_dir):
            if not filename.endswith(".png"):  # Adjust this if using different file types
                continue
            name = filename.split('_')[0]  # Extract the base name (before the first underscore)
            if name not in image_groups:
                image_groups[name] = []
            image_groups[name].append(filename)
        
        # Get all unique base names
        all_names = list(image_groups.keys())
        
        # Shuffle the names
        random.shuffle(all_names)
        
        # Determine the split indices
        total = len(all_names)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # Split the names into training, validation, and test sets
        train_names = all_names[:train_end]
        val_names = all_names[train_end:val_end]
        test_names = all_names[val_end:]
        
        # Move the files to the corresponding directories
        def move_files(names, target_dir):
            for name in names:
                for filename in image_groups[name]:
                    src = os.path.join(source_dir, filename)
                    dst = os.path.join(target_dir, filename)
                    shutil.move(src, dst)
        
        move_files(train_names, train_dir)
        move_files(val_names, val_dir)
        move_files(test_names, test_dir)
        
        print(f"Data split completed. Training: {len(train_names)} groups, Validation: {len(val_names)} groups, Test: {len(test_names)} groups.")


if __name__ == "__main__":
    # Open the api key
    with open('code/Computer_Vision/Scripts/api_key.txt', 'r') as file:
        api_key = file.read()

    # Usage
    csv_file_path = 'location_lists/location_list_v1.csv'
    output_dir = 'images'

    retriever = SatelliteImageRetriever(csv_file_path, api_key)
    # retriever.retrieve_images(output_dir)
    retriever.split_data(output_dir,output_dir)
