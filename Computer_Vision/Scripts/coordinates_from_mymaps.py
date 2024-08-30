import zipfile
import xml.etree.ElementTree as ET
import csv

"""
Image Generator Class takes in a kmz file path and returns a list of location names and 
their coordinates in the form [(name, float(lat), float(lon))].
"""

class MyMapsRetriever():
    def __init__(self, kmz_file_path):
        self.kmz_file_path = kmz_file_path
    
    # Function reads kml file into memopry given a location
    def extract_kml_from_kmz(self, kmz_file):
        with zipfile.ZipFile(kmz_file, 'r') as kmz:
            for file in kmz.namelist():
                if file.endswith('.kml'):
                    with kmz.open(file) as kml_file:
                        return kml_file.read()

    # Parse KML data
    def parse_kml(self, kml_data):
        """
        parse_kml function uses xml.etree.ElementTree to parse the XML content of the .kml file.
        It navigates the XML tree to find all Placemark elements.
        For each Placemark, it extracts the name (if available) and the coordinates.
        Coordinates are typically in the format longitude,latitude,altitude. The function splits this string and converts the longitude and latitude to floating-point numbers.
        """

        namespace = {"kml": "http://www.opengis.net/kml/2.2"}
        root = ET.fromstring(kml_data)
        coordinates = []
        
        for placemark in root.findall(".//kml:Placemark", namespace):
            name = placemark.find("kml:name", namespace).text if placemark.find("kml:name", namespace) is not None else "Unnamed"
            coord_text = placemark.find(".//kml:coordinates", namespace)
            if coord_text is not None:
                coord_text = coord_text.text.strip()
                for coord in coord_text.split():
                    lon, lat, _ = coord.split(",")
                    coordinates.append((name, float(lat), float(lon)))
        
        return coordinates
    
    # Apply the first 2 functions to retrieve the locations
    def retrieve_locations_list(self):
        kml_data = self.extract_kml_from_kmz(self.kmz_file_path)
        coordinate_list = self.parse_kml(kml_data)
        return coordinate_list

    # Write the list to a csv file
    def write_coordinates_to_csv(self, csv_file_path):
        coordinates = self.retrieve_locations_list()
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Latitude', 'Longitude'])  # Write header
            for name, lat, lon in coordinates:
                writer.writerow([name, lat, lon])
        

if __name__ == "__main__":
    kmz_file_path = '/Users/declanbracken/Downloads/Monestaries.kmz'
    csv_file_path = 'location_lists/location_list_v1.csv'
    coordinate_retriever = MyMapsRetriever(kmz_file_path)
    list = coordinate_retriever.retrieve_locations_list()
    coordinate_retriever.write_coordinates_to_csv(csv_file_path)


