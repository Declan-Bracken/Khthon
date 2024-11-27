# import requests
from PIL import Image
from io import BytesIO
import asyncio
import aiohttp

# Function to fetch a satellite image from Google Maps API
async def fetch_image(latitude, longitude, zoom=18, size=(640, 640), maptype='satellite', api_key=None):
    """
    Given a latitude, longitude, zoom, area, and api key, access the google maps static api and return the served image.
    """

    url = f"https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&size={size[0]}x{size[1]}&maptype={maptype}&key={api_key}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            # response = requests.get(url)
            content = await response.read()
            image = Image.open(BytesIO(content))
    return image
