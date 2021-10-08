import datetime
import io
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from PIL import Image
from numpy import array
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session


from tkinter import ttk, Tk, Button, Frame, Canvas, BOTH, LEFT, VERTICAL, RIGHT, X, Y, Listbox, END, Label

# Your client credentials
client_id = '982de4f4-dade-4f98-9b49-4374cd896bb6'
client_secret = '%p/,0Yrd&/mO%cdudUsby[>@]MB|2<rf1<NnXkZr'

# Create a sessionv
client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)

# Get token for the session
token = oauth.fetch_token(token_url='https://services.sentinel-hub.com/oauth/token',
                          client_id=client_id, client_secret=client_secret)



# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#             POST REQUEST FUNCTION
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def get_response (bbox, date_from_str, date_to_str, s_product, dimension, minQa):

    if s_product == "NO2":
        product_evalscritp = """
                    //VERSION=3
                    function setup() {
                        return {
                            input: ["NO2", "dataMask"],
                                output: { bands:  4},
                            }
                        }                                    
                    const minVal = 0.0
                    const maxVal = 0.0001
                    const diff = maxVal - minVal
                    const rainbowColors = [
                        [minVal, [0, 0, 0.5]],
                        [minVal + 0.125 * diff, [0, 0, 1]],
                        [minVal + 0.375 * diff, [0, 1, 1]],
                        [minVal + 0.625 * diff, [1, 1, 0]],
                        [minVal + 0.875 * diff, [1, 0, 0]],
                        [maxVal, [0.5, 0, 0]]
                    ]
                    const viz = new ColorRampVisualizer(rainbowColors)
                    function evaluatePixel(sample) {
                        var rgba= viz.process(sample.NO2)
                        rgba.push(sample.dataMask)
                        return rgba
                    }
                    """

    response = oauth.post('https://creodias.sentinel-hub.com/api/v1/process',
        json={
          "input": {
              "bounds": {
                  "properties": {
                      "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                  },
                  "bbox": bbox
              },
              "data": [
                  {
                      "type": "sentinel-5p-l2",
                      "dataFilter": {
                          "timeRange": {
                              "from": date_from_str[0] + "T" + date_from_str[1] + "Z",
                              "to": date_to_str[0] + "T" + date_to_str[1] + "Z",
                          }
                      },
                      "processing": minQa
                  }
              ]
          },
          "output": dimension,
          "evalscript": product_evalscritp
        })
    return response




def get_new_coordinates(lat, lon, distance_lat, distance_lon):
    lat_new = lat + (180 / math.pi) * (distance_lat / 6378137)
    lon_new = lon + (180 / math.pi) * (distance_lon / 6378137) / math.cos(math.pi/180*lat)
    return [lat_new, lon_new]

def get_new_longitude(lat, lon, distance_lon):
    return get_new_coordinates(lat, lon, 0, distance_lon)[1]

def get_new_latitude(lat, lon, distance_lat):
    return get_new_coordinates(lat, lon, distance_lat, 0)[0]

def get_bbox_coordinates_from_center(coordinates, distance):
    bbox_coordinates = []
    lat = coordinates[0]
    lon = coordinates[1]
    bbox_coordinates.append(get_new_longitude(lat, lon, -distance))
    bbox_coordinates.append(get_new_latitude(lat, lon, -distance))
    bbox_coordinates.append(get_new_longitude(lat, lon, distance))
    bbox_coordinates.append(get_new_latitude(lat, lon, distance))
    return bbox_coordinates




def save_image(image, product_type, location_name):
    directory_path = "../data/" + product_type + "/" + location_name + "/images/"\
                     + image["date"].strftime("%Y") + "/" + image["date"].strftime("%m") + "/"
    file_name = image["date"].strftime("%d") + "-" + str(image["time"])
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    image["image"].save(directory_path +  file_name + ".png", format="png")


def get_hourly_images(product_type, location_name, minQa_info, date_start):
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #                   PARAMETERS
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # setting coordinates
    distance_radius = 100000
    if location_name == "Bering Strait":
        bbox_coordinates = [-170.844, 65.396, -167.622, 66.230]
        range_h = 5
    if location_name == "Sabetta Port":
        sabetta_port_coordinates = [71.264765, 72.060155]
        bbox_coordinates = get_bbox_coordinates_from_center(sabetta_port_coordinates, distance_radius)
        range_h = 10.5
    # setting image dimension
    precision = 2000 # meters per pixel
    img_length = int(distance_radius*2/precision)
    dimension = {"width": img_length, "height": img_length}
    # min data quality
    if minQa_info == "high": minQa = {}
    else: minQa = {"minQa": 0}

    date_from_h = date_start
    if date_start.year == 2019: date_from_h = date_start + datetime.timedelta(hours=23)
    if date_start.year == 2020: date_from_h = date_start + datetime.timedelta(hours=23)
    if date_start.year == 2021: date_from_h = date_start + datetime.timedelta(hours=23)
    date_to_h = date_from_h
    for hours_counter in range(int(range_h / 1.5)):
        date_from_h = date_to_h
        date_to_h = date_to_h + datetime.timedelta(hours=1.5)
        date_from_str = [date_from_h.strftime("%Y-%m-%d"), date_from_h.strftime("%H:%M:%S")]
        date_to_str = [date_to_h.strftime("%Y-%m-%d"), date_to_h.strftime("%H:%M:%S")]
        print(date_from_str)
        # getting post responce
        response = get_response(bbox_coordinates, date_from_str, date_to_str, product_type, dimension, minQa)
        in_memory_file = io.BytesIO(response.content)
        img_png = Image.open(in_memory_file)
        # pushin the image in the list
        image = {
            "date": date_start,
            "time": hours_counter,
            "image": img_png
        }
        """fig = plt.figure()
        plt.imshow(img_png)
        plt.show()"""
        save_image(image, product_type, location_name)




def rename(product_type, location_name, date_start):
    if location_name == "Bering Strait":
        range_h = 5
    if location_name == "Sabetta Port":
        range_h = 10.5
    date_from_h = date_start
    if date_start.year == 2019: date_from_h = date_start + datetime.timedelta(hours=23)
    if date_start.year == 2020: date_from_h = date_start + datetime.timedelta(hours=23)
    if date_start.year == 2021: date_from_h = date_start + datetime.timedelta(hours=23)
    date_to_h = date_from_h
    for hours_counter in range(int(range_h / 1.5)):
        date_from_h = date_to_h
        date_to_h = date_to_h + datetime.timedelta(hours=1.5)
        date_from_str = [date_from_h.strftime("%Y-%m-%d"), date_from_h.strftime("%H:%M:%S")]
        date_to_str = [date_to_h.strftime("%Y-%m-%d"), date_to_h.strftime("%H:%M:%S")]
        # pushin the image in the list
        image = {
            "date": date_start,
            "time": hours_counter,
        }
        directory_path = "../data/" + product_type + "2/" + location_name + "/images/"\
                     + image["date"].strftime("%Y") + "/" + image["date"].strftime("%m") + "/"
        new_directory_path = "../data/" + product_type + "/" + location_name + "/images/" \
                         + image["date"].strftime("%Y") + "/" + image["date"].strftime("%m") + "/"
        file_name = image["date"].strftime("%d") + str(image["time"])
        my_file = Path(directory_path + file_name + ".png")
        if my_file.is_file():
            image = Image.open(directory_path +  file_name + ".png")
            Path(new_directory_path).mkdir(parents=True, exist_ok=True)
            image.save(new_directory_path + file_name[0] + file_name[1] + "-" + file_name[2] + ".png", format="png")










# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                       MAIN
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

values = {
    "product_types": ["NO2", "CO", "CH4", "SO2"],
    "locations_name": ["Bering Strait", "Sabetta Port"],
    "minQas": ["high", "all"]
}
date = datetime.datetime.now()
date_start = date.replace(year=2021, month=5, day=15, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=10, day=1, hour=0, minute=0, second=0, microsecond=0)

product_type = values["product_types"][0]
location_name = values["locations_name"][1]
minQa = values["minQas"][1]


# FOR IMAGES DOWNLOAD
#download_images(product_type, location_name, minQa, date_start, date_end)

for day_counter in range(int((date_end - date_start).days)):
    date = date_start + datetime.timedelta(days=day_counter)
    #get_hourly_images(product_type, location_name, minQa, date)
    rename(product_type, location_name, date)
