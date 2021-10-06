import datetime
import io
import json
import math

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
    if s_product == "CH4":
        product_evalscritp = """
                    //VERSION=3
                    function setup() {
                        return {
                            input: ["CH4", "dataMask"],
                                output: { bands:  4}
                            }
                        }                                    
                    const minVal = 1600.0
                    const maxVal = 2000.0
                    const diff = maxVal - minVal
                    const rainbowColors = [
                        [minVal, [1, 1, 1]],
                        [maxVal, [0, 0, 0]],
                    ]
                    const viz = new ColorRampVisualizer(rainbowColors)
                    function evaluatePixel(sample) {
                        var rgba= viz.process(sample.CH4)
                        rgba.push(sample.dataMask)
                        return rgba
                    }
                    """
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
    if s_product == "CO":
        product_evalscritp = """
                    //VERSION=3
                    function setup() {
                        return {
                            input: ["CO", "dataMask"],
                                output: { bands:  4}
                            }
                        }                                    
                    const minVal = 0.0
                    const maxVal = 0.1
                    const diff = maxVal - minVal
                    const rainbowColors = [
                        [minVal, [1, 1, 1]],
                        [maxVal, [0, 0, 0]],
                    ]
                    const viz = new ColorRampVisualizer(rainbowColors)
                    function evaluatePixel(sample) {
                        var rgba= viz.process(sample.CO)
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







# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#               RGB to VALUE
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def get_min_max_val(product_type):
    if product_type == "NO2":
        minVal = 0.0
        maxVal = 0.0001
    if product_type == "CO":
        minVal = 0.0
        maxVal = 0.1
    if product_type == "CH4":
        minVal = 1600.0
        maxVal = 2000.0
    return [minVal, maxVal]

def get_rainbow_value( rgb, precision ):
    values = [0, precision]
    diff = values[1] - values[0]
    # [0, 0, 0.5]
    if rgb[0] == 0 and rgb[1] == 0 and rgb[2] <= 128:
        return values[0]
    # [0, 0, 1]
    if rgb[0] == 0 and rgb[1] == 0:
        return values[0] + 0.125 * diff * (rgb[2]-128)/128
    # [0, 1, 1]
    if rgb[0] == 0:
        rangeVal = 0.375 - 0.125
        return (values[0]+0.125*diff) + rangeVal * diff * rgb[1]/255
    # [1, 1, 0]
    if rgb[1] == 255:
        rangeVal = 0.625 - 0.375
        return (values[0]+0.375*diff) + rangeVal * diff * rgb[0]/255
    # [1, 0, 0]
    if rgb[0] == 255 and rgb[1] >= 0:
        rangeVal = 0.875 - 0.625
        return (values[0]+0.625*diff) + rangeVal * diff * (255-rgb[1]-128)/255
    # [0.5, 0, 0]
    rangeVal = 1.0 - 0.875
    return (values[0]+0.875*diff) + rangeVal * (2*128-rgb[2])/128

def get_bw_value( rgb, precision):
    values = [0, precision]
    diff = values[1] - values[0]
    return int(values[0] + diff * (255-rgb[0])/255)






# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#              CREATING JSON ROW
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def create_image_matrix(image_array, precision):
    image_matrix = [[]]
    image_matrix.clear()
    for y in range(len(image_array)):
        image_matrix.append([])
        for x in range(len(image_array[y])):
            if image_array[y][x][3] == 0: image_matrix[y].append(-1)
            else: image_matrix[y].append(get_bw_value(image_array[y][x], precision))
    return image_matrix

def create_image_matrix_w_quality(image_array_hq, image_array_aq, precision):
    image_matrix = [[]]
    image_matrix.clear()
    for y in range(len(image_array_hq)):
        image_matrix.append([])
        for x in range(len(image_array_hq[y])):
            if image_array_hq[y][x][3] == 0:
                if image_array_aq[y][x][3] == 0:
                    image_matrix[y].append([-1, 0])
                else:
                    image_matrix[y].append([get_bw_value(image_array_aq[y][x], precision), 0])
            else:
                image_matrix[y].append([get_bw_value(image_array_hq[y][x], precision), 1])
    return image_matrix

def create_json_element(image_array, date, type):
    values = create_image_matrix(image_array, type)
    data_set = {
        "date": date,
        "values": values}
    return data_set


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



def download_hourly_images(product_type, location_name, minQa_info, date_start):
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #                   PARAMETERS
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # setting coordinates
    if location_name == "Bering Strait":
        bbox_coordinates = [-170.844, 65.396, -167.622, 66.230]
        range_h = 5
    if location_name == "Sabetta Port":
        sabetta_port_coordinates = [71.264765, 72.060155]
        bbox_coordinates = get_bbox_coordinates_from_center(sabetta_port_coordinates, 100000)
        range_h = 12
    # setting image dimension
    multiplier = 30
    width = abs(bbox_coordinates[0] - bbox_coordinates[2])
    height = abs(bbox_coordinates[1] - bbox_coordinates[3])
    n_pixel = int(math.sqrt(pow(width, 2) + pow(height, 2)) * multiplier)
    dimension = {"width": n_pixel, "height": n_pixel}
    # min data quality
    if minQa_info == "high": minQa = {}
    else: minQa = {"minQa": 0}

    date_from_h = date_start
    if date_start.year == 2019: date_from_h = date_start + datetime.timedelta(hours=22)
    if date_start.year == 2020: date_from_h = date_start + datetime.timedelta(hours=21)
    if date_start.year == 2021: date_from_h = date_start + datetime.timedelta(hours=20)
    date_to_h = date_from_h

    for hours_counter in range(int(range_h / 1.5)):
        date_from_h = date_to_h
        date_to_h = date_to_h + datetime.timedelta(hours=1.5)
        date_from_str = [date_from_h.strftime("%Y-%m-%d"), date_from_h.strftime("%H:%M:%S")]
        date_to_str = [date_to_h.strftime("%Y-%m-%d"), date_to_h.strftime("%H:%M:%S")]
        print(date_from_str)
        response = get_response(bbox_coordinates, date_from_str, date_to_str, product_type, dimension, minQa)
        in_memory_file = io.BytesIO(response.content)
        img_png = Image.open(in_memory_file)

        # plotting
        fig = plt.figure()
        plt.imshow(img_png)
        plt.show()


















# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                       MAIN
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

values = {
    "product_types": ["NO2", "CO", "CH4", "SO2"],
    "locations_name": ["Bering Strait", "Sabetta Port"],
    "minQas": ["high", "all"]
}
date = datetime.datetime.now()
date_start = date.replace(year=2021, month=5, day=19, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2020, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

product_type = values["product_types"][0]
location_name = values["locations_name"][1]
minQa = values["minQas"][1]


# FOR IMAGES DOWNLOAD
#download_images(product_type, location_name, minQa, date_start, date_end)

download_hourly_images(product_type, location_name, minQa, date_start)
