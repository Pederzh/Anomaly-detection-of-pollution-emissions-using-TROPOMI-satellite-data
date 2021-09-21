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

def get_response (bbox, date_from_str, date_to_str, s_product, dimension):
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
                        [minVal, [1, 1, 1]],
                        [maxVal, [0, 0, 0]],
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
                              "from": date_from_str + "T00:00:00Z",
                              "to": date_to_str + "T00:00:00Z",
                          }
                      },
                      "processing": {
                          "minQa": 0
                      }
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
            if image_array[y][x][3] != 0: image_matrix[y].append(get_bw_value(image_array[y][x], precision))
    return image_matrix

def create_json_element(image_array, date, type):
    values = create_image_matrix(image_array, type)
    data_set = {
        "date": date,
        "values": values}
    return data_set









# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#               PARAMETERS DEFINITION
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# product selected (NO2, CO, CH4)
product_type = "NO2"

# area coordinates
bbox_coordinates = [ 71.779, 71.138, 72.683, 71.374]
location_name = "Bering Strait" # [ -170.844, 65.396, -167.622, 66.230]
location_name = "Sabetta Port" # [ 71.779, 71.138, 72.683, 71.374]

# image dimension
multiplier = 30
width = abs(bbox_coordinates[0]-bbox_coordinates[2])
height = abs(bbox_coordinates[1]-bbox_coordinates[3])
n_pixel = int(math.sqrt(pow(width,2)+pow(height,2)) * multiplier)
dimension = { "width": n_pixel, "height": n_pixel}

# time window considered
date = datetime.datetime.now()
date_start = date.replace(year=2020, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
year = str(date_start.year)

# time range for the sampling period
time_sp = 1 # in days

# values precision (values converted in %)
precision = 10000

# if want to save png and json
save_png = True
save_json = True

# if the image to load is in a file
read_from_file = False

# directory path
directory_path = "./Data/" + location_name + "/" + product_type + "/"




# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                   RUNNING
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# GETTING AND SAVING THE IMAGES FORM TROPOMI
images = []
date_from = date_start
date_to = date_start

# initializing json data
info_set = {
            "product_type": product_type,
            "location_name": location_name,
            "bbox": bbox_coordinates,
            "image_dimension": dimension,
            "min_max": get_min_max_val(product_type),
            "precision": precision
        }
data_set = {}

for day_counter in range(int((date_end-date_start).days/time_sp)):
    date_from = date_to
    date_to = date_to + datetime.timedelta(days=time_sp)
    # setting date from string for the api call
    date_from_str = str(date_from.year)+"-"
    if (date_from.month<10): date_from_str+="0"
    date_from_str += str(date_from.month)+"-"
    if (date_from.day < 10): date_from_str += "0"
    date_from_str += str(date_from.day)
    # setting date to string for the api call
    date_to_str = str(date_to.year) + "-"
    if (date_to.month < 10): date_to_str += "0"
    date_to_str += str(date_to.month) + "-"
    if (date_to.day < 10): date_to_str += "0"
    date_to_str += str(date_to.day)
    print("calling for range   " + date_from_str + "    to    " + date_to_str)
    # GETTING THE IMAGE
    if not read_from_file:
        # FROM POST CALL
        response = get_response(bbox_coordinates, date_from_str, date_to_str, product_type, dimension)
        in_memory_file = io.BytesIO(response.content)
        img_png = Image.open(in_memory_file)
    if read_from_file:
        # FROM FILE
        img_png = Image.open(directory_path + "Images/" + date_from_str + ".png")
    # SAVING THE RESPONSE CONTENT AS AN IMAGE
    img = array(img_png)
    # SAVING PNG IMAGE
    if save_png:
        img_png.save(directory_path + "Images/" + date_from_str + ".png", format="png")
    # ADDING THE IMAGE TO JSON
    data_set[date_from_str] = create_image_matrix(img, precision)

# SAVING JSON FILES
if save_json:
    with open(directory_path+ year +'.txt', 'w') as outfile:
        json.dump(data_set, outfile)
    with open(directory_path+'info.txt', 'w') as outfile:
        json.dump(info_set, outfile)

print("END")