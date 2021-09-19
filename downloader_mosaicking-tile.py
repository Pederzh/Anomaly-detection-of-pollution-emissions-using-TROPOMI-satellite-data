import datetime
import io

import matplotlib.pyplot as plt

from PIL import Image
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

def get_response (bbox, date_from_str, date_to_str, mosaicking, dimension):

    if mosaicking == "TILE":
        product_evalscritp = """
                    //VERSION=3
                    function setup() {
                        return {
                            input: ["NO2", "dataMask"],
                                output: { bands:  4},
                                mosaicking: "ORBIT"
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
                    
                    function evaluatePixel(samples){
                      var sum = 0;
                      var nonZeroSamples = 0;
                      for (var i = 0; i < samples.length; i++) {
                        var value = samples[i].NO2;
                        if (value != 0) {
                          sum += value;
                          nonZeroSamples++;
                        }
                      }
                      return viz.process(samples[0].NO2);
                    }
                    """
    if mosaicking == "SIMPLE":
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
#               PARAMETERS DEFINITION
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# product selected (NO2, CO, CH4)
product_type = "NO2"

# area coordinates
bbox_coordinates = [ -170.844, 65.396, -167.622, 66.230]
location_name = "Bering Strait" # [ -170.844, 65.396, -167.622, 66.230]
#location_name = "Sabetta Port" # [ 71.779, 71.138, 72.683, 71.374]
#location_name = "Russia & Arctic" # [36.055, 35.531, -178.398, 81.025] [72.442, 54.825, 136.075 76.164]

# image dimension
dimension = { "width": 50, "height": 50 }
#dimension = { "width": 1024, "height": 1024 }

# time window considered
date = datetime.datetime.now()
date_start = date.replace(year=2021, month=8, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=8, day=2, hour=0, minute=0, second=0, microsecond=0)

# time range for the sampling period
time_sp = 1 # in days



# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                   RUNNING
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# GETTING AND SAVING THE IMAGES FORM TROPOMI
images = []
date_from = date_start
date_to = date_start
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
    # PRINT
    fig = plt.figure()
    # SENDING POST REQUEST
    response = get_response(bbox_coordinates, date_from_str, date_to_str, "TILE", dimension)
    in_memory_file = io.BytesIO(response.content)
    img = Image.open(in_memory_file)
    plt.imshow(img)
    plt.show()
    # response = get_response(bbox_coordinates, date_from_str, date_to_str, "SIMPLE", dimension)
    # in_memory_file = io.BytesIO(response.content)
    # img = Image.open(in_memory_file)
    # plt.imshow(img)
    # plt.show()
    # SAVING THE RESPONSE CONTENT AS AN IMAGE
    #in_memory_file = io.BytesIO(response.content)
    #images.append(Image.open(in_memory_file))
    #images[day_counter].save("./Images/" + location_name + "/" + product_type + "/" + date_from_str + " "+location_name+".png", format="png")
