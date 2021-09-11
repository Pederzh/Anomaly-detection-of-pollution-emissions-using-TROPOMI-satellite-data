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







# area coordinates
bbox_coordinates = [13.82,45.85,14.56,46.29]
bbox_str = str(bbox_coordinates[0])+"-"+str(bbox_coordinates[1])+"-"+str(bbox_coordinates[2])+"-"+str(bbox_coordinates[3])

# time window considered
date = datetime.datetime.now()
date_start = date.replace(year=2020, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2020, month=1, day=15, hour=0, minute=0, second=0, microsecond=0)
date = date + datetime.timedelta(days=5)

# time range for the sampling period
time_sp = 1 # in days

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
    response = oauth.post('https://creodias.sentinel-hub.com/api/v1/process',
                            json={
                                "input": {
                                    "bounds": {
                                        "properties": {
                                            "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                                        },
                                        "bbox": bbox_coordinates
                                    },
                                    "data": [
                                        {
                                            "type": "sentinel-5p-l2",
                                            "dataFilter": {
                                                "timeRange": {
                                                    "from": date_from_str+"T00:00:00Z",
                                                    "to": date_to_str+"T00:00:00Z",
                                                },
                                              "timeliness": "NRTI"
                                            },
                                            "processing": {
                                                "minQa": 0
                                            }
                                        }
                                    ]
                                },
                                "evalscript": """
                              //VERSION=3
                                function setup() {
                                  return {
                                    input: ["NO2", "dataMask"],
                                    output: { bands:  4 }
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
                            })
    in_memory_file = io.BytesIO(response.content)
    images.append(Image.open(in_memory_file))
    images[day_counter].save("./Images/NO2/" + date_from_str + " "+bbox_str+".png", format="png")
