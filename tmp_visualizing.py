import io

import matplotlib.pyplot as plt

from PIL import Image, ImageTk
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

bbox_coordinates = [ 13.822174072265625,
                                      45.85080395917834,
                                      14.55963134765625,
                                      46.29191774991382 ]
class Time_ranged:
    from_ = "2020-09-05T00:00:00Z"
    to_ = "2020-09-06T00:00:00Z"
    year_ = "2019"
    month_ = "09"

# GETTING THE IMAGE FORM SATELLITE S2
responseS2 = oauth.post('https://creodias.sentinel-hub.com/api/v1/process',
                      json={
                          "input": {
                              "bounds": {
                                  "bbox": bbox_coordinates
                              },
                              "data": [{
                                  "type": "sentinel-2-l2a"
                              }]
                          },
                          "evalscript": """
    //VERSION=3

    function setup() {
      return {
        input: ["B02", "B03", "B04"],
        output: {
          bands: 3
        }
      };
    }

    function evaluatePixel(
      sample,
      scenes,
      inputMetadata,
      customData,
      outputMetadata
    ) {
      return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
    }
    """
})

# GETTING THE IMAGES FORM TROPOMI
response_list = []
range_ = 20
for tmp in range(range_):
    x = tmp+1
    day_from = "" + str(x)
    day_to = "" + str(x+1)
    if (x<10):
        day_from = "0"+day_from
    if ((x+1)<10):
        day_to = "0"+day_to
    responseS5 = oauth.post('https://creodias.sentinel-hub.com/api/v1/process',
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
                                                    "from": Time_ranged().year_+"-"+Time_ranged().month_+"-"+day_from+"T00:00:00Z",
                                                    "to": Time_ranged().year_+"-"+Time_ranged().month_+"-"+day_to+"T00:00:00Z",
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
    response_list.append(responseS5.content)

# code to print the two images individually in two windows
"""
in_memory_file = io.BytesIO(responseS2.content)
im = Image.open(in_memory_file)
im.show()
in_memory_file = io.BytesIO(responseS5.content)
im = Image.open(in_memory_file)
im.show()
"""

in_memory_file = io.BytesIO(responseS2.content)
imgS2 = Image.open(in_memory_file)
list_imgS5 = []
for x in range(range_):
    in_memory_file = io.BytesIO(response_list[x])
    list_imgS5.append(Image.open(in_memory_file))
    list_imgS5[x].putalpha(150)




""" PANEL GRAPICH """

root = Tk()
root.title("TROPOMI TIME SERIES")
root.geometry("600x500")
#Button(root, text=f'BUTTON').grid(row=10, column = 1, pady = 10, padx = 10)

# creating the main frame
main_frame = Frame(root)
main_frame.pack(fill=BOTH, expand=1)

# creating a canvas
my_canvas = Canvas(main_frame)
my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

# creating the TROPOMI image
img_displayed = ImageTk.PhotoImage(list_imgS5[0])
panel = Label(root, image = img_displayed)
panel.image = img_displayed
panel.pack(side = "right", fill = "both", expand = "yes")

# creating the S2 image
img_displayed = ImageTk.PhotoImage(imgS2)
panelS2 = Label(root, image = img_displayed)
panelS2.pack(side = "left", fill = "both", expand = "yes")

# creating a scroll-bar
my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
my_scrollbar.pack(side=RIGHT, fill=Y)

# function that the scrollbar calls when scrolling
def change_image(e, a):
    if (float(a)>1-1/range_): a=str(1-1/range_)
    if (float(a)<0): a=0
    img_displayed = ImageTk.PhotoImage(list_imgS5[int(range_*float(a))])
    panel.configure(image=img_displayed)
    panel.image = img_displayed

my_scrollbar.config( command=change_image)





root.mainloop()