import io
import math

import matplotlib.pyplot as plt

from PIL import Image
from numpy import array, ndarray, asarray
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session


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
    from_ = "2021-09-05T00:00:00Z"
    to_ = "2021-09-06T00:00:00Z"

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

# GETTING THE IMAGE FORM TROPOMI
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
                                                "from": Time_ranged().from_,
                                                "to": Time_ranged().to_
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
in_memory_file = io.BytesIO(responseS5.content)
imgS5 = Image.open(in_memory_file)

#imgS5.putalpha(100)  # opacity

fig=plt.figure()
#fig.add_subplot(1,2,1)
plt.imshow(imgS2)
#fig.add_subplot(2,2,1)
plt.imshow(imgS5)
plt.show()

# for RGBA images
def img_smoothing(img, radius, decreasing_ration):
    arrayS5 = array(img)
    new_arrayS5 = arrayS5
    weights = [1.0]
    for tmp in range(radius): weights.append(weights[tmp]/decreasing_ration)
    max_radius = radius
    def distance(x1, y1, x2, y2):
        dist = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
        dist_to_return = int(dist)
        if (dist > float(dist_to_return)+0.5): dist_to_return+=1
        return dist_to_return
    for y in range(len(arrayS5)):
        for x in range(len(arrayS5[y])):
            y_start = y-max_radius
            x_start = x-max_radius
            y_end = y+max_radius
            x_end = x+max_radius
            if (y_start<0): y_start=0
            if (x_start < 0): x_start = 0
            if (x_end>len(arrayS5[y])-1): x_end = len(arrayS5[y])-1
            if (y_end > len(arrayS5)-1): y_end = len(arrayS5)-1
            pixel_tmp = [1.0, 1.0, 1.0, 1.0]
            for tmp in range(4): pixel_tmp[tmp] = float(arrayS5[y][x][tmp])
            sum = 1.0
            for y1 in range(y_start, y_end + 1):
                for x1 in range(x_start, x_end + 1):
                    dist = distance(x, y, x1, y1)
                    if (dist <= max_radius):
                        for tmp in range(4): pixel_tmp[tmp] += (float(arrayS5[y1][x1][tmp]) * weights[dist])
                        sum+=weights[dist]
            for tmp in range(4): pixel_tmp[tmp] = pixel_tmp[tmp] / sum
            for tmp in range(4):
                new_value = int(pixel_tmp[tmp])
                if (pixel_tmp[tmp]>new_value+0.5): new_value+=1
                new_arrayS5[y][x][tmp] = new_value

    return Image.fromarray(new_arrayS5, 'RGBA')

imgS5 = img_smoothing(imgS5, 3, 2)
plt.imshow(imgS5)
plt.show()


