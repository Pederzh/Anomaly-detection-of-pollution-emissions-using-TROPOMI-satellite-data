import io

import matplotlib.pyplot as plt

from PIL import Image
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

imgS5.putalpha(100)  # opacity

fig=plt.figure()
#fig.add_subplot(1,2,1)
plt.imshow(imgS2)
#fig.add_subplot(2,2,1)
plt.imshow(imgS5)
plt.show()

