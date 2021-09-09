import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

# Your client credentials
client_id = '982de4f4-dade-4f98-9b49-4374cd896bb6'
client_secret = '%p/,0Yrd&/mO%cdudUsby[>@]MB|2<rf1<NnXkZr'

# Create a session
client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)

# Get token for the session
token = oauth.fetch_token(token_url='https://services.sentinel-hub.com/oauth/token',
                          client_id=client_id, client_secret=client_secret)

# All requests using this session will have an access token automatically added
response = oauth.post('https://creodias.sentinel-hub.com/api/v1/process',
                      json={
                          "input": {
                              "bounds": {
                                  "bbox": [
                                      13.822174072265625,
                                      45.85080395917834,
                                      14.55963134765625,
                                      46.29191774991382
                                  ]
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

print(response.content)