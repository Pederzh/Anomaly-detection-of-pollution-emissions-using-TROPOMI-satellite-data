# A methodology for the identification and ML-based anomaly detection of human activity in the Arctic region based on satellite pollution data

MSc Thesis Full Document: https://bit.ly/3nzNYzw

MSc Thesis Executive Summary: https://bit.ly/3oKJZj2

## Abstract

The Arctic area is vital to Europe for geopolitical, economic, security, and environmental reasons. New economic possibilities are developing as a result of the rapid decline in ice cover, such as new trade routes and access to natural resources. Indeed, as the Arctic becomes more accessible, rivalry for control of the region grows. For this reason, there is a growing demand for Europe to increase its situational awareness in the region. The objective of ARCOS is to design and implement a platform system, services, and products in support of superior monitoring of the Arctic Region. ARCOS project defines three different scales and with different levels of user interaction, and the time span in which the thesis was conducted falls within the definition of ARCOS Level 1. That is, the integration of space data sources for the observation of abnormal behaviors with automatic analytics extraction and AI techniques. Our thesis focuses on the development of a methodology for monitoring established in-land human activities using pollution satellite data. Copernicus Sentinel-5P (TROPOMI) is the satellite data imagery source for remote pollution sensing on which the entire project is based. The image processing phase has taken a significant amount of effort since it was crucial to extract useful and correct information for pollution source identification and time-series analysis. Starting from the assumption that we do not know where human activities are in advance, we have developed a method for top-down detection of pollution sources in areas of interest. During our work, we have developed a Gaussian reconstruction of the emissions (GROTE) method to estimate the emissions by analyzing pollution. Once the data has been processed, we use the processed data to train a time-series machine learning method and generate data on expected pollution emissions for each identified location. Finally, our service can be integrated into the ARCOS project and raise an alert if the difference between the forecast value and the actual value exceeds the reference baseline for determining whether the pollution emissions value falls into the category of "usual" or "anomalous" behavior.


## REST APIs
The tool as per specifications, had to be embeddable with other services of Copernicus SEA (Support to EU External Actions) Security Service, for this reason has been designed as a RESTful service. To interact with our service, the user or whoever will be integrating our service within the vast ARCOS project would then simply need to call the API endpoints we expose. In order to let our service to bulk download and prepare daily images of the AOI and then retrieve data of the pollution values for each identified human activity and as well as an alerting.

### Prepare

#### Operation
POST /prepare

#### Description
It is the first endpoint to call. This endpoint allows the system to download TROPOMI image data from Sentinel Hub for the date period provided in a given AOI and prepare data that will be used by /alerting and /processedImage.

#### Body Schema
| Name | Type | Req. | Default | Description
| --- | --- | --- | --- | --- |
| `lat` | float | Yes | / | Latitude of the central point of the AOI. The AOI will be an area of 200km x 200km. |
| `lng` | float | Yes | / | Longitude of the central point of the AOI. The AOI will be an area of 200km x 200km. |
| `date` | string | Yes | / | Last day of the sensing period. The sensing pe- riod by default is 365 days from the provided date. |
| `product_type` | string | No | NO2 | Pollutant chemical code for TROPOMI product type. Accepted values [NO2, CH4, CO]. |
| `sentinel_hub_client_id` | string | Yes | / | Sentinel hub OAuth client id. |
| `sentinel_hub_client_secret` | string | Yes | / | Sentinel hub OAuth client id. |
| `sensing_period` | integer | No | 365 | Sensing days period until the date provided. Download and use the image data of the whole period. |
| `sensing_start_hours` | integer | No | 0 | First considered hour for daily images. |
| `sensing_range_hours` | integer | No | 24 | Considered hours range for daily images. |
| `peaks_sensing_period` | integer | No | 30 | Sensing period for location pinpointing in AOI. |

#### Body example
```yaml
{
"lat": 71.264765,
"lng": 72.060155,
"date": "2021-08-29T00:00:00",
"product_type": "NO2",
"sentinel_hub_client_id": "SentinelHub-OAuth-client-id",
"sentinel_hub_client_secret": "SentinelHub-OAuth-client-secret",
"sensing_period": 375,
"sensing_start_hours": 0,
"sensing_range_hours": 24,
"peaks_sensing_period": 30
}
```

#### Response
The prepare endpoint serves the purpose of enabling our service to download the TROPOMI images from Sentinel Hub.
Then process the images in order to identify the peaks for pinpointing the locations. Finally to process the values of pollutant emissions from the plumes to the location points through our developed GROTE method.
When the prepare endpoint, for a given location and time period, is called; a thread is started to perform the required operations.



### Alerting

#### Operation
POST /alerting

#### Description
This endpoint returns the list of pollution source points present in the AOI. For each source point it returns information about the actual detected value after being processed, and the machine-learning estimated value based on the time-series analysis.
The values are expressed through Gaussian parameters, that are: volume, peak, attenuation.
The degree of deviation of the detected value from the expected value is indicated in the ’status’ parameter.
This endpoint can only be called after the endpoint /prepare has been called and its data preparation process (in a given AOI, period, and pollutant) has been completed.

#### Body Schema
| Name | Type | Req. | Default | Description
| --- | --- | --- | --- | --- |
| `lat` | float | Yes | / | Latitude of the central point of the AOI. The AOI will be an area of 200km x 200km. |
| `lng` | float | Yes | / | Longitude of the central point of the AOI. The AOI will be an area of 200km x 200km. |
| `date` | string | Yes | / | Last day of the sensing period. The sensing pe- riod by default is 365 days from the provided date. |
| `product_type` | string | No | NO2 | Pollutant chemical code for TROPOMI product type. Accepted values [NO2, CH4, CO]. |
| `sensing_period` | integer | No | 365 | Sensing days period until the date provided. Use the image data of the whole period. |
| `peaks_sensing_period` | integer | No | 30 | Sensing period for location pinpointing in AOI. |
| `range_alerting` | integer | No | 10 | It considers the mean of the number "range_alerting" of im- ages until the provided date for alerting. |

#### Body example
```yaml
{
"lat": 71.264765,
"lng": 72.060155,
"date": "2021-08-29T00:00:00",
"product_type": "NO2",
"sensing_period": 375,
"peaks_sensing_period": 30,
"range_alerting": 10
}
```

#### Response
The alerting system was designed to provide information on emissions at source points. Together with a property that indicates whether the emission is normal, higher than expected, or abnormal.
For this reason the response returns an array as object in which are listed all the top-down identified sources of pollution in the AOI.
Each pollution source in the response has the following information:


| Name | Type | Description
| --- | --- | --- |
| `actual_value` | Gaussian parameters object | Actual sensed pollution value reconstructed from the pollution plume and expressed through Gaussian parameters. |
| `forecast_value` | Gaussian parameters object | Forecast pollution value with time-series ML algorithm and expressed through Gaussian parameters. |
| `other_information` | Location object | Pollution source location coordinates, considered date and previous days range. |
| `status` | Enum | Pollution source emission value sta- tus compared to the forecast value. Possible values: [GREEN, YELLOW, RED]. |

##### Gaussian parameters object
Every value is an average of the values of the "days_range" previous days. Since NO2 concentrations change from day to day due to weather variables; thus, conclusions cannot be drawn based just on one day of data, described in section sec. 3.6.2. Values are expressed as Gaussian parameters:
• GREEN = Normal: actual emissions are similar the expected forecast
emissions value.
• YELLOW = High: actual emissions are higher than the expected forecast emissions value.
• RED = Alerting: actual emissions are abnormal than the expected forecast emissions value.

##### Status
The property indicates whether or not the actual emissions are such that an anomaly is detected when compared to the expected emissions.
Status can assume the following values:
• Peak = a: maximum value assumed in the point of emission.
• Attenuation = b: attenuation parameter of Gaussian emission.
• Volume = value: emission value reconstructed as Gaussian volume.

#### Response Example
```yaml
{
"1": {...},
"2": {...},
"3": {
  "actual_value": {
    "attenuation": 0.11512570982940673,
    "peak": 75.44932489015221,
    "volume": 2058.8888888888887
  },
  "forecasted_value": {
    "attenuation": 0.06382749964024934,
    "peak": 245.46203834214054,
    "volume": 12081.653530801428
  },
  "other_information": {
    "coordinates": [
      70.97159059687324,
      73.75148733924422
    ],
    "date": "Fri, 27 Aug 2021 00:00:00 GMT",
    "days_range": 10
  },
  "status": "YELLOW"
}
}
```

### Processed Image

#### Operation
POST /processedImage

#### Description
This endpoint returns the processed image of the AOI in which the pollution value from source-points are shown graphically. The values are the average of estimated values in the number of ’range_daily_images’ days up to the indicated day.
This endpoint can only be called after the endpoint /prepare has been called and its data preparation process (in a given AOI, period, and pollutant) has been completed.

#### Body Schema
| Name | Type | Req. | Default | Description
| --- | --- | --- | --- | --- |
| `lat` | float | Yes | / | Latitude of the central point of the AOI. The AOI will be an area of 200km x 200km. |
| `lng` | float | Yes | / | Longitude of the central point of the AOI. The AOI will be an area of 200km x 200km. |
| `date` | string | Yes | / | Last day of the sensing period. The sensing period by default is 365 days from the provided date. |
| `product_type` | string | No | NO2 | Pollutant chemical code for TROPOMI product type. Accepted values [NO2, CH4, CO]. |
| `range_daily_images` | integer | No | 10 | It considers the mean of the number "range_alerting" of images until the provided date for generating the image. |
| `peaks_sensing_period` | integer | No | 30 | Sensing period for location pinpointing in AOI. |

#### Body example
```yaml
{
"lat": 71.264765,
"lng": 72.060155,
"date": "2021-08-29T00:00:00",
"product_type": "NO2",
"range_daily_images": 10,
"peaks_sensing_period": 30
}
```

#### Response
The processed image retrieval system is designed to return a TROPOMI-style PNG image of the 200km x 200km AOI of the central point coordinates provided.
The image has been cleaned of pixel noise, pollution plumes have been removed, and their values have been used to calculate pollution emissions from the pinpointed locations.
The processed image service, like the alerting service, uses values that are an average of the values of the "days_range" days prior to the selected day.

