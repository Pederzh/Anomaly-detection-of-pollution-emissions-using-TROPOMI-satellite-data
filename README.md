# A methodology for the identification and ML-based anomaly detection of human activity in the Arctic region based on satellite pollution data

MSc Thesis Full Document: https://bit.ly/3nzNYzw

MSc Thesis Executive Summary: https://bit.ly/3oKJZj2

## Abstract

The Arctic area is vital to Europe for geopolitical, economic, security, and environmental reasons. New economic possibilities are developing as a result of the rapid decline in ice cover, such as new trade routes and access to natural resources. Indeed, as the Arctic becomes more accessible, rivalry for control of the region grows. For this reason, there is a growing demand for Europe to increase its situational awareness in the region. The objective of ARCOS is to design and implement a platform system, services, and products in support of superior monitoring of the Arctic Region. ARCOS project defines three different scales and with different levels of user interaction, and the time span in which the thesis was conducted falls within the definition of ARCOS Level 1. That is, the integration of space data sources for the observation of abnormal behaviors with automatic analytics extraction and AI techniques. Our thesis focuses on the development of a methodology for monitoring established in-land human activities using pollution satellite data. Copernicus Sentinel-5P (TROPOMI) is the satellite data imagery source for remote pollution sensing on which the entire project is based. The image processing phase has taken a significant amount of effort since it was crucial to extract useful and correct information for pollution source identification and time-series analysis. Starting from the assumption that we do not know where human activities are in advance, we have developed a method for top-down detection of pollution sources in areas of interest. During our work, we have developed a Gaussian reconstruction of the emissions (GROTE) method to estimate the emissions by analyzing pollution. Once the data has been processed, we use the processed data to train a time-series machine learning method and generate data on expected pollution emissions for each identified location. Finally, our service can be integrated into the ARCOS project and raise an alert if the difference between the forecast value and the actual value exceeds the reference baseline for determining whether the pollution emissions value falls into the category of "usual" or "anomalous" behavior.

## Outcomes




## APIs

### /prepare

#### Description
It is the first endpoint to call. This endpoint allows the system to download TROPOMI image data from Sentinel Hub for the date period provided in a given AOI and prepare data that will be used by /alerting and /processedImage.

#### Body Schema
| Name | Type | Req. | Default | Description
| --- | --- | --- | --- | --- |
| `lat` | float | Yes | / | Latitude of the central point of the AOI. The AOI will be an area of 200km x 200km. |
| `lng` | float | Yes | / | Longitude of the central point of the AOI. The AOI will be an area of 200km x 200km. |
