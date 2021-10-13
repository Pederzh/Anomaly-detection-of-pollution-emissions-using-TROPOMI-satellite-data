import pandas as pd

import numpy as np

location_names = ["Bering Strait", "Sabetta Port"]
location_name = location_names[1]
components = ["u", "v"]  # x, y

# making data frame from csv file
fileU = pd.read_csv("./data/" + location_name + "/wind/wind-20190101_20211007-10m_" + components[0] + "-500_600_700_800.csv")
fileV = pd.read_csv("./data/" + location_name + "/wind/wind-20190101_20211007-10m_" + components[1] + "-500_600_700_800.csv")

fileU.rename(columns={"Value": "value_U"}, inplace=True)
fileU["value_V"] = fileV["Value"]

fileU['angle'] = 180 + (
            180 / np.pi * np.arctan2(fileU['value_U'] / np.sqrt(fileU['value_U'] ^ 2 + fileU['value_V'] ^ 2),
                                     fileU['value_V'] / np.sqrt(fileU['value_U'] ^ 2 + fileU['value_V'] ^ 2)))

# df = fileU[fileU.value_U != "Value"]

print(fileU.head(187))
