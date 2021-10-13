import pandas as pd

import numpy as np


def calculateAngle(value_U, value_V):
        return 180 + (
                180 / np.pi * np.arctan2(value_U / np.sqrt(np.power(value_U, 2) + np.power(value_V, 2)),
                                         value_V / np.sqrt(np.power(value_U, 2) + np.power(value_V, 2))))



location_names = ["Bering Strait", "Sabetta Port"]
location_name = location_names[1]
components = ["u", "v"]  # x, y

pd.set_option('display.float_format', lambda x: '%.5f' % x)

# making data frame from csv file

fileU = pd.read_csv("./data/" + location_name + "/wind/wind-20190101_20211007-10m_" + components[0] + "-500_600_700_800.csv", thousands=',')
fileV = pd.read_csv("./data/" + location_name + "/wind/wind-20190101_20211007-10m_" + components[1] + "-500_600_700_800.csv", thousands=',')

fileU.rename(columns={"Value": "value_U"}, inplace=True)
fileV.rename(columns={"Value": "value_V"}, inplace=True)

df = pd.concat([fileU, fileV], axis=1, join='inner')

#


df = df[df.value_U != "Value"]
print(df.info())
df["value_U"] = df["value_U"].apply(lambda x: '%.5f' % x)
df["value_V"] = df["value_V"].apply(lambda x: '%.5f' % x)

#fileU['angle'] = calculateAngle(fileU['value_U'], fileU['value_V'])

print(df.head(187))
