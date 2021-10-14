import pandas as pd
import numpy as np
import datetime
import json

# def calculateAngle(value_U, value_V):
#         return 180 + (
#                 180 / np.pi * np.arctan2(value_U / np.sqrt(np.power(value_U, 2) + np.power(value_V, 2)),
#                                          value_V / np.sqrt(np.power(value_U, 2) + np.power(value_V, 2))))

def calculateAngle(value_U, value_V):
    return 180 / np.pi * np.arctan2(value_U, value_V)


def calculateIntensity(value_U, value_V):
    return np.sqrt(np.power(value_U, 2) + np.power(value_V, 2))


location_names = ["Bering Strait", "Sabetta Port"]
location_name = location_names[1]
components = ["u", "v"]  # x, y
locNum = 184
times = [5, 6, 7, 8]
lenTimes = len(times)
pixelLen = 100
lenX = 23
lenY = 8

# making data frame from csv file

fileU = pd.read_csv(
    "./data/" + location_name + "/wind/wind-20190101_20211007-10m_" + components[0] + "-500_600_700_800.csv",
    thousands=',')
fileV = pd.read_csv(
    "./data/" + location_name + "/wind/wind-20190101_20211007-10m_" + components[1] + "-500_600_700_800.csv",
    thousands=',')

fileU.rename(columns={"Value": "value_U"}, inplace=True)
fileV.rename(columns={"Value": "value_V"}, inplace=True)

df = pd.concat([fileU, fileV], axis=1, join='outer')

#


df = df[df.value_U != "Value"]

df["value_U"] = df["value_U"].astype('|S')
df["value_V"] = df["value_V"].astype('|S')

df["value_U"] = df["value_U"].astype(float)
df["value_V"] = df["value_V"].astype(float)

df['angle'] = calculateAngle(df['value_U'], df['value_V'])
df['intensity'] = calculateIntensity(df['value_U'], df['value_V'])

print(df)
# Iterating over two columns, use `zip`
date = datetime.datetime.now()
date_start = date.replace(year=2019, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=10, day=7, hour=0, minute=0, second=0, microsecond=0)

wind = {}
windMatrix = np.full((100, 100), {})
timeIndex = 0
for i, angle, intensity in zip(df.index, df['angle'], df['intensity']):

    if i % locNum == 0:
        if i != 0:
            wind[date_start.strftime("%Y-%m-%d")][date_start.strftime("%H:%M")] = wind[date_start.strftime("%Y-%m-%d")][date_start.strftime("%H:%M")].tolist() #to print on the file the matrixes TO FIX

        if timeIndex == lenTimes:
            date_start += datetime.timedelta(days=1)
            timeIndex = 0

        wind[date_start.strftime("%Y-%m-%d")] = {}
        date_start = date_start.replace(hour=times[timeIndex])
        wind[date_start.strftime("%Y-%m-%d")][date_start.strftime("%H:%M")] = np.copy(windMatrix)
        timeIndex += 1

    subIndex = i % locNum
    x = pixelLen / lenX #4,6
    y = pixelLen / lenY #12,5

    startX = round(x * (subIndex % lenX))
    endX = round(x * (subIndex % lenX + 1))
    startY = round(y * int(subIndex / lenX))
    endY = round(y * int((subIndex / lenX + 1)))

    val = {
        "angle": angle,
        "intensity": intensity
    }
    subMatrix = np.full((endX - startX, endY - startY), val)

    wind[date_start.strftime("%Y-%m-%d")][date_start.strftime("%H:%M")][startX:endX, startY:endY] = np.copy(subMatrix)


print(wind)
info_name_write = "wind-matrix_" + location_name.strip()
file_name_write = info_name_write + date_start.strftime("%Y-%m-%d") + "_" + date_end.strftime("%Y-%m-%d") + "-500_600_700_800"
directory_path_write = "./data/" + location_name + "/wind/"
with open(directory_path_write + file_name_write + ".json", 'w') as outfile:
    json.dump(wind, outfile)
# result = [generateMatrix(x) for x in df['angle']]
