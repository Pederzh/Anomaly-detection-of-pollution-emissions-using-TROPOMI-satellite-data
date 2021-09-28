import datetime
import json
import statistics as stst
from pathlib import Path
import plotly.express as px
import pandas as pd




def list_mean_value(sorted_list):
    if len(sorted_list) == 0: return -1
    return stst.mean(sorted_list)

def list_mode_value(sorted_list):
    if len(sorted_list) == 0: return -1
    return stst.mode(sorted_list)

def list_median_value(sorted_list):
    if len(sorted_list) == 0: return -1
    return stst.median(sorted_list)

def image_to_sorted_list(image):
    list = []
    for i in range(len(image)): list += image[i]
    while -1 in list: list.remove(-1)
    list.sort()
    return list

def append_stat(obj, obj_list):
    tmp_keys = list(obj_list["statistics"].keys())
    for tmp in range(len(tmp_keys)):
        obj_list["statistics"][tmp_keys[tmp]].append(obj["statistics"][tmp_keys[tmp]])
    return obj_list

def get_image_stats(image):
    sorted_list = image_to_sorted_list(image)
    median = list_median_value(sorted_list)
    mode = list_mode_value(sorted_list)
    mean = list_mean_value(sorted_list)
    stats = {
        "statistics": {
            "mean": mean,
            "mode": mode,
            "median": median,
        },
    }
    return stats


def get_days_stats(data):
    stats = {}
    keys = list(data.keys())
    for i in range(len(keys)):
        stat_quality = {
            "statistics": {
                "mean": [],
                "mode": [],
                "median": [],
            },
        }
        stats[keys[i]] = {}
        tmp_img = []
        for y in range(len(data[keys[i]])):
            tmp_img.append([])
            for x in range(len(data[keys[i]][y])):
                tmp_img[y].append(data[keys[i]][y][x][0])
        # getting all quality image stats
        stat = get_image_stats(tmp_img)
        stat_quality = append_stat(stat, stat_quality)
        # getting high quality image stats
        for y in range(len(data[keys[i]])):
            for x in range(len(data[keys[i]][y])):
                if data[keys[i]][y][x][1] == 0 and data[keys[i]][y][x][0] != -1:
                    tmp_img[y][x] = -1
        stat = get_image_stats(tmp_img)
        stat_quality = append_stat(stat, stat_quality)
        # putting both stat in stats
        stats[keys[i]] = stat_quality
    return stats



















location_names = ["Bering Strait", "Sabetta Port"]
product_types = ["CO", "NO2", "CH4"]
statistic_types = ["mean", "median", "mode", "variance"]

location_name = location_names[0]
product_type = product_types[0]
statistic_type = statistic_types[0]

product_days = {
}

for i in range(len(product_types)):
    directory_path = "./Data/" + location_name + "/" + product_types[i] + "/"
"""with open(directory_path + "days.json") as json_file:
        days = json.load(json_file)
    product_days[product_types[i]] = days"""


date = datetime.datetime.now()
directory_path = "./Data/" + location_name + "/" + product_type + "/"
date_start = date.replace(year=2019, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=9, day=1, hour=0, minute=0, second=0, microsecond=0)
date_from = date_start
date_to = date_start

print("reading data file")
with open(directory_path + "2019.json") as json_file:
    data_2019 = json.load(json_file)
with open(directory_path + "2020.json") as json_file:
    data_2020 = json.load(json_file)
with open(directory_path + "2021.json") as json_file:
    data_2021 = json.load(json_file)
data_set = dict(data_2019["data"])
data_set.update(data_2020["data"])
data_set.update(data_2021["data"])


subdivision = 4
data = data_set[list(data_set.keys())[0]]
data_len = len(data_set[list(data_set.keys())[0]])
sub_len = int(data_len/subdivision)
keys = list(data_set.keys())
final_matrix = []
stats_matrix = []
stats = {}
for y_len in range(subdivision):
    final_matrix.append([])
    stats_matrix.append([])
    for x_len in range(subdivision):
        final_matrix[y_len].append({})
        stats_matrix[y_len].append({})
        print(subdivision*subdivision - ((y_len)*subdivision + (x_len)))
        for kk in range(len(keys)):
            sub_matrix = []
            for y in range(sub_len):
                sub_matrix.append([])
                for x in range(sub_len):
                    sub_matrix[y].append(data_set[keys[kk]][y_len * sub_len + y][x_len * sub_len + x])
            final_matrix[y_len][x_len][keys[kk]] = sub_matrix
        #stats_matrix[y_len][x_len] = get_days_stats(final_matrix[y_len][x_len])
        tmp_key = "y:" + str(y_len) + ", x:" + str(x_len)
        stats[tmp_key] = get_days_stats(final_matrix[y_len][x_len])

df_array = []

df = pd.DataFrame()

time_sp = 1
for day_counter in range(int((date_end - date_start).days)):
    date_from = date_to
    date_to = date_to + datetime.timedelta(days=time_sp)
    # setting date from string for the api call
    date_from_str = date_from.strftime("%Y-%m-%d")
    # setting date to string for the api call
    date_to_str = date_to.strftime("%Y-%m-%d")
    keys = list(stats.keys())
    for i in range(len(keys)):
        value = stats[keys[i]][date_from_str]["statistics"][statistic_type][1]
        if value == -1: value = None

        data = {"date": date_from_str,
                "position": keys[i],
                "value": value}
        df_array.append(data)


print(df_array)

df = pd.DataFrame(df_array)
print(df)

fig = px.histogram(df, x="date", y="value", color="position", barmode="overlay", nbins = int((date_end - date_start).days))
fig.show()

"""directory_path_save = "./data/" + location_name + "/" + product_type + "/charts"
Path(directory_path).mkdir(parents=True, exist_ok=True)
file_name = location_name.strip() + "_" + product_type + "_" + date_start.strftime(
    "%Y-%m-%d") + "_" + date_end.strftime("%Y-%m-%d") + "_DISTPLOT";
fig.write_html(directory_path_save + "/" + file_name + ".html")"""