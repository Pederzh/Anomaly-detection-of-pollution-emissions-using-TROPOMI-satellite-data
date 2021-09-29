import datetime
import json
from pathlib import Path
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd










def add_up_frequencies(frequencies, to_sum, precision):
    keys = list(to_sum.keys())
    for f in range(len(keys)):
        key = int_to_str(int(keys[f]), str(precision))
        if key not in frequencies.keys():
            frequencies[key] = to_sum[keys[f]]
        else:
            frequencies[key] += to_sum[keys[f]]
    return frequencies

def change_frequencies_precision(frequencies, precision):
    new_frequencies = {}
    keys = list(frequencies.keys())
    for i in range(len(keys)):
        key = str(int(int(keys[i]) % precision))
        if key not in new_frequencies.keys():
            new_frequencies[key] = frequencies[keys[i]]
        else:
            new_frequencies[key] += frequencies[keys[i]]
    return new_frequencies


def normalize_frequencies(frequencies, tot):
    keys = list(frequencies.keys())
    for i in range(len(keys)):
        frequencies[keys[i]] = frequencies[keys[i]]/tot
    return frequencies

def int_to_str(num, max_value):
    num = str(num)
    while len(num) < len(max_value)-1:
        num = "0" + num
    return num

def int_to_day_of_week(day):
    if day == 0: return "Monday"
    if day == 1: return "Tuesday"
    if day == 2: return "Wednesday"
    if day == 3: return "Thursday"
    if day == 4: return "Friday"
    if day == 5: return "Saturday"
    return "Sunday"







def get_weekend_vs_weekdays_chart(days, date_start, date_end, data_quality, location_name, product_type):
    date_to = date_start
    new_precision = 100000
    weekdays_frequencies = {}
    weekend_frequencies = {}
    tot_weekend = 0
    tot_weekdays = 0
    df_array = []
    for day_counter in range(int((date_end - date_start).days)):
        date_from = date_to + datetime.timedelta(days=day_counter)
        # setting date from string for the api call
        date_from_str = date_from.strftime("%Y-%m-%d")
        frequencies = days[date_from_str]["frequencies"][data_quality]
        non_zeroes = days[date_from_str]["statistics"]["non_zeroes"][data_quality]
        tot_pixels = days[date_from_str]["statistics"]["n_tot"][data_quality]
        # frequencies = change_frequencies_precision(frequencies, new_precision)
        if date_from.weekday() == 5 or date_from.weekday() == 6:
            if len(list(frequencies.keys())) > 0:
                tot_weekend += non_zeroes / tot_pixels
                weekend_frequencies = add_up_frequencies(weekend_frequencies, frequencies, new_precision)
        else:
            if len(list(frequencies.keys())) > 0:
                tot_weekdays += non_zeroes / tot_pixels
                weekdays_frequencies = add_up_frequencies(weekdays_frequencies, frequencies, new_precision)

    weekend_frequencies = normalize_frequencies(weekend_frequencies, tot_weekend)
    weekdays_frequencies = normalize_frequencies(weekdays_frequencies, tot_weekdays)

    keys = list(weekend_frequencies.keys())
    keys.sort()
    for i in range(len(keys)):
        df_array.append({
            "normalized_frequencies": weekend_frequencies[keys[i]],
            "values": keys[i],
            "type": "weekend"
        })
    keys = list(weekdays_frequencies.keys())
    keys.sort()
    for i in range(len(keys)):
        df_array.append({
            "normalized_frequencies": weekdays_frequencies[keys[i]],
            "values": keys[i],
            "type": "weekdays"
        })

    df = pd.DataFrame(df_array)

    fig = px.line(df, x="values", y="normalized_frequencies", color="type")
    fig.update_xaxes(categoryorder='category ascending')
    fig.show()
    return fig

def get_days_vs_days_chart(days, date_start, date_end, data_quality, location_name, product_type):
    date_to = date_start
    new_precision = 100000
    final_frequencies = [{}, {}, {}, {}, {}, {}, {}]
    tot = [0, 0, 0, 0, 0, 0, 0]
    df_array = []
    for day_counter in range(int((date_end - date_start).days)):
        date_from = date_to + datetime.timedelta(days=day_counter)
        # setting date from string for the api call
        date_from_str = date_from.strftime("%Y-%m-%d")
        frequencies = days[date_from_str]["frequencies"][data_quality]
        non_zeroes = days[date_from_str]["statistics"]["non_zeroes"][data_quality]
        tot_pixels = days[date_from_str]["statistics"]["n_tot"][data_quality]
        # frequencies = change_frequencies_precision(frequencies, new_precision)
        d_of_w = date_from.weekday()
        final_frequencies[d_of_w] = add_up_frequencies(final_frequencies[d_of_w], frequencies, new_precision)
        tot[d_of_w] += non_zeroes / tot_pixels

    for i in range(7):
        final_frequencies[i] = normalize_frequencies(final_frequencies[i], tot[i])
        keys = list(final_frequencies[i].keys())
        keys.sort()
        for j in range(len(keys)):
            df_array.append({
                "normalized_frequencies": final_frequencies[i][keys[j]],
                "values": int(keys[j]),
                "type": int_to_day_of_week(i)
            })

    df = pd.DataFrame(df_array)

    fig = px.histogram(df, x="values", y="normalized_frequencies", color="type", barmode="overlay", nbins = 50)
    fig.update_xaxes(categoryorder='category ascending')
    fig.show()
    return fig









date = datetime.datetime.now()
date_start = date.replace(year=2019, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=9, day=1, hour=0, minute=0, second=0, microsecond=0)

location_names = ["Bering Strait", "Sabetta Port"]
product_types = ["CO", "NO2", "CH4"]
boxplot_types = ["min", "quartile_025", "median", "quartile_075", "max"]
statistics_types = ["mean", "mode", "median", "variance", "zeroes", "non_zeroes"]
data_qualities = [0, 1]  # 0 also worst values | 1 is high quality
data_qualities_text = ["allQ", "highQ"]
plottings = ["weekend vs weekdays", "days vs days"]

location_name = location_names[0]
product_type = product_types[0]
statistcs_type = statistics_types[0]
d_q = 1
data_quality = data_qualities[d_q]
data_quality_text = data_qualities_text[d_q]
plotting = plottings[1]

directory_path = "./Data/" + location_name + "/" + product_type + "/Statistics/"
with open(directory_path + "days.json") as json_file:
    days = json.load(json_file)








if plotting == "weekend vs weekdays":
    fig = get_weekend_vs_weekdays_chart(days, date_start, date_end, data_quality, location_name, product_type)
if plotting == "days vs days":
    fig = get_days_vs_days_chart(days, date_start, date_end, data_quality, location_name, product_type)



"""directory_path_save = "./data/" + location_name + "/" + product_type + "/charts"
Path(directory_path).mkdir(parents=True, exist_ok=True)
file_name = location_name.strip() + "_" + str(product_type) + "_" + data_quality_text + "_" + date_start.strftime(
    "%Y-%m-%d") + "_" + date_end.strftime("%Y-%m-%d") + "_HISTOGRAM_WEEKEND_VS_WEEKDAYS"
print(file_name)
fig.write_html(directory_path_save + "/" + file_name + ".html")"""
