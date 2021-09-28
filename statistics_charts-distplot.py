import datetime
import json
from pathlib import Path
import plotly.express as px
import pandas as pd

location_names = ["Bering Strait", "Sabetta Port"]
product_types = ["CO", "NO2", "CH4"]
statistic_types = ["min", "quartile_025", "median", "quartile_075", "max"]

location_name = location_names[1]
product_type = product_types[0]
statistic_type = statistic_types[4]

product_days = {
}

for i in range(len(product_types)):
    directory_path = "./Data/" + location_name + "/" + product_types[i] + "/Statistics/"
    with open(directory_path + "days.json") as json_file:
        days = json.load(json_file)
    product_days[product_types[i]] = days


date = datetime.datetime.now()
date_start = date.replace(year=2019, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=9, day=1, hour=0, minute=0, second=0, microsecond=0)
date_from = date_start
date_to = date_start

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

    for i in range(len(product_types)):
        data = {"date": date_from_str,
                "product_type": product_types[i],
                "value": product_days[product_types[i]][date_from_str]["box_plot"][statistic_type][1]}
        df_array.append(data)


print(df_array)

df = pd.DataFrame(df_array)
print(df)

fig = px.histogram(df, x="date", y="value", color="product_type", barmode="overlay", nbins = int((date_end - date_start).days))
fig.show()

directory_path_save = "./data/" + location_name + "/" + product_type + "/charts"
Path(directory_path).mkdir(parents=True, exist_ok=True)
file_name = location_name.strip() + "_" + product_type + "_" + date_start.strftime(
    "%Y-%m-%d") + "_" + date_end.strftime("%Y-%m-%d") + "_DISTPLOT";
fig.write_html(directory_path_save + "/" + file_name + ".html")
