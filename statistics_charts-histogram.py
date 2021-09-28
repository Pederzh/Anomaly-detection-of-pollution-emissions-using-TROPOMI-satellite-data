import datetime
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

location_names = ["Bering Strait", "Sabetta Port"]
product_types = ["CO", "NO2", "CH4"]
statistic_types = ["min", "quartile_025", "median", "quartile_075", "max"]

location_name = location_names[0]
product_type = product_types[0]
statistic_type = statistic_types[2]

product_days = {
}

for i in range(len(product_types)):
    directory_path = "./Data/" + location_name + "/" + product_types[i] + "/Statistics/"
    with open(directory_path + "days.json") as json_file:
        days = json.load(json_file)
    product_days[product_types[i]] = days

date = datetime.datetime.now()
date_start = date.replace(year=2021, month=8, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=8, day=3, hour=0, minute=0, second=0, microsecond=0)
date_from = date_start
date_to = date_start

df_array = {
    "CO": [],
    "NO2": [],
    "CH4": [],
}

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
                "value": product_days[product_types[i]][date_from_str]["box_plot"][statistic_type][0] }
        df_array[product_types[i]].append(data)

fig_final = go.Figure()
for i in range(len(product_types)):
    df = pd.DataFrame(df_array[product_types[i]])
    #print(df)
    print(list(df["value"]))
    #fig[product_types[i]] = go.Histogram(x=df["date"], y=df["value"])

fig_final.add_trace(go.Histogram(x=['a', 'b', 'c']))
fig_final.add_trace(go.Histogram(x=['a', 'b', 'c']))
# Overlay both histograms
fig_final.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig_final.update_traces(opacity=0.75)
fig_final.show()


directory_path_save = "./data/" + location_name + "/" + product_type + "/charts"
Path(directory_path).mkdir(parents=True, exist_ok=True)
file_name = location_name.strip() + "_" + product_type + "_" + date_start.strftime(
    "%Y-%m-%d") + "_" + date_end.strftime("%Y-%m-%d") + "_DISTPLOT";
fig.write_html(directory_path_save + "/" + file_name + ".html")
