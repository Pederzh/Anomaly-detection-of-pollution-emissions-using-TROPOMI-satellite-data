import datetime
import json
from pathlib import Path
import plotly.express as px
import pandas as pd


date = datetime.datetime.now()
date_start = date.replace(year=2019, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=9, day=1, hour=0, minute=0, second=0, microsecond=0)
date_from = date_start
date_to = date_start

location_names = ["Bering Strait", "Sabetta Port"]
product_types = ["CO", "NO2", "CH4", "SO2"]
statistic_types = ["mean", "non_zeroes"]
special_statistic_type = statistic_types[1] #non_zeroes
qualities = [0, 1]  # 0 also worst values | 1 is high quality
qualities_text = ["lowQ", "highQ"]

product_days = {}

for l in range(len(location_names)):
    location_name = location_names[l]

    for p in range(len(product_types)):
        product_type = product_types[p]
        directory_path = "./Data/" + location_name + "/" + product_type + "/Statistics/"
        with open(directory_path + "days.json") as json_file:
            days = json.load(json_file)

        for q in range(len(qualities)):
            quality = qualities[q]
            df_array = []
            df = pd.DataFrame()
            for day_counter in range(int((date_end - date_start).days)):
                date_from = date_to + datetime.timedelta(days=day_counter)
                # setting date from string for the api call
                date_from_str = date_from.strftime("%Y-%m-%d")
                # setting date to string for the api call
                date_to_str = date_to.strftime("%Y-%m-%d")
                for i in range(len(statistic_types)):
                    value = days[date_from_str]["statistics"][statistic_types[i]][quality]
                    if statistic_types[i] == special_statistic_type:
                        value = value / days[date_from_str]["statistics"]["n_tot"][quality] * 10000
                    data = {"date": date_from_str,
                            "statistic_type": statistic_types[i],
                            "value": value
                            }
                    df_array.append(data)

            df = pd.DataFrame(df_array)

            fig = px.histogram(df, x="date", y="value", color="statistic_type", barmode="overlay", nbins=int((date_end - date_start).days))

            directory_path_save = "./data/" + location_name + "/" + product_type + "/charts"
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            file_name = location_name.strip() + "_" + product_type + "_" + str(statistic_types) + "_" + qualities_text[q] + "_" + date_start.strftime(
                "%Y-%m-%d") + "_" + date_end.strftime("%Y-%m-%d") + "_DISTPLOT"
            print(file_name)
            fig.write_html(directory_path_save + "/" + file_name + ".html")
