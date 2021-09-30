import datetime
import json
from pathlib import Path
import plotly.graph_objects as go

location_names = ["Bering Strait", "Sabetta Port"]
product_types = ["CO", "NO2", "CH4", "SO2"]
qualities = [0, 1]  # 0 also worst values | 1 is high quality
qualities_text = ["lowQ", "highQ"]

for l in range(len(location_names)):
    for p in range(len(product_types)):
        for q in range(len(qualities)):

            location_name = location_names[l]
            product_type = product_types[p]
            quality = qualities[q]

            directory_path = "./Data/" + location_name + "/" + product_type + "/Statistics/"

            with open(directory_path + "days.json") as json_file:
                days = json.load(json_file)

            date = datetime.datetime.now()
            date_start = date.replace(year=2019, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            date_end = date.replace(year=2021, month=9, day=1, hour=0, minute=0, second=0, microsecond=0)
            date_from = date_start
            date_to = date_start

            fig = go.Figure()

            time_sp = 1
            for day_counter in range(int((date_end - date_start).days)):
                date_from = date_to
                date_to = date_to + datetime.timedelta(days=time_sp)
                # setting date from string for the api call
                date_from_str = date_from.strftime("%Y-%m-%d")
                # setting date to string for the api call
                date_to_str = date_to.strftime("%Y-%m-%d")

                fig.add_trace(go.Box(y=[days[date_from_str]["box_plot"]["min"][quality],
                                        days[date_from_str]["box_plot"]["quartile_010"][quality],
                                        days[date_from_str]["box_plot"]["quartile_025"][quality],
                                        days[date_from_str]["box_plot"]["median"][quality],
                                        days[date_from_str]["box_plot"]["quartile_075"][quality],
                                        days[date_from_str]["box_plot"]["quartile_090"][quality],
                                        days[date_from_str]["box_plot"]["max"][quality]], name=date_from_str))

            #fig.show()

            directory_path_save = "./data/" + location_name + "/" + product_type + "/charts"
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            file_name = location_name.strip() + "_" + product_type + "_" + qualities_text[q] + "_" + date_start.strftime(
                "%Y-%m-%d") + "_" + date_end.strftime("%Y-%m-%d") + "_BOXPLOT"
            print(file_name)
            fig.write_html(directory_path_save + "/" + file_name + ".html")
