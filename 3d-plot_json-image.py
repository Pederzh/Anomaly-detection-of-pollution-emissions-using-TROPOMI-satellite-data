import plotly.graph_objects as go
import numpy as np
import json

directory_path = "./data/thesis-data/"
days = ["20210505", "20210402"]
types = ["1_image-matrix", "2_interpolated", "3_gauss"]

for i in range(len(days)):
    for k in range(len(types)):
        file_name = days[i] + "_" + types[k]
        with open(directory_path + file_name + ".json") as json_file:
            data_set = json.load(json_file)

        matrix = np.flipud(data_set)

        z1 = np.array(matrix)

        fig = go.Figure(data=[
            go.Surface(z=z1),
        ])

        print(file_name)
        fig.write_html(directory_path + "3D_plot" + file_name + ".html")
        fig.show()