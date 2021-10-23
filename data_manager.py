import datetime
import io
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from PIL import Image
from numpy import array
import numpy as np
from scipy.interpolate import griddata
import statistics as stst
import numpy
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session


wieghts = {
    "0": 1,
    "1": 4,
    "2": 7,
    "3": 8,
    "4": 10,
    "5": 7,
    "6": 6,
    "7": 3,
    "8": 2,
    "9": 1,
}




# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                                   GENERIC FUNCTIONS
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def get_new_coordinates(lat, lon, distance_lat, distance_lon):
    lat_new = lat + (180 / math.pi) * (distance_lat / 6378137)
    lon_new = lon + (180 / math.pi) * (distance_lon / 6378137) / math.cos(math.pi/180*lat)
    return [lat_new, lon_new]

def get_new_longitude(lat, lon, distance_lon):
    return get_new_coordinates(lat, lon, 0, distance_lon)[1]

def get_new_latitude(lat, lon, distance_lat):
    return get_new_coordinates(lat, lon, distance_lat, 0)[0]

def get_bbox_coordinates_from_center(coordinates, distance):
    bbox_coordinates = []
    lat = coordinates[0]
    lon = coordinates[1]
    bbox_coordinates.append(get_new_longitude(lat, lon, -distance))
    bbox_coordinates.append(get_new_latitude(lat, lon, -distance))
    bbox_coordinates.append(get_new_longitude(lat, lon, distance))
    bbox_coordinates.append(get_new_latitude(lat, lon, distance))
    return bbox_coordinates

def open_json(directory_path):
    return get_json_content(directory_path)

def load_json(directory_path):
    return get_json_content(directory_path)

def read_json(directory_path):
    return get_json_content(directory_path)

def get_json_content(directory_path):
    with open(directory_path + "data.json") as json_file:
        data = json.load(json_file)
    return data

def get_specific_json_content(directory_path, json_file_name):
    with open(directory_path + json_file_name + ".json") as json_file:
        data = json.load(json_file)
    return data

def get_frequencies_from_matrix(matrix):
    my_list = get_list_from_matrix(matrix)
    return get_frequencies_from_list(my_list)

def get_frequencies_from_list(my_list):
    values = {}
    for i in range(len(my_list)):
        if my_list[i] != -1:
            if str(my_list[i]) not in values.keys():
                values[str(my_list[i])] = 0
            values[str(my_list[i])] += 1
    return values

def get_list_from_matrix(matrix):
    my_list = []
    for y in range(len(matrix)):
        my_list = my_list + matrix[y]
    return my_list

def remove_zeroes_from_sorted_list(sorted_list):
    while sorted_list[0] == -1:
        del sorted_list[0]
    return sorted_list





# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                                 FOR CREATING JSON FROM IMAGE
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def get_value_from_rgb( rgb):
    """
    [minVal, [0, 0, 0.5]],
    [minVal + 0.125 * diff, [0, 0, 1]],
    [minVal + 0.375 * diff, [0, 1, 1]],
    [minVal + 0.625 * diff, [1, 1, 0]],
    [minVal + 0.875 * diff, [1, 0, 0]],
    [maxVal, [0.5, 0, 0]]
    """
    if rgb[3] == 0: return -1
    precision = 1016  # given by the rgb composition
    minVal = 0
    maxVal = minVal + precision
    diff = maxVal - minVal
    # [0, 0, 1]
    if rgb[0] == 0 and rgb[1] == 0:
        return minVal + 0.125 * diff * (rgb[2]-128)/127
    # [0, 1, 1]
    if rgb[0] == 0 and rgb[2] == 255:
        min = 0.125
        max = 0.375
        rangeVal = max - min
        return (minVal + min * diff) + rangeVal * diff * rgb[1]/255
    # [1, 1, 0]
    if rgb[1] == 255:
        min = 0.375
        max = 0.625
        rangeVal = max - min
        return (minVal + min * diff) + rangeVal * diff * rgb[0] / 255
    # [1, 0, 0]
    if rgb[0] == 255:
        min = 0.625
        max = 0.875
        rangeVal = max - min
        return (minVal + min * diff) + rangeVal * diff * (255-rgb[1]) / 255
    # [0.5, 0, 0]
    min = 0.875
    max = 1
    rangeVal = max - min
    return (minVal + min * diff) + rangeVal * diff * (127-(rgb[0]-128)) / 127


def create_image_matrix(image):
    image_matrix = [[]]
    image_matrix.clear()
    for y in range(len(image)):
        image_matrix.append([])
        for x in range(len(image[y])):
            if image[y][x][3] == 0: image_matrix[y].append(-1)
            else: image_matrix[y].append(int(get_value_from_rgb(image[y][x])))
    return image_matrix


def create_dir_json_from_image(directory_path, image_type):
    file_json = {}
    if image_type == "unprocessed":
        for hours_counter in range(10):
            file_name = str(hours_counter)
            my_file = Path(directory_path + file_name + ".png")
            if my_file.is_file():
                image = Image.open(directory_path + file_name + ".png")
                image_array = array(image)
                file_json[str(hours_counter)] = create_image_matrix(image_array)
        with open(directory_path + "data.json", 'w') as outfile:
            json.dump(file_json, outfile)

def create_all_json_from_images(product_type, location_name, date_start, date_end, image_type):
    for day_counter in range(int((date_end - date_start).days)):
        date = date_start + datetime.timedelta(days=day_counter)
        print("at day " + date.strftime("%Y-%m-%d"))
        directory_path = "../data/" + product_type + "/" + location_name + "/images/"
        directory_path = directory_path + date.strftime("%Y") + "/" + date.strftime("%m") + "/"
        directory_path = directory_path + date.strftime("%d") + "/" + image_type + "/"
        my_path = Path(directory_path)
        if my_path.is_dir():
            create_dir_json_from_image(directory_path, image_type)





# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                                   FOR CREATING BALANCED JSON
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



def get_image_distribution(data):
    sorted_list = get_list_from_matrix(data)
    sorted_list.sort()
    sorted_list = remove_zeroes_from_sorted_list(sorted_list)
    percentile = 0.1
    n_tot = int(len(sorted_list) * percentile)
    return {
        "inf_out": sorted_list[round(len(sorted_list)*0.05)-1],
        "inf": sorted_list[round(len(sorted_list)*0.25)-1],
        "median": sorted_list[round(len(sorted_list)*0.5)-1],
        "sup": sorted_list[round(len(sorted_list)*0.75)-1],
        "sup_out": sorted_list[round(len(sorted_list)*0.95)-1],
    }

def get_image_dir_distribution(data):
    tot = 0
    stats = {}
    balancer = {
        "inf_out": 0,
        "inf": 0,
        "median": 0,
        "sup": 0,
        "sup_out": 0,
    }
    keys = list(data.keys())
    for i in range(len(keys)):
        stat = get_image_distribution(data[keys[i]])
        stats[keys[i]] = stat
    if "3" in keys or "4" in keys or "5" in keys:
        if "5" in keys:
            weight = 2
            s_keys = list(stats["5"].keys())
            for sk in range(len(s_keys)):
                balancer[s_keys[sk]] += stats["5"][s_keys[sk]] * weight
            tot += weight
        if "4" in keys:
            weight = 10
            s_keys = list(stats["4"].keys())
            for sk in range(len(s_keys)):
                balancer[s_keys[sk]] += stats["4"][s_keys[sk]] * weight
            tot += weight
        if "3" in keys:
            weight = 5
            s_keys = list(stats["3"].keys())
            for sk in range(len(s_keys)):
                balancer[s_keys[sk]] += stats["3"][s_keys[sk]] * weight
            tot += weight
    else:
        for i in range(len(keys)):
            s_keys = list(stats[keys[i]].keys())
            for sk in range(len(s_keys)):
                balancer[s_keys[sk]] += stats[keys[i]][s_keys[sk]]
            tot += 1
    s_keys = list(balancer.keys())
    for sk in range(len(s_keys)):
        balancer[s_keys[sk]] = round(balancer[s_keys[sk]] / tot, 3)
    stats["balancer"] = balancer
    return stats

def get_balanced_value(value, my_stats, new_stats):
    if value == -1: return -1
    if value <= my_stats["inf_out"]:
        my_min = 0
        my_max = my_stats["inf_out"]
        new_min = 0
        new_max = new_stats["inf_out"]
    if (value >= my_stats["inf_out"]) and (value <= my_stats["inf"]):
        my_min = my_stats["inf_out"]
        my_max = my_stats["inf"]
        new_min = new_stats["inf_out"]
        new_max = new_stats["inf"]
    if (value >= my_stats["inf"]) and (value <= my_stats["median"]):
        my_min = my_stats["inf"]
        my_max = my_stats["median"]
        new_min = new_stats["inf"]
        new_max = new_stats["median"]
    if (value >= my_stats["median"]) and (value <= my_stats["sup"]):
        my_min = my_stats["median"]
        my_max = my_stats["sup"]
        new_min = new_stats["median"]
        new_max = new_stats["sup"]
    if (value >= my_stats["sup"]) and (value <= my_stats["sup_out"]):
        my_min = my_stats["sup"]
        my_max = my_stats["sup_out"]
        new_min = new_stats["sup"]
        new_max = new_stats["sup_out"]
    if value >= my_stats["sup_out"]:
        my_min = my_stats["sup_out"]
        my_max = 1016
        new_min = new_stats["sup_out"]
        new_max = 1016
    my_range = my_max - my_min
    new_range = new_max - new_min
    if (my_range == 0): my_range = new_range
    value = value - my_min
    value = value * new_range / my_range
    value = value + new_min
    return value

def get_all_balanced_matrix(data, stats):
    keys = list(data.keys())
    for i in range(len(keys)):
        matrix = data[keys[i]]
        for y in range(len(matrix)):
            for x in range(len(matrix[y])):
                data[keys[i]][y][x] = int(get_balanced_value(matrix[y][x], stats[keys[i]], stats["balancer"]))
    return data

def save_json(directory_path, json_file, json_file_name):
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    with open(directory_path + json_file_name + ".json", 'w') as outfile:
        json.dump(json_file, outfile)


def create_balanced_dir_json(directory_path):
    my_path = Path(directory_path+"unprocessed/")
    if my_path.is_dir():
        my_file = Path(directory_path+"unprocessed/data.json")
        if my_file.is_file():
            data_set = get_json_content(directory_path+"unprocessed/")
            stats = get_image_dir_distribution(data_set)
            data_set = get_all_balanced_matrix(data_set, stats)
            save_json(directory_path+"balanced/", data_set, "data")


def create_all_balanced_json(product_type, location_name, date_start, date_end):
    for day_counter in range(int((date_end - date_start).days)):
        date = date_start + datetime.timedelta(days=day_counter)
        print("at day " + date.strftime("%Y-%m-%d"))
        directory_path = "../data/" + product_type + "/" + location_name + "/images/"
        directory_path = directory_path + date.strftime("%Y") + "/" + date.strftime("%m") + "/"
        directory_path = directory_path + date.strftime("%d") + "/"
        my_path = Path(directory_path)
        if my_path.is_dir():
            create_balanced_dir_json(directory_path)






# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                                 FOR CALCULATING INTERPOLATED IMAGES
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def get_interpolated_image(points, data):
    all_points = []
    for y in range(len(data)):
        for x in range(len(data[y])):
            all_points.append([y, x])
    values = []
    for i in range(len(points)):
        values.append(data[points[i][0]][points[i][1]])
    values = numpy.array(values)
    new_values = griddata(points, values, all_points, method='cubic')
    new_data = []
    count = 0
    for y in range(len(data)):
        new_data.append([])
        for x in range(len(data[y])):
            if numpy.isnan(new_values[count]):
                new_data[y].append(0)
            else:
                if new_values[count] < 0:
                    new_data[y].append(0)
                else:
                    new_data[y].append(round(new_values[count]))
            count += 1
    return new_data

def get_sub_points(data, side_len, n):
    points = []
    for y in range(len(data)):
        for x in range(len(data[y])):
            if (y-int(n/side_len)) % side_len == 0 and (x-(n-int(n/side_len)*side_len)) % side_len == 0:
                if (data[y][x] != None and data[y][x] != -1):
                    points.append([y, x])
    return points

def get_interpolated_data(data_set, side_len, what_to_return):
    data_mean = []
    final_data = []
    list_of_data_set = []
    tot = []
    for y in range(len(data_set)):
        data_mean.append([])
        tot.append([])
        final_data.append([])
        for x in range(len(data_set[y])):
            data_mean[y].append(0)
            tot[y].append(0)
            final_data[y].append(0)
    for i in range(side_len*side_len):
        points = get_sub_points(data_set, side_len, i)
        new_data_set = get_interpolated_image(points, data_set)
        for y in range(len(data_mean)):
            for x in range(len(data_mean[y])):
                if (y - int(i / side_len)) % side_len != 0 or (x - (i - int(i / side_len) * side_len)) % side_len != 0:
                    data_mean[y][x] += new_data_set[y][x]
                    tot[y][x] += 1
                else:
                    new_data_set[y][x] = -1
        list_of_data_set.append(new_data_set)
    for y in range(len(data_mean)):
        for x in range(len(data_mean[y])):
            data_mean[y][x] = data_mean[y][x]/tot[y][x]
    if what_to_return == "mean": return data_mean

    if what_to_return == "closer_to_mean":
        for y in range(len(data_mean)):
            for x in range(len(data_mean[y])):
                closer = 0
                for i in range(len(list_of_data_set)):
                    if abs(list_of_data_set[i][y][x] - data_mean[y][x]) < abs(closer - data_mean[y][x]) and list_of_data_set[i][y][x] != -1:
                        closer = list_of_data_set[i][y][x]
                final_data[y][x] = closer
        return final_data

    if what_to_return == "median":
        for y in range(len(data_mean)):
            for x in range(len(data_mean[y])):
                values_list = []
                for i in range(len(list_of_data_set)):
                    if list_of_data_set[i][y][x] != -1: values_list.append(list_of_data_set[i][y][x])
                median = stst.median(values_list)
                final_data[y][x] = median
        return final_data

    if what_to_return == "mode":
        for y in range(len(data_mean)):
            for x in range(len(data_mean[y])):
                values_list = []
                for i in range(len(list_of_data_set)):
                    if list_of_data_set[i][y][x] != -1: values_list.append(list_of_data_set[i][y][x])
                median = stst.mode(values_list)
                final_data[y][x] = median
        return final_data

    if what_to_return == "min":
        for y in range(len(data_mean)):
            for x in range(len(data_mean[y])):
                values_list = []
                for i in range(len(list_of_data_set)):
                    if list_of_data_set[i][y][x] != -1: values_list.append(list_of_data_set[i][y][x])
                min_val = min(values_list)
                final_data[y][x] = min_val
        return final_data

    if what_to_return == "max":
        for y in range(len(data_mean)):
            for x in range(len(data_mean[y])):
                values_list = []
                for i in range(len(list_of_data_set)):
                    if list_of_data_set[i][y][x] != -1: values_list.append(list_of_data_set[i][y][x])
                min_val = max(values_list)
                final_data[y][x] = min_val
        return final_data

def get_mean_of_interpolated_data(data_set, side_lens, mean_type):
    final_data = []
    tot = 0
    for y in range(len(data_set)):
        final_data.append([])
        for x in range(len(data_set[y])):
            final_data[y].append(0)
    for i in range(len(side_lens)):
        data_mean = get_interpolated_data(data_set, side_lens[i], mean_type)
        for y in range(len(final_data)):
            for x in range(len(final_data[y])):
                final_data[y][x] += data_mean[y][x]
        tot += 1
    for y in range(len(final_data)):
        for x in range(len(final_data[y])):
            final_data[y][x] = final_data[y][x] / tot
    return final_data

def get_final_interpolated_data(data_set):
    side_lens = [2, 3, 4, 5]
    mean_types = ["closer_to_mean", "median"]
    final_data = []
    for y in range(len(data_set)):
        final_data.append([])
        for x in range(len(data_set[y])):
            final_data[y].append(0)
    for i in range(len(mean_types)):
        data_tmp = get_mean_of_interpolated_data(data_set, side_lens, mean_types[i])
        # print_image_given_matrix(data_tmp)
        for y in range(len(final_data)):
            for x in range(len(final_data[y])):
                final_data[y][x] += data_tmp[y][x]
    for y in range(len(final_data)):
        for x in range(len(final_data[y])):
            final_data[y][x] = round(final_data[y][x] / len(mean_types))
    return final_data

def create_interpolated_dir_json(directory_path):
    my_path = Path(directory_path+"balanced/")
    if my_path.is_dir():
        my_file = Path(directory_path+"balanced/data.json")
        if my_file.is_file():
            data_set = get_json_content(directory_path+"balanced/")
            new_data_set = {}
            keys = list(data_set.keys())
            for i in range(len(keys)):
                new_data_set[keys[i]] = get_final_interpolated_data(data_set[keys[i]])
            save_json(directory_path+"interpolated/", new_data_set, "data")

def create_all_interpolated_json(product_type, location_name, date_start, date_end):
    for day_counter in range(int((date_end - date_start).days)):
        date = date_start + datetime.timedelta(days=day_counter)
        print("at day " + date.strftime("%Y-%m-%d"))
        directory_path = "../data/" + product_type + "/" + location_name + "/images/"
        directory_path = directory_path + date.strftime("%Y") + "/" + date.strftime("%m") + "/"
        directory_path = directory_path + date.strftime("%d") + "/"
        my_path = Path(directory_path)
        if my_path.is_dir():
            create_interpolated_dir_json(directory_path)






# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                                   FOR IMAGES WITH LOWED NOISE
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def remove_missing_data(list):
    list_new = list.copy()
    while None in list_new or -1 in list_new:
        if -1 in list_new:
            list_new.remove(-1)
        if None in list:
            list_new.remove(-1)
    return list_new

def get_sorted_list_from_matrix(matrix):
    list = []
    for y in range(len(matrix)):
        list += matrix[y]
    list = remove_missing_data(list)
    list.sort()
    return list

def get_data_with_lowed_noise(data):
    new_data = data.copy()
    list = get_sorted_list_from_matrix(data)
    min_range = stst.mean(list)
    for y in range(len(new_data)):
        for x in range(len(new_data[y])):
            if new_data[y][x] != None and new_data[y][x] != -1:
                if new_data[y][x] <= min_range: new_data[y][x] = min_range
                new_data[y][x] -= min_range
    return new_data

def create_lowed_dir_json(directory_path):
    my_path = Path(directory_path+"balanced/")
    if my_path.is_dir():
        my_file = Path(directory_path+"balanced/data.json")
        if my_file.is_file():
            data_set = get_json_content(directory_path+"balanced/")
            new_data_set = {}
            keys = list(data_set.keys())
            for i in range(len(keys)):
                new_data_set[keys[i]] = get_data_with_lowed_noise(data_set[keys[i]])
            save_json(directory_path+"lowed/", new_data_set, "data")

def create_all_lowed_json(product_type, location_name, date_start, date_end):
    for day_counter in range(int((date_end - date_start).days)):
        date = date_start + datetime.timedelta(days=day_counter)
        print("at day " + date.strftime("%Y-%m-%d"))
        directory_path = "../data/" + product_type + "/" + location_name + "/images/"
        directory_path = directory_path + date.strftime("%Y") + "/" + date.strftime("%m") + "/"
        directory_path = directory_path + date.strftime("%d") + "/"
        my_path = Path(directory_path)
        if my_path.is_dir():
            create_lowed_dir_json(directory_path)






# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                                   FOR IMAGE CREATION FROM JSON
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def get_rgb_values(value):
    """
    [minVal, [0, 0, 0.5]],
    [minVal + 0.125 * diff, [0, 0, 1]],
    [minVal + 0.375 * diff, [0, 1, 1]],
    [minVal + 0.625 * diff, [1, 1, 0]],
    [minVal + 0.875 * diff, [1, 0, 0]],
    [maxVal, [0.5, 0, 0]]
    """
    rgb = [0, 0, 0, 255]
    if (value == -1): return [0, 0, 0, 0]
    precision = 1016  # given by the rgb composition
    prop = value / precision
    # [0, 0, 1]
    if prop <= 0.125:
        min = 0
        max = 0.125
        rangeVal = max - min
        new_value = round((prop - min) / rangeVal * 127)
        rgb[2] = 128 + new_value
        return rgb
    # [0, 1, 1]
    if prop <= 0.375:
        min = 0.125
        max = 0.375
        rangeVal = max - min
        new_value = round((prop - min)/rangeVal * 255)
        rgb[1] = new_value
        rgb[2] = 255
        return rgb
    # [1, 1, 0]
    if prop <= 0.625:
        min = 0.375
        max = 0.625
        rangeVal = max - min
        new_value = round((prop - min)/rangeVal * 255)
        rgb[0] = new_value
        rgb[1] = 255
        rgb[2] = 255 - new_value
        return rgb
    # [1, 0, 0]
    if prop <= 0.875:
        min = 0.625
        max = 0.875
        rangeVal = max - min
        new_value = round((prop - min) / rangeVal * 255)
        rgb[0] = 1
        rgb[1] = 255 - new_value
        return rgb
    # [0.5, 0, 0]
    min = 0.875
    max = 1
    rangeVal = max - min
    new_value = round(127 - (prop - min) / rangeVal * 127)
    rgb[0] = 128 + new_value
    return rgb


def create_image_from_matrix(data):
    image = []
    for y in range(len(data)):
        image.append([])
        for x in range(len(data)):
            image[y].append(get_rgb_values(data[y][x]))
    return image

def save_image(directory_path, file_name, rgbt_matrix):
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    array = np.array(rgbt_matrix, dtype=np.uint8)
    image = Image.fromarray(array)
    image.save(directory_path + file_name + ".png", format="png")

def create_dir_images_from_json(directory_path):
    my_file = Path(directory_path + "data.json")
    if my_file.is_file():
        data_set = get_json_content(directory_path)
        keys = list(data_set.keys())
        for i in range(len(keys)):
            new_image = create_image_from_matrix(data_set[keys[i]])
            save_image(directory_path, keys[i], new_image)


def create_images_from_all_json(product_type, location_name, date_start, date_end, image_type):
    for day_counter in range(int((date_end - date_start).days)):
        date = date_start + datetime.timedelta(days=day_counter)
        print("at day " + date.strftime("%Y-%m-%d"))
        directory_path = "../data/" + product_type + "/" + location_name + "/images/"
        directory_path = directory_path + date.strftime("%Y") + "/" + date.strftime("%m") + "/"
        directory_path = directory_path + date.strftime("%d") + "/" + image_type + "/"
        my_path = Path(directory_path)
        if my_path.is_dir():
            create_dir_images_from_json(directory_path)







# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                                   FOR CALCULATING MEAN JSON
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def create_dir_mean_from_json(directory_path):
    my_file = Path(directory_path + "data.json")
    mean_matrix = []
    tot_matrix = []
    if my_file.is_file():
        data_set = get_json_content(directory_path)
        keys = list(data_set.keys())
        for i in range(len(keys)):
            for y in range(len(data_set[keys[i]])):
                if len(mean_matrix) == y:
                    mean_matrix.append([])
                    tot_matrix.append([])
                for x in range(len(data_set[keys[i]][y])):
                    if len(mean_matrix[y]) == x:
                        mean_matrix[y].append(0)
                        tot_matrix[y].append(0)
                    if data_set[keys[i]] != -1:
                        mean_matrix[y][x] += wieghts[keys[i]] * data_set[keys[i]][y][x]
                        tot_matrix[y][x] += wieghts[keys[i]]
        for y in range(len(mean_matrix)):
            for x in range(len(mean_matrix[y])):
                mean_matrix[y][x] = round(mean_matrix[y][x] / tot_matrix[y][x])
        save_json(directory_path, mean_matrix, "mean")

def create_all_mean_json(product_type, location_name, date_start, date_end, image_type):
    for day_counter in range(int((date_end - date_start).days)):
        date = date_start + datetime.timedelta(days=day_counter)
        print("at day " + date.strftime("%Y-%m-%d"))
        directory_path = "../data/" + product_type + "/" + location_name + "/images/"
        directory_path = directory_path + date.strftime("%Y") + "/" + date.strftime("%m") + "/"
        directory_path = directory_path + date.strftime("%d") + "/" + image_type + "/"
        my_path = Path(directory_path)
        if my_path.is_dir():
            create_dir_mean_from_json(directory_path)




def create_dir_image_from_json(directory_path, json_file_name):
    my_file = Path(directory_path + json_file_name + ".json")
    if my_file.is_file():
        data_set = get_specific_json_content(directory_path, json_file_name)
        new_image = create_image_from_matrix(data_set)
        save_image(directory_path, json_file_name, new_image)


def create_images_from_all_mean_json(product_type, location_name, date_start, date_end, image_type):
    for day_counter in range(int((date_end - date_start).days)):
        date = date_start + datetime.timedelta(days=day_counter)
        print("at day " + date.strftime("%Y-%m-%d"))
        directory_path = "../data/" + product_type + "/" + location_name + "/images/"
        directory_path = directory_path + date.strftime("%Y") + "/" + date.strftime("%m") + "/"
        directory_path = directory_path + date.strftime("%d") + "/" + image_type + "/"
        my_path = Path(directory_path)
        if my_path.is_dir():
            create_dir_image_from_json(directory_path, "mean")




























# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                       MAIN
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

values = {
    "product_types": ["NO2", "CO", "CH4", "SO2"],
    "locations_name": ["Bering Strait", "Sabetta Port"],
    "minQas": ["high", "all"],
    "image_types": ["unprocessed", "balanced", "lowed", "interpolated", ]
}
date = datetime.datetime.now()
date_start = date.replace(year=2021, month=5, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=6, day=1, hour=0, minute=0, second=0, microsecond=0)

product_type = values["product_types"][0]
location_name = values["locations_name"][1]
minQa = values["minQas"][1]
image_type = values["image_types"][2]

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#           TO CREATE ALL JSON FILES
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#create_all_json_from_images(product_type, location_name, date_start, date_end, image_type)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#       TO CREATE BALANCED JSON FILES
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#create_all_balanced_json(product_type, location_name, date_start, date_end)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   TO CREATE ALL IMAGES FROM BALANCED JSON
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#create_images_from_all_json(product_type, location_name, date_start, date_end, "balanced")

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#               JSON MEAN OF IMAGES
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#create_all_mean_json(product_type, location_name, date_start, date_end, "balanced")

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   TO CREATE ALL IMAGES FROM ALL MEAN JSON
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#create_images_from_all_mean_json(product_type, location_name, date_start, date_end, "balanced")



# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#       TO CREATE LOWED JSON FILES
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#create_all_lowed_json(product_type, location_name, date_start, date_end)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   TO CREATE ALL IMAGES FROM LOWED JSON
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#create_images_from_all_json(product_type, location_name, date_start, date_end, "lowed")

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#               JSON MEAN OF IMAGES
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#create_all_mean_json(product_type, location_name, date_start, date_end, "lowed")

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   TO CREATE ALL IMAGES FROM ALL MEAN JSON
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#create_images_from_all_mean_json(product_type, location_name, date_start, date_end, "lowed")








# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#       TO CREATE INTERPOLATED JSON FILES
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
create_all_interpolated_json(product_type, location_name, date_start, date_end)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   TO CREATE ALL IMAGES FROM INTERPOLATED JSON
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
create_images_from_all_json(product_type, location_name, date_start, date_end, "interpolated")

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#               JSON MEAN OF IMAGES
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
create_all_mean_json(product_type, location_name, date_start, date_end, "interpolated")

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   TO CREATE ALL IMAGES FROM ALL MEAN JSON
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
create_images_from_all_mean_json(product_type, location_name, date_start, date_end, "interpolated")