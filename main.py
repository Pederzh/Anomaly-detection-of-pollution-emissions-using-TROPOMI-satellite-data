from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

import datetime
import json
from pathlib import Path
import math
import matplotlib.pyplot as plt
from PIL import Image
import io

from data_downloader_hours import main_downloader
from data_manager import main_processer
from peaks_manager import main_peak_finder
from plumes_manager import main_reconstructor
from alerting_manager import main_alerter



def gauss_value(parameters, point):
    A = parameters[0]
    B = parameters[1]
    y = point[0]
    x = point[1]
    return (A * pow(math.e, -B * (pow(x, 2) + pow(y, 2))))

def get_gaussian_parameters(volume):
    A = pow(volume, 2 / 3) / pow(math.pi, 2 / 3)
    if A == 0: B = 1
    else: B = 1 / (pow(A, 1 / 2))
    return [A, B, volume]

def create_gaussian_image_w_parameters(image, parameters, point):
    gaussian_image = []
    for y in range(len(image)):
        gaussian_image.append([])
        for x in range(len(image[y])):
            gauss_point = [
                abs(point[0] - y),
                abs(point[1] - x)
            ]
            gaussian_image[y].append(round(gauss_value(parameters, gauss_point)))
    return gaussian_image

def create_gaussian_image(image, volume, point):
    parameters = get_gaussian_parameters(volume)
    return create_gaussian_image_w_parameters(image, parameters, point)

def get_standard_rgba_values(value):
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
        rgb[0] = 255
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
            image[y].append(get_standard_rgba_values(data[y][x]))
    return image

def print_image_given_matrix(matrix):
    image = create_image_from_matrix(matrix)
    plt.imshow(image)
    plt.show()

def get_json_content_w_name(directory_path, name):
    with open(directory_path + name + ".json") as json_file:
        data = json.load(json_file)
    return data

def get_ranges_to_download(date_start, date_end, location_name, product_type):
    date_ranges = []
    date_range = {}
    for day_counter in range(int((date_end - date_start).days) - 1):
        date_act = date_start + datetime.timedelta(days=day_counter)
        date_str = date_act.strftime("%Y-%m-%d")
        date_str = date_str.split("-")
        directory_path = "./Data/" + product_type + "/" + location_name + "/images/"
        directory_path += date_str[0] + "/" + date_str[1] + "/" + date_str[2] + "/unprocessed/"
        my_path = Path(directory_path)
        if not my_path.is_dir():
            if "date_start" not in date_range.keys():
                date_range["date_start"] = date_act
        else:
            if "date_start" in date_range.keys():
                date_range["date_end"] = date_act
                date_range["days"] = (date_range["date_end"] - date_range["date_start"]).days
                date_ranges.append(date_range)
                date_range = {}
    if "date_start" in date_range.keys():
        date_range["date_end"] = date_end
        date_range["days"] = (date_range["date_end"] - date_range["date_start"]).days
        date_ranges.append(date_range)
    return date_ranges

def get_ranges_to_process(date_start, date_end, location_name, product_type):
    date_ranges = []
    date_range = {}
    for day_counter in range(int((date_end - date_start).days) - 1):
        date_act = date_start + datetime.timedelta(days=day_counter)
        date_str = date_act.strftime("%Y-%m-%d")
        date_str = date_str.split("-")
        directory_path = "./Data/" + product_type + "/" + location_name + "/images/"
        directory_path += date_str[0] + "/" + date_str[1] + "/" + date_str[2] + "/lowed/"
        my_file = Path(directory_path + "data.json")
        if not my_file.is_file():
            if "date_start" not in date_range.keys():
                date_range["date_start"] = date_act
        else:
            if "date_start" in date_range.keys():
                date_range["date_end"] = date_act
                date_range["days"] = (date_range["date_end"] - date_range["date_start"]).days
                date_ranges.append(date_range)
                date_range = {}
    if "date_start" in date_range.keys():
        date_range["date_end"] = date_end
        date_range["days"] = (date_range["date_end"] - date_range["date_start"]).days
        date_ranges.append(date_range)
    return date_ranges

def main_preparation(date_start, date_end, start_h, range_h, coordinates, location_name, product_type, range_wieghts,
                     range_for_mean, cliend_id, client_secret):

    # DOWNLOADING IMAGES
    ranges_to_download = get_ranges_to_download(date_start, date_end, location_name, product_type)
    for to_download in ranges_to_download:
        main_downloader(to_download["date_start"], to_download["date_end"], start_h, range_h, coordinates,
                        location_name, product_type, cliend_id, client_secret)

    # PROCESSING IMAGES
    ranges_to_process = get_ranges_to_process(date_start, date_end, location_name, product_type)
    for to_process in ranges_to_process:
        main_processer(product_type, location_name, to_process["date_start"], to_process["date_end"], range_wieghts)

    # PEAK FINDING AND GROTE
    if len(ranges_to_process) > 0:
        main_peak_finder(product_type, location_name, date_start, date_end, range_for_mean)
        main_reconstructor(product_type, location_name, date_start, date_end, range_for_mean)




def main_alerting(product_type, location_name, date_start, date_end, data_range, range_prediction):

    directory_path = "./Data/" + product_type + "/" + location_name + "/range_data/"
    directory_path = directory_path + str(data_range)

    # checking peaks file
    my_file = Path(directory_path + "/peaks/peaks.json")
    if not my_file.is_file(): return {"error": "peaks file not found, please process"}
    peaks = get_json_content_w_name(directory_path + "/peaks/", "peaks")
    if len(peaks) == 0: return {"error": "no peaks found"}

    responce = {}
    for peak in peaks:

        # checking GROTE file
        my_file = Path(directory_path + "/gaussian_shapes/peak_" + str(peak["id"])+ "/parameters.json")
        if not my_file.is_file(): return {"error": "GROTE file not found, please process"}
        params = get_json_content_w_name(directory_path + "/gaussian_shapes/peak_" + str(peak["id"])+ "/", "parameters")
        if len(list(params.keys())) == 0: return {"error": "no parameter found, please process"}
        if len(list(params.keys())) < 100 or (date_end - date_start).days < 100: return {"error": "there is too little data"}
        found = False
        for i in range(range_prediction):
            new_data = date_end - datetime.timedelta(days=i + 1)
            if new_data.strftime("%Y-%m-%d") in params: found = True
        if not found: return {"error": "lack of data to forecast, please process"}
        # getting the responce
        res = main_alerter(product_type, location_name, date_start, date_end, data_range, peak["id"], 2, range_prediction)
        responce[str(peak["id"])] = res

    responce_img = []
    for y in range(100):
        responce_img.append([])
        for x in range(100):
            responce_img[y].append(0)

    for peak in peaks:
        img = responce[str(peak["id"])]["forecasted_value"]["GROTE_image"]
        for y in range(100):
            for x in range(100):
                responce_img[y][x] += responce

    return responce

def main_processed_image(product_type, location_name, date_start, date_end, peaks_sensing_period):
    dir_path = "./Data/" + product_type + "/" + location_name + "/range_data/" + str(peaks_sensing_period) + "/"
    my_file = Path(dir_path + "peaks/peaks.json")
    if not my_file.is_file(): return None
    peaks = get_json_content_w_name(dir_path + "peaks" + "/", "peaks")
    final_img = []
    for y in range(100):
        final_img.append([])
        for x in range(100):
            final_img[y].append(0)
    for peak in peaks:
        new_path = dir_path + "gaussian_shapes/peak_" + str(peak["id"]) + "/"
        my_file = Path(new_path + "parameters.json")
        if not my_file.is_file(): return None
        gaus_params = get_json_content_w_name(new_path, "parameters")
        mean_img = []
        tot = 0
        for y in range(100):
            mean_img.append([])
            for x in range(100):
                mean_img[y].append(0)
        for day_counter in range(int((date_end - date_start).days)):
            date_act = date_start + datetime.timedelta(days=day_counter)
            date_str = date_act.strftime("%Y-%m-%d")
            if date_str in gaus_params:
                tot += 1
                img = []
                for y in range(100):
                    img.append([])
                    for x in range(100):
                        img[y].append(0)
                gaus_image = create_gaussian_image(img, gaus_params[date_str][2], peak["point"])
                for y in range(100):
                    for x in range(100):
                        mean_img[y][x] += gaus_image[y][x]
        if tot == 0: return None
        for y in range(100):
            for x in range(100):
                final_img[y][x] += mean_img[y][x] / tot
    image = Image.fromarray(final_img)
    # create file-object in memory
    file_object = io.BytesIO()
    # write PNG in file-object
    image.save(file_object, 'PNG')
    # move to beginning of file so send_file() it will read from start
    file_object.seek(0)
    return file_object

