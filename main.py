from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

import datetime
import json
from pathlib import Path

from data_downloader_hours import main_downloader
from data_manager import main_processer
from peaks_manager import main_peak_finder
from plumes_manager import main_reconstructor
from alerting_manager import main_alerter



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




def main_forecasting(product_type, location_name, date_start, date_end, data_range, range_prediction):

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

    return responce















