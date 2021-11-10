from flask import Flask, jsonify, send_file
import flask.scaffold
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
from flask_restful import Api, Resource, reqparse
from main import main_preparation, main_forecasting
import pickle
import numpy as np
import json
import datetime

app = Flask(__name__)
api = Api(app)

# Create parser for the payload data
# parser = reqparse.RequestParser()
# parser.add_argument('data')  # propietà che verrà parsata dalla request

class TropomiTest(Resource):
    def get(self):
        return "Working"

# Define how the api will respond to the post requests
class TropomiPreparation(Resource):
    def post(self):

        #TODO check body of the request

        # Default values
        date = datetime.datetime.now()
        date_start = date.replace(year=2021, month=1, day=1, hour=0, minute=0, second=0, microsecond=0) #number of training days
        date_end = date.replace(year=2021, month=12, day=3, hour=0, minute=0, second=0, microsecond=0)
        date_range = 30
        start_h = 0
        range_h = 24
        coordinates = [71.264765, 72.060155] #sabetta
        location_name = "Sabetta Port"
        product_type = "NO2"
        default_weights = {}
        client_id = '982de4f4-dade-4f98-9b49-4374cd896bb6'
        client_secret = '%p/,0Yrd&/mO%cdudUsby[>@]MB|2<rf1<NnXkZr'
        for i in range(16):
            default_weights[str(i)] = 1
        #-------

        main_preparation(date_start, date_end, start_h, range_h, coordinates, location_name, product_type, default_weights, date_range,
                         client_id, client_secret)

        content = {
            "message": "Preparation started, try calling /alerting to check if the system is ready for alerting",
            "status": 200,
        }

        return jsonify(content) # serialize it again so that they will be returned back to the application in a proper format.


class TropomiPredictor(Resource):
    def post(self):
        content = request.get_json()

        #TODO check body of the request

        # Default values
        date = datetime.datetime.now()
        date_start = date.replace(year=2021, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        date_requested = date.replace(year=2021, month=12, day=3, hour=0, minute=0, second=0, microsecond=0)
        date_range = 30
        range_alerting = 10
        location_name = "Sabetta Port"
        product_type = "NO2"
        #-------

        result = main_forecasting(product_type, location_name, date_start, date_requested, date_range, range_alerting)

        return jsonify(result) # serialize it again so that they will be returned back to the application in a proper format.


api.add_resource(TropomiPreparation, '/prepare')
api.add_resource(TropomiPredictor, '/alerting')
api.add_resource(TropomiTest, '/test')

# {
#     coordinates: {
#         lat:
#         lng:
#     },
#     value_actual: {
#         volume: (final val)
#         peak: (a)
#         attenuation: (b)
#     },
#     value_expected: {
#         volume: (final val)
#         peak: (a)
#         attenuation: (b)
#     },
#     status: //green, yellow, orange, red
#
# }

if __name__ == '__main__':
    # Load model
    print("start")
    #with open('model.pickle', 'rb') as f:
    #    model = pickle.load(f)

    app.run(debug=True)
