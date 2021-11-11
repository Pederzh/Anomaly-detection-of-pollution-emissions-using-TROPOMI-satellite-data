from flask import Flask, jsonify, send_file
import flask.scaffold

flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
from flask_restful import Api, Resource, reqparse
from main import main_preparation, main_forecasting
import datetime

app = Flask(__name__)
api = Api(app)

# Create parser for the payload data
tropomiPreparationParser = reqparse.RequestParser()
tropomiPreparationParser.add_argument('lat', type=float, required=True, help="Parameter 'lat' cannot be blank")
tropomiPreparationParser.add_argument('lng', type=float, required=True, help="Parameter 'lng' cannot be blank")
tropomiPreparationParser.add_argument('date', required=True, help="Parameter 'date' cannot be blank")
tropomiPreparationParser.add_argument('sensing_period', type=int, default=375)
tropomiPreparationParser.add_argument('product_type', default='NO2', choices=('NO2', 'CH4', 'CO'),
                                      help="Please provide a valid value between: 'NO2', 'CH4', 'CO'")
tropomiPreparationParser.add_argument('sentinel_hub_client_id', required=True,
                                      help="Parameter 'sentinel_hub_client_id' cannot be blank, get OAuth keys from Sentinel Hub")
tropomiPreparationParser.add_argument('sentinel_hub_client_secret', required=True,
                                      help="Parameter 'sentinel_hub_client_secret' cannot be blank, get OAuth keys from Sentinel Hub")
tropomiPreparationParser.add_argument('peaks_sensing_period', type=int, default=30)
tropomiPreparationParser.add_argument('sensing_start_hours', type=int, default=0)
tropomiPreparationParser.add_argument('sensing_range_hours', type=int, default=24)

tropomiAlertingParser = reqparse.RequestParser()
tropomiAlertingParser.add_argument('lat', type=float, required=True, help="Parameter 'lat' cannot be blank")
tropomiAlertingParser.add_argument('lng', type=float, required=True, help="Parameter 'lng' cannot be blank")
tropomiAlertingParser.add_argument('date', required=True, help="Parameter 'date' cannot be blank")
tropomiAlertingParser.add_argument('sensing_period', type=int, default=375)
tropomiAlertingParser.add_argument('product_type', default='NO2', choices=('NO2', 'CH4', 'CO'),
                                   help="Please provide a valid value between: 'NO2', 'CH4', 'CO'")
tropomiAlertingParser.add_argument('peaks_sensing_period', type=int, default=30)
tropomiAlertingParser.add_argument('range_alerting', type=int, default=10)


# Define how the api will respond to the post requests
class TropomiPreparation(Resource):
    def post(self):

        args = tropomiPreparationParser.parse_args()
        print(args)

        # --- Compute dates ---
        try:
            date = datetime.datetime.fromisoformat(args["date"])
        except ValueError:
            return 'Invalid date format', 400

        d = datetime.datetime.now()
        date_start = date - datetime.timedelta(days=args["sensing_period"])
        if date_start.year < 2021:
            date_start = d.replace(year=2021, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

        # --- Compute location ---
        coordinates = [args["lat"], args["lng"]]
        location_name = "[" + str(coordinates[0]) + "_" + str(coordinates[1]) + "]"  # DEFAULT

        default_weights = {}
        for i in range(16):
            default_weights[str(i)] = 1
        # -------


        main_preparation(date_start, date, args["sensing_start_hours"], args["sensing_range_hours"],
                         coordinates, location_name,
                         args["product_type"], default_weights, args["peaks_sensing_period"],
                         args["sentinel_hub_client_id"], args["sentinel_hub_client_secret"])

        content = {
            "message": "Preparation started, try calling /alerting to check if the system is ready for alerting",
            "status": 200,
        }

        # serialize it again so that they will be returned back to the application in a proper format.
        return jsonify(content)


class TropomiPredictor(Resource):
    def post(self):
        args = tropomiAlertingParser.parse_args()

        # --- Compute dates ---
        try:
            date = datetime.datetime.fromisoformat(args["date"])
        except ValueError:
            return 'Invalid date format', 400

        d = datetime.datetime.now()
        date_start = date - datetime.timedelta(days=args["sensing_period"])
        if date_start.year < 2021:
            date_start = d.replace(year=2021, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

        # --- Compute location ---
        coordinates = [args["lat"], args["lng"]]
        location_name = "[" + str(coordinates[0]) + "_" + str(coordinates[1]) + "]"  # DEFAULT
        # -------

        result = main_forecasting(args["product_type"], location_name, date_start, date,
                                  args["peaks_sensing_period"], args["range_alerting"])

        # serialize it again so that they will be returned back to the application in a proper format.
        return jsonify(result)


api.add_resource(TropomiPreparation, '/prepare')
api.add_resource(TropomiPredictor, '/alerting')

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
    # with open('model.pickle', 'rb') as f:
    #    model = pickle.load(f)

    app.run(debug=True)
