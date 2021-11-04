from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse, request
import pickle
import numpy as np
import json

app = Flask(__name__)
api = Api(app)

# Create parser for the payload data
# parser = reqparse.RequestParser()
# parser.add_argument('data')  # propietà che verrà parsata dalla request


# Define how the api will respond to the post requests
class TropomiPredictor(Resource):
    def post(self):
        print(request.is_json)
        content = request.get_json()
        content["status"] = "success"
        return jsonify(content) # serialize it again so that they will be returned back to the application in a proper format.


api.add_resource(TropomiPredictor, '/tropomiPredictor')

if __name__ == '__main__':
    # Load model
    print("start")
    #with open('model.pickle', 'rb') as f:
    #    model = pickle.load(f)

    app.run(debug=True)
