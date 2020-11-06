import os
import sys

import json
import gym
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm

# Get prediction model
from core.model import Model

from GA.ga import GeneticNetworks

from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical

from core.market_env_v0 import MarketEnvironmentV0
from core.utils import *
from misc.networks import get_session

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import pathlib
import pandas as pd

# CNN
from cnn_prediction.cnn_data_loader import CNNDataLoader
from cnn_prediction.cnn_data_loader_buy_sell import CNNDataLoaderBuySell
from cnn_prediction.cnn_model import CNNModel
from core.technical_analysis import add_technical_indicators_with_intervals
import keras_metrics
import tensorflow.keras.backend as K

from flask import Flask
from flask import request
from flask import jsonify
app = Flask(__name__)

def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

class CNNModel:
    def __init__(self, model_folder):
        self.session = tf.Session()
        self.graph = tf.get_default_graph()
        self.model_folder = model_folder
        self.model = None
        # for some reason in a flask app the graph/session needs to be used in the init else it hangs on other threads
        with self.graph.as_default():
            with self.session.as_default():
                print("neural network initialised")

    def load(self, file_name=None):
        """
        :param file_name: [model_file_name, weights_file_name]
        :return:
        """
        with self.graph.as_default():
            with self.session.as_default():
                try:
                    model_name = file_name[0]
                    weights_name = file_name[1]

                    if model_name is not None:
                        # load the model
                        filepath = os.path.join(self.model_folder, model_name)
                        self.model = tf.keras.models.load_model(filepath, custom_objects={
                            "f1_score": f1_score,
                            "binary_precision": keras_metrics.precision(),
                            "binary_recall": keras_metrics.recall(),
                        })
                    if weights_name is not None:
                        # load the weights
                        weights_path = os.path.join(self.model_folder, weights_name)
                        self.model.load_weights(weights_path)
                    print("Neural Network loaded: ")
                    print('\t' + "Neural Network model: " + model_name)
                    print('\t' + "Neural Network weights: " + weights_name)
                    return True
                except Exception as e:
                    print(e)
                    return False

    def predict(self, x):
        with self.graph.as_default():
            with self.session.as_default():
                y = self.model.predict(x)
        return y

## Variables
configs = None
action_size = None
state_size = None
size = None
agent = None
cnn_cols = None
cols = None
model = None
graph = None

def load_config():
    global configs
    configs = json.load(open('config-EURUSD-30M_v3.0141.json', 'r'))

def load_hyperparameters():
    print(">>>>>>>>>>>>>>> ...Loading hyperparameters")
    global action_size
    global state_size 
    global size

    action_size = 3              # 3 possible actions: sit, buy, sell
    state_size = 15
    size = 347

    print(">>>>>>>>>>>>>>> Hyperparameters loaded")

def load_ga_agent():
    print(">>>>>>>>>>>>>>> ...Loading agent")
    global agent

    ga = GeneticNetworks(architecture=(state_size, ) + tuple(configs['agent']['layers']) + (action_size,),
                                    population_size=configs['agent']['params']['population_size'], 
                                    generations=configs['agent']['params']['generations'],
                                    episodes=configs['agent']['params']['episodes'], 
                                    mutation_variance=configs['agent']['params']['mutation_variance'],
                                    render_env=False,
                                    survival_ratio=configs['agent']['params']['survival_ratio'],
                                    verbose=True)
    ga.load_weights(configs['agent']['best_agent'])
    agent = ga.best_network
    print(">>>>>>>>>>>>>>> Agent loaded")

def load_cnn():
    print(">>>>>>>>>>>>>>> ...Loading CNN")
    global cnn_cols
    global cols
    global model
    
    model = CNNModel('saved_models')
    model.load([configs['model']['pre_trained_model'], configs['model']['pre_trained_weights']])

    # Get CNN cols
    cnn_cols = []
    for indicator in configs['model']['inputs']['technical_indicators']:
        for interval in configs['model']['inputs']['intervals']:
            cnn_cols.append(indicator + '_' + str(interval))
    img_size = len(configs['model']['inputs']['technical_indicators'])


    # Get cols
    cols = []
    # Add detrend OHLC
    for col in [a for a in configs['agent']['cols'] if configs['agent']['cols'][a]["detrend"]]:
        cols.append(col + '_detrend')
    for col in [a for a in configs['agent']['cols'] if not configs['agent']['cols'][a]["detrend"]]:
        cols.append(col)
    # Add technical indicators
    for indicator in configs['agent']['technical_indicators']:
        for interval in configs['agent']['intervals']:
            cols.append(indicator + '_' + str(interval))

    print(">>>>>>>>>>>>>>> CNN loaded")



    

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/action/<market_code>', methods = ['POST'])
def signal(market_code):
    data = request.json

    if market_code == "EURUSD":
        # Check input 
        if (not 'Close' in data) or (not 'Open' in data) or (not 'High' in data) or (not 'Low' in data):
            raise InvalidUsage('The request must have a valid Open, High, Low, Close values', status_code=400)
        if (len(data['Close']) < size) or (len(data['Open']) < size) or (len(data['High']) < size) or (len(data['Low']) < size):
            raise InvalidUsage('We only accept request if more than ' + str(size) + ' candles', status_code=400)


        df_ohlc = pd.DataFrame(data)
        # Adding info detrend
        for col in [a for a in configs['agent']['cols'] if configs['agent']['cols'][a]["detrend"]]:
            df_ohlc[col + '_detrend'] = df_ohlc.get([col]) - df_ohlc.get([col]).shift()

        # Get CNN prediction
        dataloader = CNNDataLoaderBuySell(
            df_ohlc,
            indicators=configs['model']['inputs']['technical_indicators'],
            intervals=configs['model']['inputs']['intervals'],
            window_size=10
        )
        x, _ = dataloader.get_train_data(cnn_cols, window_size=15, class_unbalacing=False)
        pred = model.predict(x)[-1]
        print("Got CNN prediction ", pred.shape)

        # Adding TA
        df_ohlc = add_technical_indicators_with_intervals(
            df_ohlc, 
            indicators=configs['agent']['technical_indicators'], 
            intervals=configs['agent']['intervals']
        )
        

        df_ohlc["CNNClassifier_hold"], df_ohlc["CNNClassifier_buy"], df_ohlc["CNNClassifier_sell"] = np.nan, np.nan, np.nan
        df_ohlc["CNNClassifier_hold"].iloc[-1] = pred[0]
        df_ohlc["CNNClassifier_buy"].iloc[-1] = pred[1]
        df_ohlc["CNNClassifier_sell"].iloc[-1] = pred[2]

        df_ohlc = df_ohlc.dropna()
        print(df_ohlc)
        state = df_ohlc.get(cols).values[-1]
        action = agent.act(state)
        print("Action ", action)
        
        return jsonify({ 'action': str(action) })

    else:
        raise InvalidUsage('Not valid market code', status_code=400)

if __name__ == '__main__':
    load_config()
    load_hyperparameters()
    load_ga_agent()
    load_cnn()

    app.run(debug=False, host='0.0.0.0')

