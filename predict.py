import os; os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import model_from_json
from keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt
import joblib
import numpy as np

import dataHelper as dh
from Param import Param

def predict(file_number=Param.file_number):
    path = Param.path
    filenames = Param.filenames

    # load model from single file
    model = load_model(Param.best_model_filename)
    print("Loaded model from disk")

    # load scaler
    scaler = joblib.load(Param.output_scaler_filename)

    # Choose these parameters so that we keep as much data as possible
    time_steps = Param.time_steps
    batch_size = Param.batch_size

    n_in = Param.n_in
    n_out = Param.n_out

    # load test data
    print("Fetching test data.")
    test_X = dh.get_input(path, filenames[file_number], dh.Datatype.TEST, n_in, n_out, time_steps, batch_size)
    test_Y = dh.get_output(path, filenames[file_number], dh.Datatype.TEST, n_in, n_out, time_steps, batch_size)

    # make predictions
    print("Predicting the future.")
    predictions = model.predict(test_X, batch_size=100)

    # scale values back
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

    # plot to compare predicted with actual
    plt.plot(predictions, label='predicted')
    plt.plot(test_Y, label='actual')
    plt.legend()
    plt.title("Predicted price vs. Actual price")
    plt.xlabel("Timestep")
    plt.ylabel("Price")
    plt.savefig("Price" + str(file_number) + ".png")
    print("Saved plot for model " + str(file_number))
    plt.show()