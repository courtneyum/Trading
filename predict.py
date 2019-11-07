import os; os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import model_from_json
from keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt
import joblib
import numpy as np

import dataHelper as dh
from Param import Param

def predict(file_number, compute_predictions=Param.compute_predictions):

    # load model from single file
    model = load_model(Param.best_model_filename + str(file_number) + ".h5")
    print("Loaded model from disk")

    # load scaler
    scaler = joblib.load(Param.output_scaler_filename)

    # load test data
    print("Fetching test data.")
    test_X = dh.get_testing_input(Param.filenames[file_number])
    test_Y = dh.get_testing_output(Param.filenames[file_number])

    # can't predict the first 10 timesteps
    test_Y = test_Y[range(Param.time_steps, test_Y.shape[0])]

    # make predictions
    print("Predicting the future.")
    if compute_predictions:
        predictions = np.zeros([test_X.shape[0] - Param.time_steps, Param.n_targets])
        for i in range(test_X.shape[0] - Param.time_steps):
            test_X_i = test_X[range(i, i + Param.time_steps)]
            predictions[i, :] = model.predict(test_X_i.reshape(1, 10, 8), batch_size=Param.batch_size)

        # scale values back
        predictions = scaler.inverse_transform(predictions)
        np.save(Param.predictions_filename + str(file_number), predictions)
    else:
        predictions = np.load(Param.predictions_filename)
    # END if

    # plot to compare predicted with actual
    
    n_out = Param.n_out
    for j in range(Param.n_out):
        fig, axs = plt.subplots(4)
        fig.suptitle('Time t' + str(j))
        for i in range(4):
            axs[i].plot(predictions[:, j*n_out+i], label='predicted')
            axs[i].plot(test_Y[:, j*n_out+i], label='actual')
            axs[i].legend()
            axs[i].set_title(Param.columns[i])
            axs[i].set(xlabel = 'Timestep', ylabel = 'Price')
        # END for
        plt.savefig("Plots/" + Param.price_fig_filename + str(file_number) + str(j) + ".png")
        print("Saved plot for model " + str(file_number) + " and timestep " + str(j))
        plt.show()
    # END for

def test(file_number):
    # load model from single file
    model = load_model(Param.best_model_filename)
    print("Loaded model from disk")

    # load scaler
    scaler = joblib.load(Param.output_scaler_filename)

    # load test data
    print("Fetching test data.")
    test_X = dh.get_testing_input(Param.filenames[file_number])
    test_Y = dh.get_testing_output(Param.filenames[file_number])

    # make predictions
    print("Predicting the future.")
    predictions = model.predict(test_X, batch_size=Param.batch_size)

    # scale values back
    predictions = scaler.inverse_transform(predictions)

    # plot to compare predicted with actual
    
    n_out = Param.n_out
    for j in range(Param.n_out):
        fig, axs = plt.subplots(4)
        fig.suptitle('Time t' + str(j))
        for i in range(4):
            axs[i].plot(predictions[:, j*n_out+i], label='predicted')
            axs[i].plot(test_Y[:, j*n_out+i], label='actual')
            axs[i].legend()
            axs[i].set_title(Param.columns[i])
            axs[i].set(xlabel = 'Timestep', ylabel = 'Price')
        # END for
        plt.savefig("Price" + str(file_number) + str(j) + ".png")
        print("Saved plot for model " + str(file_number) + " and timestep " + str(j))
        plt.show()
    # END for