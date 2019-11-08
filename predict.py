import os; os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import model_from_json
from keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt
import joblib
import numpy as np
from pathlib import Path

import dataHelper as dh
from Param import Param

def predict(file_number, compute_predictions=Param.compute_predictions):

    # load model from single file
    model = load_model(str(Path(Param.models_dir) / (Param.best_model_filename + str(file_number) + ".h5")))
    print("Loaded model from disk")

    # load scaler
    scaler = joblib.load(str(Path(Param.models_dir) / Param.output_scaler_filename))

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
        np.save(str(Path(Param.data_dir) / (Param.predictions_filename + str(file_number))), predictions)
    else:
        predictions = np.load(str(Path(Param.data_dir) / (Param.predictions_filename + str(file_number) + ".npy")))
        print("Loaded predictions from disk")
    # END if

    # evaluate accuracy
    accuracy = evaluate_predictions(predictions, test_Y)
    print("Prediction accuracy for model " + str(file_number) + ": " + str(accuracy))

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
        plt.savefig(str(Path("Plots/") / (Param.price_fig_filename + str(file_number) + str(j) + ".png")))
        print("Saved plot for model " + str(file_number) + " and timestep " + str(j))

        if Param.show_plots:
            plt.show()
        # END if
    # END for
    return predictions
# END predict

def evaluate_predictions(predictions, test_Y):
    # confusion_matrix[0,0] = number of true positives (we guessed that price would increase and it did)
    # confusion_matrix[0,1] = number of false negatives (we guessed that prices would decrease and it didn't)
    # confusion_matrix[1,0] = number of false positives (we guessed that price would increase and it didn't)
    # confusion_matrix[1,1] = number of true negatives (we guessed that price would decrease and it did)
    confusion_matrix = np.zeros([2,2, Param.n_targets])
    accuracy = np.zeros([Param.n_targets, 1])
    for i in range(1, len(predictions)):
        delta_pred = predictions[i] - predictions[i-1]
        delta_y = test_Y[i] - test_Y[i-1]
        for j in range(Param.n_targets):
            if delta_pred[j] > 0 and delta_y[j] > 0:
                confusion_matrix[0,0,j] += 1
            elif delta_pred[j] < 0 and delta_y[j] >= 0:
                confusion_matrix[0,1,j] += 1
            elif delta_pred[j] > 0 and delta_y[j] <= 0:
                confusion_matrix[1,0,j] += 1
            elif delta_pred[j] < 0 and delta_y[j] < 0:
                confusion_matrix[1,1,j] += 1
            # END if
        # END for
    # END for

    for i in range(Param.n_targets):
        accuracy[i] = (confusion_matrix[0,0,i] + confusion_matrix[1,1,i])/(confusion_matrix[0,0,i] + confusion_matrix[1,1,i] + confusion_matrix[0,1,i] + confusion_matrix[1,0,i])
    # END for
    return accuracy
# END evaluate_predictions