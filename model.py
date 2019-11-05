import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import os; os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import model_from_json
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

import dataHelper as dh
from Param import Param

def create_model():
    # design network
    print("Compiling neural network.")
    model = Sequential()
    model.add(LSTM(units=20, batch_input_shape=(Param.batch_size, Param.time_steps, Param.n_features), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units=20))
    model.add(Dropout(0.5))
    model.add(Dense(20,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    optimizer = optimizers.Adam(lr=0.0001)
    model.compile(loss='mape', optimizer=optimizer)
    return model

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def model(file_number=Param.file_number):

    # set parameters
    path = Param.path
    filenames = Param.filenames

    time_steps = Param.time_steps
    batch_size = Param.batch_size

    n_in = Param.n_in
    n_out = Param.n_out

   # retrieve and format training data
    print("Fetching training data.")
    train_X = dh.get_input(path, filenames[file_number], dh.Datatype.TRAIN, n_in, n_out, time_steps, batch_size)
    train_Y = dh.get_output(path, filenames[file_number], dh.Datatype.TRAIN, n_in, n_out, time_steps, batch_size)

    # checkpoint
    filepath=Param.best_model_filename
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # These values help us determine where to split the data for training and validation
    num_batches = math.floor(train_X.shape[0]/batch_size)
    num_batches_val = math.floor(Param.validation_split*num_batches)
    num_batches_train = num_batches - num_batches_val
    #num_folds = math.ceil(num_batches/num_batches_val)

    # split data into training and validation sets
    val_X = train_X[num_batches_train*batch_size:, :, :]
    val_Y = train_Y[num_batches_train*batch_size:]

    train_X = train_X[0:num_batches_train*batch_size, :, :]
    train_Y = train_Y[0:num_batches_train*batch_size]

    # fit network
    model = create_model()
    history = model.fit(train_X, train_Y, epochs=100, batch_size=Param.batch_size, validation_data=(val_X, val_Y), verbose=2,
        callbacks=callbacks_list)

    # plot history
    plt.plot(history.history['loss'], label='mape')
    plt.plot(history.history['val_loss'], label='val_mape')
    plt.legend()
    plt.title("Training loss vs. Validation loss")
    plt.xlabel("Timestep")
    plt.ylabel("Mean absolute percentage error")
    plt.savefig("Loss" + str(file_number) + ".png")

    plt.show()

    # save model to file
    model.save(Param.model_filename)
    print("Saved model " + str(file_number) + " to disk")
# END model
