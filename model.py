import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math

import os; os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint
from keras import optimizers

class Datatype:
    TRAIN = "train"
    TEST = "test"

def get_data(path, filename, type, n_in, n_out):
    # retrieve data and filter out unwanted columns
    filename = path + "smoothed_" + type + filename
    df = pd.read_csv(filename)
    df = df.iloc[:, 0:5]

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = df
    df_scaled[columns[0:4]] = scaler.fit_transform(df[columns[0:4]])

    # Reframe as a supervised learning problem
    df_reframed = series_to_supervised(df_scaled, n_in, n_out)

    return df_reframed
#END get_data

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    	data: Sequence of observations as a list or NumPy array.
    	n_in: Number of lag observations as input (X).
    	n_out: Number of observations as output (y).
    	dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    	Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = data
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
#END series_to_supervised

# BEGIN
path = "K:/My Drive/School/DataScienceComp/Data/"
filenames = ["DataBAX.csv", "DataCGB.csv", "DataCURA.csv", "DataSXF.csv"]
columns = ["open", "high", "low", "close", "time"]

# Choose these parameters so that we keep as much data as possible
time_steps = 10
batch_size = 100

n_in = 2
n_out = 1

#retrieve and format training data
df_train = get_data(path, filenames[0], Datatype.TRAIN, n_in, n_out)
print(df_train.head(10))

#retrieve and format test data
df_test = get_data(path, filenames[0], Datatype.TEST, n_in, n_out)
print(df_test.head(10))

#split data into input and output
df_train_X = df_train.iloc[:, 0:5*n_in]
df_train_Y = df_train.iloc[:, 5*n_in]

df_test_X = df_test.iloc[:, 0:5*n_in]
df_test_Y = df_test.iloc[:, 5]

# cut the data off so that the number of rows is divisible by time_steps*batch_size
div_by = time_steps*batch_size
num_data = math.floor(len(df_train_X.index)/div_by)*div_by

df_train_X_cut = df_train_X[:num_data]
df_train_Y_cut = df_train_Y[:num_data]
df_test_X_cut = df_test_X[:num_data]
df_test_Y_cut = df_test_Y[:num_data]

#Convert from dataframe to ndarray
train_X = df_train_X_cut.values
test_X = df_test_X_cut.values

train_Y = df_train_Y_cut.values
test_Y = df_test_Y_cut.values

#reshape input to be 3D [samples, timesteps, features]
x = math.floor(train_X.shape[0]/time_steps) #math.floor input should be an integer but we have to cast to use as an index
train_X = train_X.reshape((x, time_steps, train_X.shape[1]))
test_X = test_X.reshape((x, time_steps, test_X.shape[1]))

#output should have same number of rows as input, take every xth value as output
train_Y = train_Y[range(0, len(train_Y), time_steps)]
test_Y = test_Y[range(0, len(test_Y), time_steps)]

# design network
model = Sequential()
model.add(LSTM(units=50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(50))
model.add(Dropout(0.2))

model.add(Dense(1))

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_logarithmic_error', optimizer=adam)

# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# fit network
history = model.fit(train_X, train_Y, epochs=100, batch_size=100, callbacks=callbacks_list, validation_data=(test_X, test_Y), verbose=2,
                    shuffle=False)

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
print("done\n")



