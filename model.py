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

import dataHelper as dh
from Param import Param

# BEGIN

# set paramters
path = Param.path
filenames = Param.filenames
columns = Param.columns

time_steps = Param.time_steps
batch_size = Param.batch_size

n_in = Param.n_in
n_out = Param.n_out

file_number = Param.file_number

#retrieve and format training data
train_X = dh.get_input(path, filenames[file_number], dh.Datatype.TRAIN, n_in, n_out, time_steps, batch_size)
train_Y = dh.get_output(path, filenames[file_number], dh.Datatype.TRAIN, n_in, n_out, time_steps, batch_size)

# design network
model = Sequential()
model.add(LSTM(units=100, batch_input_shape=(Param.batch_size, train_X.shape[1], train_X.shape[2]), dropout=0.0, recurrent_dropout=0.0, stateful=True, kernel_initializer='random_uniform'))
model.add(Dropout(0.5))

model.add(Dense(20,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

optimizer = optimizers.RMSprop(lr=0.0001)
model.compile(loss='mape', optimizer=optimizer)

# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# split data into training and validation sets
num_batches = math.floor(train_X.shape[0]/batch_size)
num_batches_val = math.floor(Param.validation_split*num_batches)
num_batches_train = num_batches - num_batches_val

val_X = train_X[num_batches_train*batch_size:, :, :]
val_Y = train_Y[num_batches_train*batch_size:]

train_X = train_X[0:num_batches_train*batch_size, :, :]
train_Y = train_Y[0:num_batches_train*batch_size]

# fit network
history = model.fit(train_X, train_Y, epochs=100, batch_size=Param.batch_size, validation_data=(val_X, val_Y), verbose=2,
                    shuffle=False, callbacks=callbacks_list)

# plot history
# plot metrics
plt.plot(history.history['loss'], label='mape')
plt.plot(history.history['val_loss'], label='val_mape')
plt.legend()

plt.show()

# save model to single file
model.save(Param.model_filename)
print("Saved model to disk")
