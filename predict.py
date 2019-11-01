import os; os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import model_from_json
from keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt
import joblib

import dataHelper as dh
from Param import Param

# BEGIN
path = Param.path
filenames = Param.filenames
columns = Param.columns

# load model from single file
model = load_model(Param.model_filename)
print("Loaded model from disk")

# load scaler
scaler = joblib.load(Param.scaler_filename)

# Choose these parameters so that we keep as much data as possible
time_steps = Param.time_steps
batch_size = Param.batch_size

n_in = Param.n_in
n_out = Param.n_out

# load training data
test_X = dh.get_input(path, filenames[1], dh.Datatype.TEST, n_in, n_out, time_steps, batch_size)
test_Y = dh.get_output(path, filenames[1], dh.Datatype.TEST, n_in, n_out, time_steps, batch_size)
train_Y = dh.get_output(path, filenames[1], dh.Datatype.TRAIN, n_in, n_out, time_steps, batch_size)

predictions = model.predict(test_X)

# scale values back
test_Y = scaler.inverse_transform(test_Y)
predictions = scaler.inverse_transform(predictions)

# plot to compare predicted with actual
plt.plot(predictions, label='predicted')
plt.plot(test_Y, label='actual')
plt.legend()
plt.show()