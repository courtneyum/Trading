import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import math

from Param import Param

class Datatype:
    TRAIN = 1
    TEST = 2

def get_data(path, filename, type, n_in, n_out, time_steps, batch_size):
    # retrieve data and drop unwanted columns
    if type == Datatype.TRAIN:
        filename = path + "smoothed_train" + filename
    else:
        filename = path + "smoothed_test" + filename
    #END if
    
    df = pd.read_csv(filename)
    df.drop(df.columns[0], axis=1, inplace=True)
    df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)

    # Reframe as a supervised learning problem
    df_reframed = series_to_supervised(df, n_in, n_out)

    return df_reframed
#END get_data

def get_input(path, filename, type, n_in, n_out, time_steps, batch_size):
    df_X = get_data(path, filename, type, n_in, n_out, time_steps, batch_size)
    #split data into input and output
    df_X.drop(df_X.columns[4*n_in:len(df_X.columns)], axis=1, inplace=True)

    # fetch scaler
    if type == Datatype.TRAIN:
        scaler = MinMaxScaler()
    else:
        scaler = joblib.load(Param.input_scaler_filename)
    #END if
    
    # normalize features
    df_X.iloc[:,:] = scaler.fit_transform(df_X)

    #END if

    if type == Datatype.TRAIN:
        joblib.dump(scaler, Param.input_scaler_filename)
    # END if

    # cut the data off so that the number of rows is divisible by time_steps*batch_size
    div_by = time_steps*batch_size
    num_data = math.floor(len(df_X.index)/div_by)*div_by

    df_X_cut = df_X[:num_data]

    #Convert from dataframe to ndarray
    X = df_X_cut.values

    #reshape input to be 3D [samples, timesteps, features]
    num_samples = math.floor(X.shape[0]/time_steps) #math.floor input should be an integer but we have to cast to use as an index
    X = X.reshape((num_samples, time_steps, X.shape[1]))
    X = X[range(len(X)-1)]

    return X
#END get_input

def get_output(path, filename, type, n_in, n_out, time_steps, batch_size):
    df_Y = get_data(path, filename, type, n_in, n_out, time_steps, batch_size)
    df_Y.drop(df_Y.columns[0:4*n_in], axis=1, inplace=True)
    df_Y.drop(df_Y.columns[1:len(df_Y.columns)], axis=1, inplace=True)

    # fetch scaler
    if type == Datatype.TRAIN:
        scaler = MinMaxScaler()
    else:
        scaler = joblib.load(Param.output_scaler_filename)
    #END if
    
    # normalize features
    if type == Datatype.TRAIN:
        index = df_Y.index
        columns = df_Y.columns
        values = scaler.fit_transform(df_Y.values.reshape(-1, 1))

        df_Y = pd.DataFrame(data=values, index=index, columns=columns)
    #END if

    if type == Datatype.TRAIN:
        joblib.dump(scaler, Param.output_scaler_filename)
    # END if

    # cut the data off so that the number of rows is divisible by time_steps*batch_size
    div_by = time_steps*batch_size
    num_data = math.floor(len(df_Y.index)/div_by)*div_by

    df_Y_cut = df_Y[:num_data]

    #Convert from dataframe to ndarray
    Y = df_Y_cut.values

    #output should have same number of rows as input, take every xth value as output
    Y = Y[range(time_steps+1, len(Y), time_steps)]

    return Y
#END get_output

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
