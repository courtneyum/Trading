import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
import copy
from momentum_indicators import *

from Param import Param

class Datatype:
    TRAIN = "train"
    TEST = "test"

def get_data(path, filename, type):
    # retrieve data and filter out unwanted columns
    filename = path + type + filename
    df = pd.read_csv(filename)
    df = df.iloc[:, 0:5]
    return df
#END get_data

def ema(data, period):
    period = round(period)

    smoothed_data = copy.deepcopy(data)

    n = len(smoothed_data)
    nsma = math.ceil((period-1)/2)
    start_data = 1
    end_sma = max([1,min([start_data+nsma-1, n])])
    
    for i in range(start_data, end_sma):
        smoothed_data.iloc[i] = smoothed_data.iloc[i] + smoothed_data.iloc[i-1]
    #END for loop
    
    nsma = end_sma - start_data + 1
    smoothed_data.iloc[start_data:end_sma] = np.divide(smoothed_data.iloc[start_data:end_sma], range(2, nsma + 1))

    alpha = 2/(1+period)
    for i in range(end_sma, n):
        smoothed_data.iloc[i] = alpha*smoothed_data.iloc[i] + (1 - alpha)*smoothed_data.iloc[i-1]
    #END for loop

    return smoothed_data
#END ema

# BEGIN
path = Param.path
filenames = Param.filenames
columns = Param.columns

for i in range(0, len(filenames)):
    # fetch unsmoothed data
    df_train = get_data(path, filenames[i], Datatype.TRAIN)
    df_test = get_data(path, filenames[i], Datatype.TEST)

    #df_train = df_train.iloc[0:1000, :]
    #df_test = df_test.iloc[0:1000, :]

    for j in range(len(Param.periods)):
        period = periods[i]
        # calculate momentum indicators
        simple_mvg_avg = sma(period, df_train.loc[:, 'close'])
        rate_of_change = roc(period, df_train.loc[:, 'close'])
        rel_strength_ind = rsi(period, df_train.loc[:, 'close'])
        comm_chann_ind = cci(df_train.loc[:, 'high'], df_train.loc[:, 'low'], df_train.loc[:, 'close'], period)

    
    # save data
    df_train_smooth.to_csv(path+"smoothed_"+Datatype.TRAIN+filenames[i])
    df_test_smooth.to_csv(path+"smoothed_"+Datatype.TEST+filenames[i])

    
#END for loop

