import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
import copy

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

    # smooth data
    span = 50
    df_train_smooth = pd.DataFrame(index=df_train.index, columns=df_train.columns)
    df_test_smooth = pd.DataFrame(index=df_test.index, columns=df_test.columns)
    for j in range(0, len(columns)):
        if columns[j] == "time":
            continue
        #END if

        df_train_smooth.iloc[:,j] = ema(df_train.iloc[:, j], span)
        df_test_smooth.iloc[:,j] = ema(df_test.iloc[:, j], span)

        #plt.plot(df_train.iloc[:,j], 'r--', df_train_smooth.iloc[:,j], 'b--')
        #plt.show()
    #END for loop
    
    # save data
    df_train_smooth.to_csv(path+"smoothed_"+Datatype.TRAIN+filenames[i])
    df_test_smooth.to_csv(path+"smoothed_"+Datatype.TEST+filenames[i])

    
#END for loop

