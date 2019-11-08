from Param import Param
from model import model
from predict import predict

import dataHelper as dh
from trade import trade

# BEGIN

for i in range(len(Param.filenames)):

    if Param.remodel:
        model(file_number=i)
    else:
        #Fetch data to ensure proper scalers are available
        train_X = dh.get_training_input(Param.filenames[i])
        train_Y = dh.get_training_output(Param.filenames[i])
    # END if 

    predictions = predict(file_number=i)
    profit = trade(predictions, file_number=i)
# END for