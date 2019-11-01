class Param:
    path = "K:/My Drive/School/DataScienceComp/Data/"
    filenames = ["DataBAX.csv", "DataCGB.csv", "DataCURA.csv", "DataSXF.csv"]
    columns = ["open", "high", "low", "close", "time"]

    time_steps = 10
    batch_size = 100
    validation_split = 0.33

    n_in = 2
    n_out = 1

    file_number = 0

    model_filename = "model" + str(file_number) + ".h5"
    #model_filename = "lstm_model.h5"

    scaler_filename = "scaler.save"