class Param:
    path = "K:/My Drive/School/DataScienceComp/Data/"
    filenames = ["DataBAX.csv", "DataCGB.csv", "DataCURA.csv", "DataSXF.csv"]
    columns = ["open", "high", "low", "close"]

    time_steps = 10
    batch_size = 100
    validation_split = 0.33

    n_in = 2
    n_out = 1
    n_features = len(columns*n_in)

    file_number = 0

    model_filename = "model" + str(file_number) + ".h5"
    best_model_filename = "best_model" + str(file_number) + ".h5"

    input_scaler_filename = "input_scaler.save"
    output_scaler_filename = "output_scaler.save"