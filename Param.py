class Param:
    path = "K:/My Drive/School/DataScienceComp/Data/"
    filenames = ["DataBAX.csv", "DataCGB.csv", "DataCURA.csv", "DataSXF.csv"]
    columns = ["open", "high", "low", "close"]

    time_steps = 10
    batch_size = 100
    validation_split = 0.33

    n_in = 2
    n_out = 3
    n_features = len(columns)*n_in
    n_targets = len(columns)*n_out

    model_filename = "model"
    best_model_filename = "best_model"

    input_scaler_filename = "input_scaler.save"
    output_scaler_filename = "output_scaler.save"

    predictions_filename = "predictions"
    price_fig_filename = "Price"

    remodel = True
    compute_predictions = True
    use_best_model = True # as opposed to final model