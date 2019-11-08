Setting up your Python environment: 

IMPORTANT: This procedure and our code has only been tested on Windows. If you try to setup or run our code on any other operating system, the code may require adjustment and your setup process may be more difficult.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) Python 3.7 for Windows
2. Install [Visual Studio Code](https://code.visualstudio.com/download)
3. Open Visual Studio Code and install the Python extension from Microsoft.
	The button to open the extensions panel is found on the left hand side.
	If VSCode gives a popup suggesting that you install a linter, it is helpful but not needed. A linter will help catch 		errors in code before you run and will also colour code keywords for improved readability.
4. Open the "Trading" folder in Visual Studio Code. Double click "driver.py" to open this file in the editor.
5. Look at the bottom right on the VS Code window and find where it says "Plain Text". Click here and select "Python" from the dropdown that appears.
6. Next, VS Code should automatically ask you to select a Python interpreter. Select [Miniconda Path]\python.exe as your Python interpreter. Your selection should become apparent in the bottom left of the VS Code window.
7. The following need to be added to your "Path" environment variable:  
	* [Miniconda Path]\Scripts
	* [Miniconda Path]
	* [Miniconda Path]\Library\bin
8. Install the following packages: 
	* pandas  
	* matplotlib  
	* numpy
	* theano
	* keras
	* sklearn
	* joblib  
with the command `pip install [package_name]` in the Visual Studio Code terminal.
9. Find the file "scan_perform.c" as part of the project code. Insert this file in [Miniconda Path]\Lib\site-packages\theano\scan_module\c_code. If the c_code folder does not exist, create it at the specifed location. **Note that this is not our original code but was taken from github as a theano package bug fix.** If the file is already found at that location, then ignore this step.
10. Now you should be ready to run our program. The entry point is "driver.py".    


We built a neural network based on LSTM (long short term memory) recurrent neural network architecture using the Keras library in python. This problem was framed as a regression problem.

**Training**: We did a 2:1 training:validation split of the data during the training phase and trained a new model for each dataset given. Our features consisted of two timesteps each of “high”, “low, “close” and “open” data points and our targets consisted of the three following steps of each quantity. This means that the network predicts what the next three “high”, “low”, “open”, and “close” values will be. 
In practice, we would need to retrain the model every so often as the trends captured will not be valid for the rest of time.

**driver.py**
This is a script that will create a model and then make predictions for each dataset. It is the entrypoint of our code.
If Param.remodel = False, then it will load a precomputed model. Precomputed models have been included as part of the project code.
If Param.compute_predictions = False, then it will load precomputed predictions. Graphs will be plotted and success metrics will be computed and reported. Precomputed predictions have also been included as part of the project code.
If either of these parameters are set to True, it will take 1-2 hours to compute.

**model.py**
1. `create_model()` The network architecture is defined here as well as most of its hyperparameters. Two LSTM layers are interleaved with Dropout to prevent overfitting. We use an Adam optimizer with a learning rate of 0.0001. As a loss function, we use mean absolute percentage error so that we see error in terms of a percentage. We found that mean absolute error or mean squared error did not give an accurate representation of the scale of the error. However, there is the risk of dividing by zero if there is a true target value equal to zero. This would force us to use a different loss function.
3. `model()` This is the main function of this module. Here, we load in the data, perform a training/validation split, train the model with the call to model.fit(...) and then plot the results.

**predict.py**
1. `predict()` Here we load the model that the data was trained on and fetch the testing data. The past 10 days are used to predict the next three days.
2. `evaluate_predictions()` Here we calculate the accuracy of the model for each target. The accuracy is the number of times that the model guesses the sign of a change in price correctly divided by the total number of guesses.

**dataHelper.py**
1. `get_data()` This function fetches the wanted data from csv, removes unwanted columns, and converts it to the format with columns [t-1, t-2, t, t+1, t+2] for each column in the loaded data. T stands for timestep.
2. `get_training_input()` This function fetches input data for training purposes. We fetch the training data and remove target columns. Then we scale it using a min_max scaler so that each data point is between 0 and 1. Then we save this scaler for use later. We ensure that the length of the data is divisible by time_steps * batch_size where time_steps is the number of time_steps used to make a prediction and batch_size is the amount of samples used to train at a time. A sample consists of time_steps * n_features datapoints.
3. `get_training_output()` This function fetches output data for training purposes. The differences here are that we remove the feature columns after fetching the data, and at the end we only take every 10th value so that there is the same number of output samples as input samples.
4. `get_testing_input()` The difference between this function and the training function is that we do not reshape to 3 dimensions. This reshaping is done at the predictions phase.
5. `get_testing_output()` The difference between this function and the training function is that we keep every value.
6. `series_to_supervised()` Supervised learning regression problems require a certain data arrangement. This is best explained [here](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/) where we got this code from.

**prepare.py**
1. `get_data()` A simpler version of the function from **data_helper**. Just read the csv and drop unwanted columns.
2. `ema()` Smooth the data with an exponential moving average to reduce noise.
3. This is a script and should be run before the driver program. No need for the given data set as smoothed versions have been provided.

**Param.py**
This is a static class of parameters:  
* `models_dir` is the location the precomputed models are saved to
* `data_dir` is the location that the data is stored. Precomputed predictions also get stored here.
* `filenames` is an array containing each of the filenames of the data that we wish to process.
* `columns` is an array of the relevant column names
* `time_steps` is the number of time_steps to include in a training sample. If this number is higher then the network gets more information for each prediction, and if it is lower it gets less.
* `batch_size` amount of samples to train on at a time.
* `validation_split` the amount of our training data to allow the network to use for validation.
* `n_in` the number of time steps of each feature to train on.
* `n_out` the number of time steps of each target to predict.
* `n_features` the total number of features
* `n_targets` the total number of targets. Currently, each columns is used as a feature and a target, but at different times.
* `loss` The loss function to use. This built in function will measure error in our network. Other options are "mae" (mean absolute error) or "mse" (mean squared error).
* `learning_rate` a hyperparameter required by the optimizer (we use Adam) of the network. Small learning rate helps us avoid skipping over the optimum configuration.
* `model_verbosity` determines how much output you want to see from the network during training. For speed, use 0. 
* `model_filename` the name to give the precomputed model files. An index will be appended to each one to differentiate them.
* `best_model_filename` same as previous, but the name for the best model. This is the model with the minimum validation loss.
* `input_scaler_filename` is the filename of the input scaler that will be saved during training to be used during testing.
* `output_scaler_filename` same as previous but for the output
* `predictions_filename` the filename to give computed predictions when saving. An index will be appended to differentiate them.
* `price_fig_filename` the filename to give figures that compare predicted prices and actual prices during testing. Again, an index will be appended to differentiate.
* `remodel` a boolean that tells the program whether to train, or use a precomputed model.
* `compute_predictions` a boolean that tells the program whether to compute predictions, or load precomputed ones.
* `use_best_model` if True, we use the best model from `best_model_filename`. Otherwise, we use the final model that is created after all training has finished.
* `show_plots` if true, the program will pause after each figure is created for you to view it. In order to continue, you will have to close the plots.
* `col_to_trade_on` This is the column whose price fluctuations decide whether we make a trade. Currently it is the closing price, but can be changed to any of the other columns.
