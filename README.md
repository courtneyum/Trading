Setting up your Python environment: 

IMPORTANT: This procedure and our code has only been tested on Windows. If you try to setup or run our code on any other operating system, the code may require adjustment and your setup process may be more difficult.

1: Install Miniconda Python 3.7 for Windows: https://docs.conda.io/en/latest/miniconda.html
2: Install Visual Studio Code: https://code.visualstudio.com/download
3: Open Visual Studio Code and install the Python extension from Microsoft.
	The button to open the extensions panel is found on the left hand side.
4: Select ~\Miniconda3\python.exe as your Python interpreter.
5: Install the following packages: 
	pandas
	matplotlib
	numpy
	theano
	keras
	sklearn
	joblib
with the command "pip install [package_name]" in the Visual Studio Code terminal.
6: Find the file "scan_perform.c" as part of the project code. Insert this file in ~\Miniconda3\Lib\site-packages\theano\scan_module\c_code. If the c_code folder does not exist, create it at the specifed location. **Note that this is not our original code but was taken from github as a theano package bug fix.
7: Now you should be ready to run our program. Open the "Trading" folder in Visual Studio Code. The entry point is "driver.py".
