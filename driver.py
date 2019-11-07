from Param import Param
from model import model
from predict import predict

# BEGIN

for i in range(len(Param.filenames)):
    Param.file_number = i

    if Param.remodel:
        model(file_number=i)
    # END if 

    predict(file_number=i)
# END for