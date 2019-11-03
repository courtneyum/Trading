from Param import Param
from model import model
from predict import predict

# BEGIN

for i in range(len(Param.filenames)):
    file_number = i
    model(file_number=file_number)
    predict(file_number=file_number)
# END for