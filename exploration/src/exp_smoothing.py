import pandas as pd
import numpy as np
from appai_lib import preprocessing
import os

data_dir = "../data/processed/june/"
out_dir = "../data/processed/june/smoothed/"

data_dir = os.path.abspath(os.path.dirname(__file__)) + "/" + data_dir
if not os.path.exists(data_dir):
    print("%s is not existed!" % data_dir)
    
out_dir = os.path.abspath(os.path.dirname(__file__)) + "/" + out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


selected_meters = [4,11,12,77, 20,21,22]
for name in selected_meters:
    print (name)
    in_file = data_dir + str(name) + ".csv"
    meter_data = pd.read_csv(in_file)

    column_name = "ep_diff"
    win_size = 11
   
    df = preprocessing.median_smoothing(meter_data, column_name, win_size)
    df = preprocessing.moving_average_smoothing(meter_data, column_name, win_size)
    df.to_csv(out_dir + str(name) + ".csv")
