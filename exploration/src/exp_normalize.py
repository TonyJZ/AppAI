#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:30:13 2019

@author: Toukir Imam (toukir@appropolis.com)
"""
import os
import re
import pandas as pd
from sklearn.preprocessing import minmax_scale
def scale(data):
    data["ap_scaled"] =  minmax_scale(data.ap)
    return data



data_dir = "../data/processed/april_may_elapsed_filtered/"
out_dir = "../data/processed/april_may_elapsed_filtered_scaled/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


#selected_meters = [1, 3, 4, 5, 11, 12, 14, 19, 20, 21, 22, 26, 32, 33, 40, 42, 43, 45, 46, 50, 52, 53, 58, 60,61, 65, 66, 67, 68, 69, 72, 73, 76, 77, 79, 80]
print(os.listdir(data_dir))
#in_files = [f for f in os.listdir(data_dir) if int(f.split(".")[0]) in selected_meters]
in_files = [f for f in os.listdir(data_dir)]

print(in_files)

for file in in_files:
    in_file = data_dir + file
    meter_data = pd.read_csv(in_file)
    scaled_data = scale(meter_data)
    scaled_data.to_csv(out_dir + file)


