#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:25:08 2019

@author: Toukir Imam (toukir@appropolis.com)
"""
import pandas as pd
from appai_lib import preprocessing
import os
import numpy as np

in_dir ="../data/processed/july_processed/"
# out_dir = "../data/processed/july_processed_subsampled_1h_ep_diff/"
out_dir = "../data/processed/july_processed_ep_diff/"

in_dir = os.path.abspath(os.path.dirname(__file__)) + "/" + in_dir
out_dir = os.path.abspath(os.path.dirname(__file__)) + "/" + out_dir

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#selected_meters = [1,3,4,5,12,14,19,20,21,22,26,32,33,40,42,43,50,52,53,58,60,61,65,66,67,68,69,72,73,76,77,80,106,210,245,249,343]
# selected_meters = [1,5,19,32,50,52,66,73]
selected_meters = [19]
#selected_meters = [3,19,21,68]

# interval = "1440T"     # 1 day
# interval = "360T"     # 6 hour
# interval = "60T"     # 1 hour
interval = "5T"       # 5 mins


for name in selected_meters:
    print("processing " + str(name))
    in_file = in_dir + str(name) + ".csv"
    meter_data = pd.read_csv(in_file)
    #_, meter_data = preprocessing.split_data(meter_data,"2019-02-28 00:00:00")
    meter_data.index = pd.DatetimeIndex(meter_data.create_date)
    sub_sampled = meter_data.resample(interval).max()
    
    #recalculate the elapsed time
    sub_sampled = preprocessing.add_elapsed_time(sub_sampled) 

    ep = sub_sampled["ep"]
    dep = np.diff(ep.values,1)
    

    d1list = dep.tolist()
    d1list.insert(0,0)

    sub_sampled["ep_diff"] = pd.Series(d1list, index=sub_sampled.index)
    sub_sampled.to_csv(out_dir + str(name) + ".csv")
    exit
