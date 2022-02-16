#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:34:10 2019

@author: Toukir Imam (toukir@appropolis.com)
"""
import pandas as pd
import numpy as np
from appai_lib import preprocessing
import os


#in_dir = "../data/benchmark/label/converted/"
in_dir = "../data/processed/june/"
out_dir = "../data/processed/june_labeled_2/"
label_dir = "../data/benchmark/label_set_2/"

#selected_meters = [65, 77, 106, 210, 245, 249, 343]
selected_meters = [1,3,4,5,12,14,19,20,21,22,26,32,33,40,42,43,50,52,53,58,60,61,65,66,67,68,69,72,73,76,77,80,106,210,245,249,343]
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for name in selected_meters:
    anomaly_dir = label_dir + str(name) + "/"
    anomaly_files = [anomaly_dir + x for x in os.listdir(anomaly_dir)]
    data_file = in_dir + str(name) + ".csv"
    print("data file " + data_file)
    data = pd.read_csv(data_file)
    
    all_anomaly = []
    for an_file in anomaly_files:
        print(" anomaly file " + an_file)
        an_data = pd.read_csv(an_file, index_col = 0)
        all_anomaly.append(an_data)
    all_anomaly = pd.concat(all_anomaly)
    real_ep_diff = data["create_date"].isin(all_anomaly.date)
    data["label"] = real_ep_diff
    data.to_csv(out_dir + str(name) + ".csv")
    