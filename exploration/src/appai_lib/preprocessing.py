#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:32:58 2019

@author: appropolis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import gc
import dateutil.parser
from sklearn.preprocessing import minmax_scale
from appai_lib import my_io
import scipy.signal as signal


def add_first_derivative(df, column_name):
    if {'elapsed_time'}.issubset(df.columns):
        df.sort_values(by=["elapsed_time"])
        df.reset_index()
        pass
    else :
        print("Error: no elapsed_time column")

    time = df["elapsed_time"]
    ep = df[column_name]
    # df.dtypes
    # print(type(time))
    # print(type(ep))
    # print(time)
    # print(ep)
    dt = np.diff(time.values,1)
    dep = np.diff(ep.values,1)
    d1 = dep / dt * 3600

    d1list = d1.tolist()
    d1list.insert(0,0)

    df[column_name + "_diff"] = pd.Series(d1list, index=df.index)
    # print(type(d1))
    # print(d1)
    return df

def add_time_features(data, holidays):
    data["workday"] = data.apply(lambda row: 0 if pd.to_datetime(row.create_date).date() in holidays else 1, axis = 1 )
    data["weekday"] = data.apply(lambda row: pd.to_datetime(row.create_date).date().weekday()/7, axis = 1)
    data["hour"] = data.apply(lambda row: pd.to_datetime(row.create_date).hour/24, axis = 1)
    data["minute"] = data.apply(lambda row: pd.to_datetime(row.create_date).minute/60, axis = 1)
    return data


def add_scale(data, column_name):
    data[column_name + "_scaled"] =  minmax_scale(data[column_name])
    return data

def add_elapsed_time(sorted_data, starttime = None):
    #sorted_data = dataset.sort_values(by=["create_date"])
    first_date = dateutil.parser.parse(sorted_data.iloc[0,:]["create_date"])
    sorted_data["elapsed_time"] = sorted_data.apply(lambda row : (dateutil.parser.parse(row.create_date) - first_date ).total_seconds(), axis =1)
    return sorted_data

def preprocess(in_file, out_file):
    all_data = my_io.load_data(in_file)
    filter_out_ids = all_nan + zero_sum
    print("filtering out " + str(len(filter_out_ids)))
    filter_out(all_data, out_file, filter_out_ids )
    #filtered_data = filter_data(all_data)

def filter_out(all_data,filter_meter_ids, selected_meters = None):
    print(all_data.shape)
    filtered_data = all_data[~all_data.meter_id.isin(filter_meter_ids)]
    print("first filter " + str(filtered_data.shape))

    filtered_data = filtered_data[~((np.isnan(filtered_data.ap))) ]
    print("second filter " + str(filtered_data.shape))

    if selected_meters != None:
        filtered_data = all_data[all_data.meter_id.isin(selected_meters)]
        print("third filter " + str(filtered_data.shape))

    print ("removed " + str(all_data.shape[0] - filtered_data.shape[0]))
    return filtered_data
    #filtered_data.to_csv(out_file)

def filter_data(all_data):
    grouped = all_data.groupby('meter_id')
    nan_ids = []
    print("All Nan:")
    for name, group in grouped:
        ep = group["ep"]
        ap = group["ap"]
        #print(create_column)
        #print(type(create_column))
        #print(create_column.size)
        #  count non NaN ep column
        nan_count_ep = len(ep) - ep.count()
        nan_count_ap = len(ap) - ap.count()
        if (nan_count_ep == len(ep)) and (nan_count_ap == len(ap)):
            nan_ids.append(name)
            #print(group)
        #remove all zero columns
    print(", ".join(map(str, nan_ids)))
    print("Zero sums:")
    zero_sum_ids = []
    for name,group in grouped:
        ep = group["ep"]
        ap = group["ap"]
        if(ep.sum() <= 0 and ap.sum() <= 0):
            zero_sum_ids.append(name)
    print(", ".join(map(str, zero_sum_ids)))

def split_data(all_data, split_date):
    train_data = all_data[all_data.create_date <= split_date].reset_index(drop=True)
    test_data = all_data[all_data.create_date > split_date].reset_index(drop = True)
    print ("train_data " + str(train_data.create_date.max()))
    print ("test_data " + str(test_data.create_date.max()))
    return (train_data, test_data)

def median_smoothing(df, column_name, win_size):
    # x =df[column_name].tolist()
    # signal.medfilt(x, win_size)
    # df["median_filter"] = pd.Series(x)

    df[column_name +"_med_smooth"] = signal.medfilt(df[column_name], win_size)
    return df

# type = "centered" or "trailing" 
def moving_average_smoothing(df, column_name, win_size = 3, type="centered"):
    if(type == "centered" ):
        df[column_name + "_avg_center_smooth"] = df[column_name].rolling(window=win_size, center=True).mean()
    elif(type == "trailing"):
        df[column_name + "_avg_trailing_smooth"] = df[column_name].rolling(window=win_size, center=False).mean()
    else :
        pass
    return df

def exponential_weighted_smoothing(df, column_name, win_size = 3):
    df[column_name + "_avg_center_smooth"] = df[column_name].ewm(span=win_size).mean()
    return df