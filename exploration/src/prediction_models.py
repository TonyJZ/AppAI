#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:13:44 2019

@author: Toukir
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def load_data(in_file):
    #column_list = ["meter_data_id","meter_id","pv_a","pv_b","pv_c","pc_a","pc_b","pc_c","ep","pf","hr","hd","ap","rp","uab","ubc","uca","in","create_date"]
    all_data = pd.read_csv(in_file)
    print("data loaded")
    return all_data

def train(train_set):
    mean_aps = pd.DataFrame(columns=["meter_id","mean_ap"])
    grouped = train_set.groupby('meter_id')
    for name, group in grouped:
        mean = group["ap"].median()
        mean_aps = mean_aps.append({"meter_id":name,"mean_ap":mean}, ignore_index=True)
    return mean_aps

def real_time_prediction(test_set):
    n = 1
    predict_result = pd.DataFrame(columns=["meter_id","ap","meter_data_id","create_date","predicted_ap"])
    groups = test_set.groupby("meter_id")
    for name,group in groups:
        print("processing meter " + str(name))
        group = group.sort_values(by=["create_date"])
        for index in range(0,group.shape[0]):
            row = group.iloc[index]
            if index > n:
                rows = group.iloc[index - n : index]
            else:
                rows = group.iloc[:index]
            predict_result = predict_result.append({"meter_id":row.meter_id,
                                   "meter_data_id":row.meter_data_id,
                                   "ap":row.ap,"create_date":row.create_date,
                                   "predicted_ap":rows.mean().ap}, ignore_index=True)
    return predict_result

def predict(test_set, model):
    def f(row):
        #return 3
        #mean_ap = model.loc[model.meter_id == row.meter_id,:]["mean_ap"]
        mean_ap = model.loc[model.meter_id == row.meter_id].iloc[0]["mean_ap"]
        return mean_ap
        #return model.loc[model.meter_id == row.meter_id,:]["mean_ap"]

    predict_result = test_set[["meter_id","meter_data_id","create_date", "ap"]]
    predict_result["predicted_ap"] = predict_result.apply(f,axis=1)
    return predict_result

def evaluate(result):
    rmse = np.sqrt(np.mean((result["ap"] - result["predicted_ap"])**2))
    return rmse

def split_data(all_data, split_date):
    train_data = all_data[all_data.create_date <= split_date]
    test_data = all_data[all_data.create_date > split_date]
    print ("train_data " + str(train_data.create_date.max()))
    print ("test_data " + str(test_data.create_date.max()))
    return (train_data, test_data)

    #print(all_data["create_date"])
    #print (type(all_data["create_date"][0]))

if __name__ == "__main__":
    in_file = "../data/filtered_data_36.csv"
    #in_file = "../data/filtered_data_head_10000.csv"
    all_data = load_data(in_file)
    (train_data, test_data) = split_data(all_data,"2019-04-18 01:00:00")
    #(train_data, test_data) = split_data(all_data,"2019-04-01 01:00:00")
    #model = train(train_data)
    #predicted_ap = predict(test_data,model)
    #accuracy = evaluate(predicted_ap)
    #print("RMSE is " + str(accuracy))
    predict_result = real_time_prediction(test_data)
    rmse = evaluate(predict_result)
    print("rmse " + rmse)
