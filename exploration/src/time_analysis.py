#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:08:42 2019

@author: Toukir
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import dateutil.parser
import gc
from datetime import datetime
from datetime import timedelta
from pandas.plotting import autocorrelation_plot
import math
import statsmodels.api as sm

sns.set(style="dark")

def load_data(in_file):
    #column_list = ["meter_data_id","meter_id","pv_a","pv_b","pv_c","pc_a","pc_b","pc_c","ep","pf","hr","hd","ap","rp","uab","ubc","uca","in","create_date"]
    all_data = pd.read_csv(in_file)
    print("data loaded")
    return all_data

def convert_create_date(dataset, starttime=None):
    sorted_data = dataset.sort_values(by=["create_date"])
    print("first date" )
    first_date = dateutil.parser.parse(sorted_data.iloc[0,:]["create_date"])
    print(first_date)
    print((first_date - first_date).total_seconds())
    sorted_data["elapsed_time"] = sorted_data.apply(lambda row : (dateutil.parser.parse(row.create_date) - first_date ).total_seconds(), axis =1)
    return sorted_data

def autocorrelation(data):
    grouped = data.groupby("meter_id")
    with PdfPages('auto_corr.pdf') as pdf:
        for name, group in grouped:
            print("plotting " + str(name))
            group = group.sort_values(by=["elapsed_time"])
            fig = plt.figure(figsize =(40,40))

            fig.suptitle("meter id = " + str(name),fontsize=30)
            ax = plt.subplot(3,1,1)
            sns_plot = sns.lineplot(ax = ax, y="ap", x ="elapsed_time" , data = group)
            ax2 = plt.subplot(3,1,2)
            ax2.set_title("Autocorrelation with 95% and 99% confidence bands")
            autocorrelation_plot( group["ap"], ax = ax2,)
            #sm.graphics.tsa.plot_acf(group["ap"],ax= ax2)
            ax3 = plt.subplot(3,1,3)
            sm.graphics.tsa.plot_pacf(group["ap"],ax= ax3,lags=500)
            pdf.savefig()
            plt.clf()
            plt.close()


def draw_time_series(data):
    grouped = data.groupby("meter_id")
    seg_length = timedelta(days=26).total_seconds()
    with PdfPages('time_analysis_weekly.pdf') as pdf:
        for name, group in grouped:
            group = group.sort_values(by=["create_date"])

            num_subplots = math.ceil(group["elapsed_time"].max() / seg_length)
            print("num_subplots " + str(num_subplots))
            fig = plt.figure(figsize=(40, 40))
            fig.suptitle("meter id = " + str(name),fontsize=30)
            fig.subplots_adjust(hspace = .4)
            for i in range(0, num_subplots):
                segment = group.loc[(group["elapsed_time"] > i * seg_length) & (group["elapsed_time"] <= (i + 1) * seg_length)]
                ax = plt.subplot(num_subplots + 1 , 1,i+1)
                ax.set_title("week " + str(i))
                #ax.set_ylabel("ap", fontsize=30)
                #ax.set_xlabel("create time(seconds)", fontsize=30)
                sns_plot = sns.lineplot(ax = ax, y="ap", x ="elapsed_time" , data = segment)

                #plot auto correlation


            pdf.savefig()
            plt.clf()
            plt.close()
            gc.collect()
if __name__ == "__main__":
    in_file = "../data/filtered_data_36.csv"
    #in_file = "../data/filtered_data_head_10000.csv"
    #all_data = load_data(in_file)
    #data_elapsed = convert_create_date(all_data,"2019-04-01 00:00:00")
    #data_elapsed.to_csv("../data/filtered_data_36_w_elapsed.csv")
    data = load_data("../data/filtered_data_36_w_elapsed.csv")
    #draw_time_series(data)
    autocorrelation(data)