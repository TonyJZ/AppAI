#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:47:06 2019

@author: Toukir Imam (toukir@appropolis.com)
"""
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import gc
import os
import re
import matplotlib
from appai_lib import vis
from appai_lib import preprocessing
from pandas.plotting import autocorrelation_plot

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

exp_name = "time_analysis_january_ap"
ex = Experiment(exp_name)
ex.observers.append(MongoObserver.create(db_name="appai_sacred"))
@ex.config
def config():
    #selected_meters = [77, 85, 106, 108, 210, 245, 249, 375, 396, 404, 509, 543, 562, 53, 65, 79, 114, 265, 371, 409, 427, 591, 13, 80, 81, 102, 121, 161, 164, 184, 206, 238, 257, 274, 293, 311,343,348,365,405,411,438,443,443,491,532,549,592]
    selected_meters = [1,3,4,5,12,14,19,20,21,22,26,32,33,40,42,43,50,52,53,58,60,61,65,66,67,68,69,72,73,76,77,80,106,210,245,249,343]
    #selected_meters = [1, 3, 4, 5, 11, 12, 14, 19, 20, 21, 22, 26, 32, 33, 40, 42, 43, 45, 46, 50, 52, 53, 58, 60,61, 65, 66, 67, 68, 69, 72, 73, 76, 77, 79, 80]
    #selected_meters = [65, 77, 106, 210, 245, 249, 343]
    input_dir = "../data/processed/january/"

@ex.automain
def run(selected_meters,input_dir):
    with PdfPages("../reports/" + exp_name + ".pdf" ) as pdf:    
        for name in selected_meters:
            print("drawing meter " + str(name))
            in_file = input_dir + str(name) + ".csv"    
            data = pd.read_csv(in_file )
            fig = plt.figure(figsize =(25,26/4 * 6))
            fig.suptitle("Meter id = " + str(name))
            
            data_1, data_t = preprocessing.split_data(data,"2019-02-28 00:00:00")
            data_2, data_t = preprocessing.split_data(data_t, "2019-03-31 00:00:00")
            data_3, data_t = preprocessing.split_data(data_t, "2019-04-30 00:00:00")
            data_4, data_5 = preprocessing.split_data(data_t, "2019-05-31 00:00:00")
            #data_5, data_t = preprocessing.split_data(data_t, "2019-06-30 00:00:00")
            
            #February
            ax = plt.subplot(5, 1, 1)
            sharey = ax
            #vis.draw_time_series(ax, "ep_diff", "elapsed_time", data_1, "February", 604800, data_1["create_date"].iloc[0] ) 
            
            #ax = plt.subplot(5, 1, 2)
            #vis.draw_time_series(ax, "ep_diff_med_smooth", "elapsed_time", data_1, "February ep_diff_smooth",604800, data_1["create_date"].iloc[0]) 
            
            vis.draw_time_series(ax, "ap", "elapsed_time", data_1, "February ap",604800, data_1["create_date"].iloc[0]) 
            
            #ax = plt.subplot(6, 1, 3)
            #vis.draw_scatter_label(ax, "ep_diff", "elapsed_time", data_1, "February",604800, data_1["create_date"].iloc[0],"label",["Anomaly","Normal"]) 
            
            #ax = plt.subplot(10, 1, 4)
            #vis.draw_time_series(ax, "ap_med_smooth", "elapsed_time", data_1, "Aprtil ap Median smoothing",604800, data_1["create_date"].iloc[0]) 
            
            #ax = plt.subplot(6, 1, 3)
            #ax.set_title("April autocorr")
            #autocorrelation_plot(ax = ax, series = data_1["ep_diff"])
            
            
            
            #March
            ax = plt.subplot(5, 1, 2)
            #vis.draw_time_series(ax, "ep_diff", "elapsed_time", data_2, "March ep_diff",604800, data_2["create_date"].iloc[0]) 
            
            #ax = plt.subplot(5, 1, 5)
            #vis.draw_time_series(ax, "ep_diff_med_smooth", "elapsed_time", data_2, "March ep_diff_med_smooth",604800, data_2["create_date"].iloc[0]) 
            vis.draw_time_series(ax, "ap", "elapsed_time", data_2, "March ap",604800, data_2["create_date"].iloc[0]) 
            
            #ax = plt.subplot(6, 1, 6)
            #vis.draw_scatter_label(ax, "ep_diff", "elapsed_time", data_2, "March", 604800, data_2["create_date"].iloc[0],"label",["Anomaly","Normal"]) 
            
            #ax = plt.subplot(10, 1, 9)
            #vis.draw_time_series(ax, "ap_med_smooth", "elapsed_time", data_2, "May ep_diff Median smoothing",604800, data_2["create_date"].iloc[0]) 
            
            #ax = plt.subplot(6, 1, 6)
            #ax.set_title("May autocorr")
            #autocorrelation_plot(ax = ax,series =  data_2["ep_diff"])
            
            #April
            ax = plt.subplot(5, 1, 3)
            #vis.draw_time_series(ax, "ep_diff", "elapsed_time", data_3, "April ep_diff",604800, data_3["create_date"].iloc[0]) 
            
            #ax = plt.subplot(5, 1, 5)
            #vis.draw_time_series(ax, "ep_diff_med_smooth", "elapsed_time", data_3, "April ep_diff_med_smooth" ,604800, data_3["create_date"].iloc[0]) 
            vis.draw_time_series(ax, "ap", "elapsed_time", data_3, "April ap" ,604800, data_3["create_date"].iloc[0]) 
            #May
            ax = plt.subplot(5, 1, 4)
            #vis.draw_time_series(ax, "ep_diff", "elapsed_time", data_4, "May",604800, data_4["create_date"].iloc[0]) 
            
            #ax = plt.subplot(5, 1, 5)
            #vis.draw_time_series(ax, "ep_diff_med_smooth", "elapsed_time", data_4, "May ep_diff_med_smooth", 604800, data_4["create_date"].iloc[0]) 
            vis.draw_time_series(ax, "ap", "elapsed_time", data_4, "May ap", 604800, data_4["create_date"].iloc[0]) 
            
            #June
            ax = plt.subplot(5, 1, 5)
            #vis.draw_time_series(ax, "ep_diff", "elapsed_time", data_5, "May", 604800, data_5["create_date"].iloc[0]) 
            
            #ax = plt.subplot(5, 1, 5)
            #vis.draw_time_series(ax, "ep_diff_med_smooth", "elapsed_time", data_5, "June ep_diff_med_smooth", 604800, data_5["create_date"].iloc[0]) 
            vis.draw_time_series(ax, "ap", "elapsed_time", data_5, "June ap", 604800, data_5["create_date"].iloc[0]) 
            
            pdf.savefig(fig)
            plt.close()
    ex.add_artifact("../reports/"+ exp_name +".pdf")
    
    