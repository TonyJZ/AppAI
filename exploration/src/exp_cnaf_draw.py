#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 17:04:47 2019

@author: Toukir Imam (toukir@appropolis.com)
"""
import pandas as pd
import numpy as np
import datetime
import gc
import tensorflow as tf
import os
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import re
from datetime import timedelta
import dateutil.parser

def evaluate(y,yhat):
    rmse = np.sqrt(np.mean((y - yhat)**2))
    return rmse

def draw(exp_name):
    sns.set_style("darkgrid")
    data_dir = os.path.abspath(os.path.dirname(__file__))  + "/../data/results/" + exp_name +"/"
    model_dir = os.path.abspath(os.path.dirname(__file__)) +  "/../data/models/" + exp_name + "/"
    report_dir = os.path.abspath(os.path.dirname(__file__)) + "/../reports/"

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    result_files = [f for f in os.listdir(data_dir) if re.match(r'^[0-9]*.csv',f)]
    result_files.sort(key=lambda x: int(x.split(".")[0]))
    print(result_files)
    with PdfPages( report_dir + exp_name + ".pdf") as pdf:
        for file in result_files:
            print(file)
            result = pd.read_csv(data_dir + file )
            train_result = pd.read_csv(data_dir+"train_" + file)
#            hst = pd.read_csv(data_dir + "history_" + file, header=None)
            rolling_result = pd.read_csv(data_dir + "rolling_result_" + file)
            
            name = file.split(".")[0]

            fig = plt.figure(figsize =(25,40))
            fig.suptitle("Meter id = " + str(name))
            #draw the training set
            #ax = plt.subplot(4,1,1)
            #ax.set_title("loss")
            #sns.lineplot(y = np.sqrt(hst[1]), x = [ i for i in range(0,len(hst[1]))])

            #ax.set_ylabel("rmse")
            #ax.set_xlabel("epoc")
            irow = 1
            n_sections = 4
            n_rows = n_sections + 3
            
            ax = plt.subplot(n_rows, 1, irow)
            train_rmse = evaluate(train_result.yhat,train_result.y)
            sns.lineplot(ax = ax, y = "y", x = "ds", data = train_result, label = "Real")
            #draw_time_series(ax, "yhat", "ds", train_result, "blank",604800, train_result["create_date"].iloc[0]) 
            sns.lineplot(ax = ax, y = "yhat", x = "ds", data = train_result, label = "Predicted")
            # sns.lineplot(ax = ax, y = "ydiff", x = "ds", data = train_result, label = "Difference")
            ax.set_title("Training data, RMSE = " + str(train_rmse))

            ax.set_ylabel("EP_diff")
            ax.set_xlabel("create_time")
            
            labels = ax.get_xticks().tolist()
            labels = [(dateutil.parser.parse(train_result.create_date.iloc[0]) + timedelta(seconds = label)).strftime("%m, %d, %a") for label in labels]
            ax.set_xticklabels(labels)
            irow += 1

            #draw the result
            ax = plt.subplot(n_rows,1, irow)
            rmse = evaluate(result.y, result.yhat_multistep)
            ax.set_title("Predicting RMSE = " + str(rmse))
            
            sns.lineplot(ax = ax, y="y", x ="ds" , data = result,label="Real")
            sns.lineplot(ax = ax, y="yhat_singlestep", x ="ds" , data = result,label = "singlepointPredicted")
            sns.lineplot(ax = ax, y="yhat_multistep", x ="ds" , data = result,label = "multistepPredicted")
            # sns.lineplot(ax = ax, y="ydiff", x ="ds" , data = result, label = "Difference")
            
            
            ax.set_ylabel("EP_diff")
            ax.set_xlabel("create_time")
            
            labels = ax.get_xticks().tolist()
            labels = [(dateutil.parser.parse(train_result.create_date.iloc[0]) + timedelta(seconds = label)).strftime("%m, %d, %a") for label in labels]
            ax.set_xticklabels(labels)
            irow += 1
            
            for i in range(0, n_sections):
                sec_result = pd.read_csv(data_dir + "sec_" + str(i) + "_" + file )
                #print(sec_result.columns)
                ax = plt.subplot(n_rows, 1, irow + i)
                rmse = evaluate(sec_result.y,sec_result.yhat_multistep)
                ax.set_title("Multistep RMSE = " + str(rmse))
                
                sns.lineplot(ax = ax, y="y", x ="ds" , data = sec_result,label="Real")
                sns.lineplot(ax = ax, y="yhat_singlestep", x ="ds" , data = sec_result,label = "Singlestep Predicted")
                sns.lineplot(ax = ax, y="yhat_multistep", x ="ds" , data = sec_result,label = "Multistep Predicted")

                ax.set_ylabel("EP_prime")
                ax.set_xlabel("create_time")
            
                labels = ax.get_xticks().tolist()
                labels = [(dateutil.parser.parse(sec_result.create_date.iloc[0]) + timedelta(seconds = label)).strftime("%m, %d, %a") for label in labels]
                ax.set_xticklabels(labels)
 
                #sns.lineplot(ax = ax, y="ydiff", x ="ds" , data = result, label = "Difference")
                #print(sec_result)
            
            irow += n_sections

            #draw rolling mean error
            ax = plt.subplot(n_rows,1, irow)
            ax.set_title("multistep prediction errors")
            sns.lineplot(ax = ax, y = rolling_result.mean_error*rolling_result.scale_factor,x = rolling_result.index, data = rolling_result, label = " Rolling prediction error")
            
            pdf.savefig()
            plt.clf()
            plt.close()
            
if __name__ == "__main__":
    draw("cnaf_1h_ep_diff_july_rolling_section_2")