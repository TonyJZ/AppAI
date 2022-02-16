#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:56:49 2019

@author: Toukir Imam (toukir@appropolis.com)
"""
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import timedelta
import dateutil.parser

import os
import re
import matplotlib
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import datetime
sns.set_style("white")
# import ray
def evaluate(y,yhat):
    rmse = np.sqrt(np.mean((y - yhat)**2))
    return rmse

def draw_scatter_label(ax, y, x, data,ax_title = None, seg_length = None, start_date = None, label=None, legend = None):
    start_date = dateutil.parser.parse(start_date)
    one_data = data[data[label] == 1]
    zero_data = data[data[label] == 0]
    sns.scatterplot(ax = ax, y = y, x = x, data = one_data)
    sns.scatterplot(ax = ax, y = y, x = x, data = zero_data)
    ax.legend(labels = legend)
    labels = ax.get_xticks().tolist()
    labels = [(start_date + timedelta(seconds = label)).strftime("%m,%d,%a") for label in labels]
    
    ax.set_xticklabels(labels)
    ax.set_title(ax_title)
    if seg_length != None:
        y_max = data[y].max()
        i = 0
        to_exit = False
        #ax.fill_betweenx(np.full(data[x].size,y_max), 0,1000000,color='red')
        colors_gb = np.linspace(0,1,data[x].iloc[-1] // seg_length + 1)
        while True:
            x_max =data[x][0] +  (i + 1) * seg_length
            x_min =data[x][0] +  i * seg_length
            if x_max > data[x].iloc[-1]:
                x_max = data[x].iloc[-1]
                to_exit = True
                y = np.full( data[x].size ,y_max)
            ax.add_patch(patches.Rectangle((x_min,0),x_max,y_max,color=(colors_gb[i], colors_gb[i], colors_gb[i], 0.2)))
            if to_exit:
                break
            i = i+1
        
        ax.set_xticklabels(labels)
    return ax


def draw_time_series(ax, y, x, data,ax_title = None,seg_length = None, start_date = None):
    start_date = dateutil.parser.parse(start_date)
    
    sns.lineplot(ax = ax, y = y, x = x, data = data)
    labels = ax.get_xticks().tolist()
    labels = [(start_date + timedelta(seconds = label)).strftime("%m,%d,%a") for label in labels]
    
    ax.set_xticklabels(labels)
    ax.set_title(ax_title)
    if seg_length != None:
        y_max = data[y].max()
        i = 0
        to_exit = False
        #ax.fill_betweenx(np.full(data[x].size,y_max), 0,1000000,color='red')
        colors_gb = np.linspace(0,1,data[x].iloc[-1] // seg_length + 1)
        while True:
            x_max =data[x][0] +  (i + 1) * seg_length
            x_min =data[x][0] +  i * seg_length
            if x_max > data[x].iloc[-1]:
                x_max = data[x].iloc[-1]
                to_exit = True
                y = np.full( data[x].size ,y_max)
            ax.add_patch(patches.Rectangle((x_min,0),x_max,y_max,color=(colors_gb[i], colors_gb[i], colors_gb[i], 0.2)))
            if to_exit:
                break
            i = i+1
        
        ax.set_xticklabels(labels)
    return ax
    
#@ray.remote
def plot(result,train_result,hst, name):
    fig = plt.figure(figsize =(25,26))
    #fig = plt.figure()
    fig.suptitle("Meter id = " + str(name))
    #draw the training set
    begin = datetime.datetime.now()
    ax = plt.subplot(5,1,1)
    ax.set_title("loss")
    sns.lineplot(y = np.sqrt(hst[1]), x = [ i for i in range(0,len(hst[1]))])
    ax.set_ylabel("rmse")
    ax.set_xlabel("epoc")
    train_rmse = evaluate(train_result.yhat,train_result.y)
    print ("epoc time " + str(datetime.datetime.now() - begin))
    begin = datetime.datetime.now()
    
    ax = plt.subplot(5,1,2)
    ax.set_title("Training data, RMSE = " + str(train_rmse))
    
    #plt.plot(train_result.ds,train_result.y)
    #plt.plot(train_result.ds,train_result.yhat)
    #plt.plot(train_result.ds,train_result.ydiff)
    sns.lineplot(ax = ax, y = "y", x = "ds", data = train_result)
    sns.lineplot(ax = ax, y = "yhat", x = "ds", data = train_result)
    sns.lineplot(ax = ax, y = "ydiff", x = "ds", data = train_result)


    ax.legend(labels = ["Real","Predicted"])
    #every_nth = train_result.ds.count()//6
    #for n, label in enumerate(ax.xaxis.get_ticklabels()):
    #    if n % every_nth != 0:
    #        label.set_visible(False)


    ax.set_ylabel("ap")
    ax.set_xlabel("create_time")

    #draw the result
    print ("training result time " + str(datetime.datetime.now() - begin))
    begin = datetime.datetime.now()
    
    ax = plt.subplot(5,1,3)
    rmse = evaluate(result.y, result.yhat)
    ax.set_title("RMSE = " + str(rmse))

    sns.lineplot(ax = ax, y="y", x ="ds" , data = result)
    sns.lineplot(ax = ax, y="yhat", x ="ds" , data = result)
    #sns.lineplot(ax = ax, y="ydiff", x ="ds" , data = result)
    
    #plt.plot(result.ds,result.y)
    #plt.plot(result.ds,result.yhat)
    #plt.plot(result.ds,result.ydiff)
    ax.legend(labels = ["Real","Predicted","Difference"])
    every_nth = result.ds.count()//6
    #for n, label in enumerate(ax.xaxis.get_ticklabels()):
    #    if n % every_nth != 0:
    #        label.set_visible(False)
    #ax.set_ylabel("ap")
    #ax.set_xlabel("create_time")
    
    #draw the Error
    print ("prediction time " + str(datetime.datetime.now() - begin))
    begin = datetime.datetime.now()
    ax = plt.subplot(5,1,4)
    rmse = evaluate(result.y, result.yhat)
    ax.set_title("Difference, RMSE = " + str(rmse))
    sns.lineplot(ax = ax, y="ydiff", x ="ds" , data = result)
    
    #plt.plot(result.ds,result.y)
    #plt.plot(result.ds,result.yhat)
    #plt.plot(result.ds,result.ydiff)
    ax.legend(labels = ["Difference"])
    #every_nth = result.ds.count()//6
    #for n, label in enumerate(ax.xaxis.get_ticklabels()):
    #    if n % every_nth != 0:
    #        label.set_visible(False)
    ax.set_ylabel("ap")
    ax.set_xlabel("create_time")
    
    #draw anomaly detection
    print ("prediction diff time " + str(datetime.datetime.now() - begin))
    begin = datetime.datetime.now()
    
    
    ax = plt.subplot(5,1,5)
    #plt.plot(result.ds, result.yhat)
    sns.lineplot(ax = ax, y="yhat",x = "ds", data = result)
    std = result.yhat.std()
    ax.fill_between(result.ds, result.yhat - std * 4,result.yhat + std * 4, color="lightblue", alpha=0.2)

    outside = result[((result["y"] > result.yhat + std * 4) | (result["y"] < result.yhat - std * 4))]
    inside = result[~((result["y"] > result.yhat + std * 4) | (result["y"] < result.yhat - std * 4))]
    ax.set_title("Normal= " + str(inside.yhat.count()) + " Outlier = " + str(outside.yhat.count()) + " Outlier(%) =" +   str(outside.yhat.count() * 100 /result.yhat.count()))
    sns.scatterplot(ax = ax, y = "y",x = "ds", data = inside, color="green")
    sns.scatterplot(ax = ax, y = "y",x = "ds", data = outside, color= "red")
    ax.legend(labels=["Prediction","Threshold", "Normal","Outlier"])
    #every_nth = result.ds.count()//6
    #for n, label in enumerate(ax.xaxis.get_ticklabels()):
    #    if n % every_nth != 0:
    #       label.set_visible(False)
    ax.set_ylabel("ap")
    ax.set_xlabel("create_time")
    #pdf.savefig()
    #plt.clf()
    print ("scatterplot time " + str(datetime.datetime.now() - begin))
    return fig
    #canvas = FigureCanvas(fig)
    #canvas.draw()
    #s, (width, height) = canvas.print_to_buffer()
    #image =  np.fromstring(s, np.uint8).reshape((height, width, 4))
    #return image

def vis_result2(exp_name):
    data_dir = os.path.abspath(os.path.dirname(__file__)) + "/../" + "../data/results/" + exp_name +"/"
    model_dir = os.path.abspath(os.path.dirname(__file__)) + "/../" + "../data/models/" + exp_name + "/"
    report_dir = os.path.abspath(os.path.dirname(__file__)) + "/../" + "../reports/"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    result_files = [f for f in os.listdir(data_dir) if re.match(r'^[0-9]*.csv',f)]
    result_files.sort(key=lambda x: int(x.split(".")[0]))
    print(result_files)
    figures = []
    for file in result_files:
        print(file)
        result = pd.read_csv(data_dir + file )
        train_result = pd.read_csv(data_dir+"train_" + file)
        hst = pd.read_csv(data_dir + "history_" + file, header=None)
        
        name = file.split(".")[0]
        fig = plot(result, train_result, hst, name)
        #fig = plot.remote(result, train_result, hst, name)
        figures.append({"figure": fig,"name":name})
    with PdfPages("../reports/" + exp_name + ".pdf" ) as pdf:    
        for f in figures:
            begin = datetime.datetime.now()
            #buf = ray.get(f["figure"])
            fig = f["figure"]
            pdf.savefig(fig)
            plt.close(fig)
            #print (buf)
            #matplotlib.image.imsave( report_dir + f["name"] +".png",buf)
            print ("saving time " + str(datetime.datetime.now() - begin))
            

def vis_result(exp_name,y_lim = True):
    sns.set_style("dark")
    data_dir = os.path.abspath(os.path.dirname(__file__)) + "/../" + "../data/results/" + exp_name +"/"
    model_dir = os.path.abspath(os.path.dirname(__file__)) + "/../" + "../data/models/" + exp_name + "/"
    report_dir = os.path.abspath(os.path.dirname(__file__)) + "/../" + "../reports/"

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
            hst = pd.read_csv(data_dir + "history_" + file, header=None)

            name = file.split(".")[0]

            fig = plt.figure(figsize =(25,26))
            fig.suptitle("Meter id = " + str(name))
            #draw the training set
            ax = plt.subplot(5,1,1)
            ax.set_title("loss")
            sns.lineplot(y = np.sqrt(hst[1]), x = [ i for i in range(0,len(hst[1]))])

            ax.set_ylabel("rmse")
            ax.set_xlabel("epoc")
            train_rmse = evaluate(train_result.yhat,train_result.y)
            ax = plt.subplot(5,1,2)
            if y_lim:
                ax.set_ylim([-2,2])
            ax.set_title("Training data, RMSE = " + str(train_rmse))
            
            #plt.plot(train_result.ds,train_result.y)
            #plt.plot(train_result.ds,train_result.yhat)
            #plt.plot(train_result.ds,train_result.ydiff)
            sns.lineplot(ax = ax, y = "y", x = "ds", data = train_result)
            sns.lineplot(ax = ax, y = "yhat", x = "ds", data = train_result)
            # sns.lineplot(ax = ax, y = "ydiff", x = "ds", data = train_result)


            ax.legend(labels = ["Real","Predicted"])
            #every_nth = train_result.ds.count()//6
            #for n, label in enumerate(ax.xaxis.get_ticklabels()):
            #    if n % every_nth != 0:
            #        label.set_visible(False)


            ax.set_ylabel("ep_diff")
            ax.set_xlabel("create_time")

            #draw the result
            ax = plt.subplot(5,1,3)
            
            if y_lim : 
                ax.set_ylim([-2,2])
            
            rmse = evaluate(result.y, result.yhat)
            ax.set_title("RMSE = " + str(rmse))

            sns.lineplot(ax = ax, y="y", x ="ds" , data = result)
            sns.lineplot(ax = ax, y="yhat", x ="ds" , data = result)
            #sns.lineplot(ax = ax, y="ydiff", x ="ds" , data = result)
            
            #plt.plot(result.ds,result.y)
            #plt.plot(result.ds,result.yhat)
            #plt.plot(result.ds,result.ydiff)
            ax.legend(labels = ["Real","Predicted"])
            #every_nth = result.ds.count()//6
            #for n, label in enumerate(ax.xaxis.get_ticklabels()):
            #    if n % every_nth != 0:
            #        label.set_visible(False)
            ax.set_ylabel("ep_diff")
            ax.set_xlabel("create_time")
            
            #draw the Error
            ax = plt.subplot(5,1,4)
            if y_lim:
                ax.set_ylim([-2,2])
            
            rmse = evaluate(result.y, result.yhat)
            ax.set_title("Difference, RMSE = " + str(rmse))
            sns.lineplot(ax = ax, y="ydiff", x ="ds" , data = result)

            # ax.fill_between(result.ds, result.threshold_lower, result.threshold_upper, color="lightblue", alpha=0.2)
            if "rolling_std" in result:
                result["upper_border"] = result.rolling_std
                result["lower_border"] = -1.0 * result.rolling_std
                # ax.fill_between(result.ds, lower_border, upper_border, color="tomato", alpha=0.2) 
                sns.lineplot(ax = ax, y="upper_border", x = "ds", linestyle='--', color="coral", data = result)
                sns.lineplot(ax = ax, y="lower_border", x = "ds", linestyle='--', color="coral", data = result)
            
            #plt.plot(result.ds,result.y)
            #plt.plot(result.ds,result.yhat)
            #plt.plot(result.ds,result.ydiff)
            ax.legend(labels = ["Difference", "rollingUpperBoundary", "rollingLowerBoundary"])
            #every_nth = result.ds.count()//6
            #for n, label in enumerate(ax.xaxis.get_ticklabels()):
            #    if n % every_nth != 0:
            #        label.set_visible(False)
            ax.set_ylabel("ep_diff")
            ax.set_xlabel("create_time")
            
            #draw anomaly detection
            ax = plt.subplot(5,1,5)
            if y_lim:
                ax.set_ylim([-2,2])
            
            #plt.plot(result.ds, result.yhat)
            sns.lineplot(ax = ax, y="yhat",x = "ds", data = result, label= "Prediction")
            #std = result.yhat.std()

            ax.fill_between(result.ds, result.threshold_lower, result.threshold_upper, color="lightblue", alpha=0.2)

            if "label" in result:
                true_positive = result[(result["label"]  == True) & (result["label_hat"] == True)]
                true_negative = result[(result["label"]  == False) & (result["label_hat"] == False)]
                false_positive = result[(result["label"]  == False) & (result["label_hat"] == True)]
                false_negative = result[(result["label"]  == True) & (result["label_hat"] == False)]
                
                ax.set_title("Total = " + str(result.label.count()) + ", TP = " + str(true_positive.label.count()) + ", TN = " + str(true_negative.label.count()) + ", FP = " + str(false_positive.label.count()) + ", FN = "  + str(false_negative.label.count()))

                sns.scatterplot(ax = ax, y = "y",x = "ds", data = true_positive, color="green" ,label="True Positive")
                sns.scatterplot(ax = ax, y = "y",x = "ds", data = true_negative, color= "blue", label = "True Negative")
                sns.scatterplot(ax = ax, y = "y",x = "ds", data = false_positive, color= "red", label = "False Positive")
                sns.scatterplot(ax = ax, y = "y",x = "ds", data = false_negative, color= "yellow",label = "False Negative")
                #ax.legend(labels=["Prediction","Threshold", "True Positive","True Negative", "False Positive", "False Negative"])
                ax.legend()
            
            #for sustained only
            elif "label_hat" in result:
                outside = result[(result["label_hat"]  == True)]
                inside = result[(result["label_hat"] == False)]
                ax.fill_between(result.ds, result.threshold_lower, result.threshold_upper, color="lightblue", alpha=0.2 ,label="Threshold")
                #sns.scatterplot(ax = ax, y = "y",x = "ds", data = inside, color= "green", label = "Inside")
                sns.scatterplot(ax = ax, y = "y",x = "ds", data = outside, color="red",label="Outlier")
                ax.legend()
                
            else:
                
                std = result.y.std()
                outside = result[((result["y"] > result.yhat + std * 4) | (result["y"] < result.yhat - std * 4))]
                inside = result[~((result["y"] > result.yhat + std * 4) | (result["y"] < result.yhat - std * 4))]
                ax.set_title("Normal= " + str(inside.yhat.count()) + " Outlier = " + str(outside.yhat.count()) + " Outlier(%) =" +   str(outside.yhat.count() * 100 /result.yhat.count()))
                
                sns.scatterplot(ax = ax, y = "y",x = "ds", data = outside, color="red")
                sns.scatterplot(ax = ax, y = "y",x = "ds", data = inside, color= "green")
      
            
            #every_nth = result.ds.count()//6
            #for n, label in enumerate(ax.xaxis.get_ticklabels()):
            #    if n % every_nth != 0:
            #       label.set_visible(False)
            ax.set_ylabel("ep_diff")
            ax.set_xlabel("create_time")
            pdf.savefig()
            plt.clf()
            plt.close()



